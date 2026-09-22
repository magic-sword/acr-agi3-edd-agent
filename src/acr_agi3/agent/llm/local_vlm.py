"""Google ADK 2.0 互換のローカル VLM (Vision-Language Model) アダプター.

Qwen2.5-VL 等のオープンマルチモーダルモデルをインプロセスでロードし、
ARC のグリッド画像とテキスト (SKILL.md やプロンプト) を統合して推論します。
インターネット接続なしの完全オフライン動作に対応しています。
"""

import inspect
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai.types import Content, Part, FunctionCall, FunctionResponse
from PIL import Image
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

# リアルタイム監視用ログファイル (logs/agent_live_execution.log)
LIVE_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "logs" / "agent_live_execution.log"
LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(LIVE_LOG_PATH, mode="a", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)
logger.setLevel(logging.INFO)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def resolve_model_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Qwen2.5-VL モデル重みディレクトリを自動解決."""
    if custom_path == "mock":
        return None
    if custom_path and custom_path not in ("auto", "default"):
        p = Path(custom_path)
        if p.exists() and (p / "config.json").exists():
            return p

    candidates = [
        Path("/kaggle/input/qwen2-5-vl-3b-instruct"),
        Path("/kaggle/input/qwen2.5-vl-3b-instruct"),
        Path("/workspace/models/Qwen2.5-VL-3B-Instruct"),
        REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return c
    return None


class LocalQwenVL(BaseLlm):
    """Google ADK 2.0 向けローカル Qwen2.5-VL マルチモーダル推論アダプター."""

    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _generate_fn: Optional[Callable[..., str]] = PrivateAttr(default=None)

    # プロセス内シングルトンキャッシュ (再ロードによる VRAM 浪費・時間遅延を防止)
    _shared_model: Any = None
    _shared_processor: Any = None
    _shared_model_path: Optional[str] = None

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        generate_fn: Optional[Callable[..., str]] = None,
        device: str = "cuda",
        torch_dtype: Any = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        allow_cpu_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        """初期化.

        Args:
            model_name_or_path: モデル名またはローカル重みパス (省略時は自動検出)
            generate_fn: テスト・モック用の生成関数 (fn(prompt, images=None) -> output_text)
            device: 実行デバイス ('cuda', 'cpu')
            torch_dtype: データ型 (torch.bfloat16, torch.float16等)
            load_in_8bit: 8bit量子化
            load_in_4bit: 4bit量子化
            allow_cpu_fallback: GPU が使用不能な場合に CPU へのフォールバックを許可するか
        """
        resolved = resolve_model_path(model_name_or_path)
        actual_path = str(resolved) if resolved else (model_name_or_path or "mock")
        super().__init__(model=actual_path, **kwargs)
        self._generate_fn = generate_fn

        if generate_fn is None and actual_path != "mock" and resolved is not None:
            self._init_vlm(
                model_name_or_path=str(resolved),
                device=device,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                allow_cpu_fallback=allow_cpu_fallback,
            )
        elif generate_fn is None and actual_path != "mock":
            logger.info("LocalQwenVL: Model weights not found, operating in safe mock mode.")

    def _init_vlm(
        self,
        model_name_or_path: str,
        device: str,
        torch_dtype: Any,
        load_in_8bit: bool,
        load_in_4bit: bool,
        allow_cpu_fallback: bool = False,
    ) -> None:
        """Qwen2.5-VL モデルとプロセッサをオフライン初期化 (キャッシュ再利用)."""
        if (
            LocalQwenVL._shared_model is not None
            and LocalQwenVL._shared_processor is not None
            and LocalQwenVL._shared_model_path == model_name_or_path
        ):
            self._model = LocalQwenVL._shared_model
            self._processor = LocalQwenVL._shared_processor
            return

        import torch

        # GPU が要求されているのに利用できない場合、無断フォールバックを防止
        if device.startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError(
                    f"GPU (CUDA) が指定されていますが、torch.cuda.is_available() が False です。\n"
                    f"CPU 推論は 30〜40 倍遅延し、重大な性能低下（評価に10時間以上）を招くため自動フォールバックは無効化されています。\n"
                    f"NVML エラーや GPU パススルーを確認するか、意図的に CPU で動かす場合は allow_cpu_fallback=True または device='cpu' を指定してください。"
                )
            logger.warning(
                "CUDA requested but unavailable. Falling back to CPU because allow_cpu_fallback=True."
            )

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "local_files_only": True,  # 完全オフラインロード
            }

            if load_in_4bit or load_in_8bit:
                model_kwargs["device_map"] = "auto"
                if load_in_4bit:
                    model_kwargs["load_in_4bit"] = True
                elif load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
            else:
                model_kwargs["device_map"] = device if torch.cuda.is_available() else "cpu"

            self._processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **model_kwargs,
            )
            LocalQwenVL._shared_model = self._model
            LocalQwenVL._shared_processor = self._processor
            LocalQwenVL._shared_model_path = model_name_or_path
            logger.info(f"Loaded Qwen2.5-VL model from {model_name_or_path} successfully on {self._model.device}.")
        except Exception as e:
            logger.error(
                f"Failed to initialize Qwen2.5-VL for '{model_name_or_path}': {e}."
            )
            raise RuntimeError(
                f"Failed to initialize Qwen2.5-VL on device '{device}' for '{model_name_or_path}': {e}"
            ) from e

    @staticmethod
    def _declaration_to_schema(fd: Any) -> Dict[str, Any]:
        """ADK / Google GenAI の FunctionDeclaration を OpenAI / Qwen 互換の JSON Schema 辞書に変換."""
        name = getattr(fd, "name", "")
        desc = getattr(fd, "description", "") or ""
        params: Dict[str, Any] = {"type": "object", "properties": {}}

        # ADK skill-management tools use parameters_json_schema; FunctionTool
        # may instead use parameters. Preserve both official declaration forms.
        raw_params = getattr(fd, "parameters_json_schema", None)
        if raw_params is None:
            raw_params = getattr(fd, "parameters", None)
        if raw_params is not None:
            if isinstance(raw_params, dict):
                params = raw_params
            elif hasattr(raw_params, "model_dump"):
                try:
                    params = raw_params.model_dump(exclude_none=True, by_alias=True)
                except Exception:
                    params = raw_params.model_dump(exclude_none=True)
            elif hasattr(raw_params, "to_json_dict"):
                params = raw_params.to_json_dict()
            elif hasattr(raw_params, "properties"):
                props = {}
                for k, v in getattr(raw_params, "properties", {}).items():
                    p_type = getattr(v, "type", "string")
                    p_desc = getattr(v, "description", "")
                    props[k] = {"type": str(p_type).lower(), "description": p_desc}
                params = {
                    "type": "object",
                    "properties": props,
                    "required": list(getattr(raw_params, "required", []) or []),
                }

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params,
            },
        }

    def _format_tools_for_qwen(self, tools: List[Any]) -> str:
        """ADK ツールリストから Qwen 公式の <tools> ... </tools> プロンプトブロックを構築."""
        declarations: List[Dict[str, Any]] = []
        for t in tools:
            # 1. google.genai.types.Tool (function_declarations)
            if hasattr(t, "function_declarations") and t.function_declarations:
                for fd in t.function_declarations:
                    declarations.append(self._declaration_to_schema(fd))
            # 2. 単一の FunctionDeclaration
            elif hasattr(t, "name") and (hasattr(t, "parameters") or hasattr(t, "description")):
                declarations.append(self._declaration_to_schema(t))
            # 3. 辞書形式
            elif isinstance(t, dict):
                if "function" in t:
                    declarations.append(t)
                elif "name" in t:
                    declarations.append({"type": "function", "function": t})

        if not declarations:
            return ""

        tools_json_lines = "\n".join(json.dumps(d, ensure_ascii=False) for d in declarations)
        return (
            "\n# Tools\n\n"
            "You may call one or more functions to assist with the user specification.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json_lines}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": <args-json-object>}\n'
            "</tool_call>\n"
        )

    def _extract_text_and_images(self, llm_request: LlmRequest) -> Tuple[str, List[Image.Image]]:
        """LlmRequest からテキストプロンプトと画像リストを抽出."""
        parts_text: List[str] = []
        images: List[Image.Image] = []

        # システム指示 (SKILL.md や System Prompt)
        if llm_request.config and hasattr(llm_request.config, "system_instruction"):
            sys_inst = llm_request.config.system_instruction
            if isinstance(sys_inst, str):
                parts_text.append(f"System: {sys_inst}\n")
            elif hasattr(sys_inst, "parts"):
                for p in sys_inst.parts:
                    if hasattr(p, "text") and p.text:
                        parts_text.append(f"System: {p.text}\n")

        # ツール定義の注入 (ADK ➔ Qwen ネイティブ形式)
        if llm_request.config and hasattr(llm_request.config, "tools") and llm_request.config.tools:
            tools_block = self._format_tools_for_qwen(llm_request.config.tools)
            if tools_block:
                parts_text.append(tools_block)

        for content in llm_request.contents or []:
            for part in getattr(content, "parts", []):
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    fc_args = fc.args if isinstance(fc.args, dict) else {}
                    parts_text.append(
                        f"<tool_call>\n{json.dumps({'name': fc.name, 'arguments': fc_args}, ensure_ascii=False)}\n</tool_call>"
                    )
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    # Observation tools return real images on demand. Keep image
                    # bytes out of the text prompt while preserving frame labels.
                    response = fr.response
                    if isinstance(response, dict) and "screen_observation" in response:
                        import base64
                        import io

                        observation = response["screen_observation"]
                        labels = []
                        observation_images = []
                        for item in observation.get("images", []):
                            img = Image.open(io.BytesIO(base64.b64decode(item["data"]))).convert("RGB")
                            observation_images.append(img)
                            labels.append({k: v for k, v in item.items() if k != "data"})
                        # The most recent explicit look replaces older visual
                        # context; a before/after pair stays together and ordered.
                        images = observation_images
                        response = {"screen_observation": {
                            "current_frame_id": observation["current_frame_id"], "images": labels,
                        }}
                    parts_text.append(
                        f"<tool_response>\n{json.dumps(response, ensure_ascii=False, default=str)}\n</tool_response>"
                    )
                # 画像バイトデータまたは PIL Image の取得
                elif hasattr(part, "inline_data") and part.inline_data:
                    import io

                    raw_bytes = part.inline_data.data
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    images.append(img)

        prompt = "\n".join(parts_text)
        return prompt, images

    def _detect_tool_call(
        self,
        output: Any,
        available_tools: Optional[Set[str]] = None,
        loaded_skills: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        raw_call: Optional[Tuple[str, Dict[str, Any]]] = None
        if isinstance(output, Part) and hasattr(output, "function_call") and output.function_call:
            raw_call = output.function_call.name, dict(output.function_call.args or {})

        elif isinstance(output, dict):
            if "name" in output and "args" in output:
                raw_call = output["name"], output.get("args") or {}
            elif "tool_call" in output:
                tc = output["tool_call"]
                if isinstance(tc, str):
                    raw_call = tc, output.get("tool_args", {})
                elif isinstance(tc, dict) and "name" in tc:
                    raw_call = tc["name"], tc.get("args") or tc.get("arguments") or {}

        elif isinstance(output, str):
            # 1. <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
            m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", output, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    name = data.get("name")
                    args = data.get("arguments") or data.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if name:
                        raw_call = name, args
                except Exception:
                    pass

            # 3. 平文テキストからのツール呼び出し検出 (モデルがタグを省略した場合の防護)
            if not raw_call:
                m_skill = re.search(
                    r"(?:load|use)\s+['\"]?([a-z\-]+(?:inspector|controller|deliberation|planner|engine|compiler|tester|guard))['\"]?",
                    output,
                    re.IGNORECASE,
                )
                if m_skill:
                    raw_call = "load_skill", {"skill_name": m_skill.group(1).lower()}

            if not raw_call:
                m_act = re.search(
                    r"(?:step_action|call\s+step_action)\s*[:\(]\s*(?:action\s*=\s*)?['\"]?(ACTION[0-9]|UP|DOWN|LEFT|RIGHT|[0-9]+)['\"]?",
                    output,
                    re.IGNORECASE,
                )
                if m_act:
                    raw_call = "step_action", {"action": m_act.group(1).upper()}

            if not raw_call:
                m_click = re.search(
                    r"(?:click_at|call\s+click_at)\s*[:\(]\s*(?:x\s*=\s*)?([0-9]+)\s*,\s*(?:y\s*=\s*)?([0-9]+)",
                    output,
                    re.IGNORECASE,
                )
                if m_click:
                    raw_call = "click_at", {"x": int(m_click.group(1)), "y": int(m_click.group(2))}

        if raw_call:
            return self._normalize_tool_call(
                raw_call[0],
                raw_call[1],
                available_tools=available_tools,
                loaded_skills=loaded_skills,
            )
        return None

    def _extract_available_tools(self, llm_request: Optional[LlmRequest]) -> Set[str]:
        """現在利用可能な関数宣言ツール名を抽出."""
        if not llm_request or not llm_request.config or not llm_request.config.tools:
            return set()
        names: Set[str] = set()
        for t in llm_request.config.tools:
            if hasattr(t, "function_declarations") and t.function_declarations:
                for fd in t.function_declarations:
                    if hasattr(fd, "name"):
                        names.add(fd.name)
            elif hasattr(t, "name"):
                names.add(t.name)
            elif isinstance(t, dict):
                if "function" in t and "name" in t["function"]:
                    names.add(t["function"]["name"])
                elif "name" in t:
                    names.add(t["name"])
        return names

    def _extract_loaded_skills(self, llm_request: Optional[LlmRequest]) -> Set[str]:
        """会話履歴から既にロードされたスキル名を抽出."""
        if not llm_request or not llm_request.contents:
            return set()
        loaded = set()
        for c in llm_request.contents:
            for p in getattr(c, "parts", []):
                if hasattr(p, "function_call") and p.function_call and p.function_call.name == "load_skill":
                    args = p.function_call.args or {}
                    if "skill_name" in args:
                        loaded.add(str(args["skill_name"]).replace("_", "-"))
        return loaded

    def _normalize_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        available_tools: Optional[Set[str]] = None,
        loaded_skills: Optional[Set[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Qwen2.5-VL のモデル特有の語彙揺らぎやスキル/ツールの混同を正規化."""
        # 1. visual_inspector / visual-inspector 呼び出しの適応
        if name in ("visual_inspector", "visual-inspector"):
            if "view" in args or not args:
                name = "observe_screen"
                args = {"view": args.get("view", "current")}
            else:
                return "load_skill", {"skill_name": "visual-inspector"}

        # 2. game_controller / game-controller 呼び出しの適応
        if name in ("game_controller", "game-controller"):
            if "action" in args or "action_id" in args:
                name = "step_action"
                if "action" in args and "action_id" not in args:
                    act_val = args["action"]
                    if isinstance(act_val, str) and act_val.startswith("ACTION"):
                        try:
                            args["action_id"] = int(act_val.replace("ACTION", ""))
                        except ValueError:
                            args["action_id"] = 1
                    elif isinstance(act_val, int):
                        args["action_id"] = act_val
            elif "x" in args and "y" in args:
                name = "click_at"
            else:
                return "load_skill", {"skill_name": "game-controller"}

        # step_action 引数の正規化 (action -> action_id)
        if name == "step_action" and "action" in args and "action_id" not in args:
            act_val = args["action"]
            if isinstance(act_val, str) and act_val.startswith("ACTION"):
                try:
                    args["action_id"] = int(act_val.replace("ACTION", ""))
                except ValueError:
                    args["action_id"] = 1
            elif isinstance(act_val, int):
                args["action_id"] = act_val

        # 3. load_skill_resource で画像やファイルを読もうとした場合の安全補正 (リトライループ防止)
        if name == "load_skill_resource":
            name = "observe_screen"
            args = {"view": "current"}

        # 4. load_skill のスキル名正規化 (アンダースコア/ハイフン)
        if name == "load_skill" and "skill_name" in args:
            args["skill_name"] = str(args["skill_name"]).replace("_", "-")
            return name, args

        # 5. Progressive Disclosure 防護: 未開放ツールの初回自動スキルロード変換
        # 注意: 既に一度ロード済みなのにツールが存在しない場合は、モード制約による不許可なので再変換してはならない（無限ループ防止）
        loaded = loaded_skills or set()
        if available_tools and "load_skill" in available_tools:
            # observe_screen がまだ開示されていない場合 -> visual-inspector をロード
            if name == "observe_screen" and "observe_screen" not in available_tools:
                if "visual-inspector" not in loaded:
                    return "load_skill", {"skill_name": "visual-inspector"}

            # causal deliberation ツールがまだ開示されていない場合 -> causal-deliberation をロード
            causal_tools = {
                "set_goal",
                "need_causal_knowledge",
                "need_experiment",
                "assess_result",
                "resolve_question",
                "plan_actions",
                "continue_plan",
                "answer_visible_question",
                "use_known_rules",
            }
            if name in causal_tools and name not in available_tools:
                if "causal-deliberation" not in loaded:
                    return "load_skill", {"skill_name": "causal-deliberation"}

            # step_action / click_at / reset_game がまだ開示されていない場合 -> game-controller をロード
            if name in ("step_action", "click_at", "reset_game") and name not in available_tools:
                if "game-controller" not in loaded:
                    return "load_skill", {"skill_name": "game-controller"}

        # 6. REVIEW モード安全適応: assess_result が利用可能で resolve_question が未開放の状況で
        # モデルが resolve_question を呼んだ場合、assess_result へ正規化して進行を保証
        if (
            name == "resolve_question"
            and available_tools
            and "assess_result" in available_tools
            and "resolve_question" not in available_tools
        ):
            evidence = args.get("effect") or args.get("evidence") or args.get("answer") or "Observed piece movement"
            outcome = "supported" if "not" not in str(evidence).lower() and "fail" not in str(evidence).lower() else "refuted"
            b_id = int(args.get("before_frame_id") or 1)
            a_id = int(args.get("after_frame_id") or (b_id + 1))
            return "assess_result", {
                "outcome": outcome,
                "evidence": str(evidence),
                "before_frame_id": b_id,
                "after_frame_id": a_id,
            }

        return name, args

    def _build_qwen_messages(
        self, llm_request: LlmRequest, initial_images: List[Image.Image]
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Construct official Qwen ChatML messages with system, user, assistant, and tool turns."""
        messages: List[Dict[str, Any]] = []
        all_images: List[Image.Image] = list(initial_images)

        # 1. System Prompt & Tools
        system_parts: List[str] = []
        if llm_request.config and hasattr(llm_request.config, "system_instruction"):
            sys_inst = llm_request.config.system_instruction
            if isinstance(sys_inst, str):
                system_parts.append(sys_inst.strip())
            elif hasattr(sys_inst, "parts"):
                for p in sys_inst.parts:
                    if hasattr(p, "text") and p.text:
                        system_parts.append(p.text.strip())

        if llm_request.config and hasattr(llm_request.config, "tools") and llm_request.config.tools:
            tools_block = self._format_tools_for_qwen(llm_request.config.tools)
            if tools_block:
                system_parts.append(tools_block.strip())

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 2. Process conversation turns from llm_request.contents
        for content in llm_request.contents or []:
            role = getattr(content, "role", "user")

            for part in getattr(content, "parts", []):
                # A. Tool call from model
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    fc_args = fc.args if isinstance(fc.args, dict) else {}
                    fc_str = json.dumps({"name": fc.name, "arguments": fc_args}, ensure_ascii=False)
                    messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{fc_str}\n</tool_call>",
                    })

                # B. Tool response
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response = fr.response
                    if isinstance(response, dict) and "screen_observation" in response:
                        import base64
                        import io

                        obs = response["screen_observation"]
                        labels = []
                        for item in obs.get("images", []):
                            raw = base64.b64decode(item["data"])
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                            all_images.append(img)
                            labels.append({k: v for k, v in item.items() if k != "data"})
                        cleaned_response = dict(response)
                        cleaned_response["screen_observation"] = {
                            "current_frame_id": obs.get("current_frame_id"),
                            "images": labels,
                        }
                        response = cleaned_response
                    resp_str = json.dumps(response, ensure_ascii=False, default=str)
                    messages.append({
                        "role": "tool",
                        "name": fr.name,
                        "content": resp_str,
                    })

                # C. Regular text
                elif hasattr(part, "text") and part.text:
                    if role == "model":
                        messages.append({"role": "assistant", "content": part.text})
                    else:
                        messages.append({"role": "user", "content": part.text})

                # D. Image data
                elif hasattr(part, "inline_data") and part.inline_data:
                    import io

                    raw_bytes = part.inline_data.data
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    all_images.append(img)

        # Attach any images to the latest user message or create one if needed
        if all_images:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    items: List[Dict[str, Any]] = []
                    for img in all_images:
                        items.append({"type": "image", "image": img})
                    if isinstance(user_content, str):
                        items.append({"type": "text", "text": user_content})
                    elif isinstance(user_content, list):
                        items.extend([it for it in user_content if it.get("type") != "image"])
                    msg["content"] = items
                    break

        return messages, all_images

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """マルチモーダル (画像 + テキスト) リクエストを推論処理."""
        prompt, images = self._extract_text_and_images(llm_request)

        # 1. カスタム生成関数 (モック・テスト用) の場合
        if self._generate_fn is not None:
            if inspect.iscoroutinefunction(self._generate_fn):
                generated = await self._generate_fn(prompt, images=images)
            else:
                generated = self._generate_fn(prompt, images=images)

            available_tools = self._extract_available_tools(llm_request)
            loaded_skills = self._extract_loaded_skills(llm_request)
            tc = self._detect_tool_call(
                generated, available_tools=available_tools, loaded_skills=loaded_skills
            )
            if tc:
                tool_name, tool_args = tc
                if not isinstance(tool_args, dict):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}
                part = Part.from_function_call(name=tool_name, args=tool_args)
                logger.info(f"[LocalQwenVL] Native Tool Call Detected (mock): {tool_name}({tool_args})")
            elif isinstance(generated, Part):
                part = generated
            else:
                part = Part.from_text(text=str(generated))

            yield LlmResponse(
                content=Content(role="model", parts=[part]),
                turn_complete=True,
            )
            return

        # 2. Qwen2.5-VL モデルの場合
        if self._model is not None and self._processor is not None:
            import torch
            from qwen_vl_utils import process_vision_info

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            messages, vision_images = self._build_qwen_messages(llm_request, images)

            # コンテキスト安全制限 (OOM 防止): システム指示 + 直近 10 メッセージ
            if len(messages) > 12:
                sys_msgs = [m for m in messages if m["role"] == "system"]
                other_msgs = [m for m in messages if m["role"] != "system"][-10:]
                messages = sys_msgs + other_msgs

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                generated_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                if "</tool_call>" in generated_text:
                    end_idx = generated_text.find("</tool_call>") + len("</tool_call>")
                    generated_text = generated_text[:end_idx]
                logger.info(f"[LocalQwenVL] raw generated_text: {generated_text!r}")

            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # ARC-AGI-3 動的ゲーム環境向けモックアクション生成
            avail_match = re.search(r"Available Actions:\s*([^\n]+)", prompt)
            avail_actions = (
                [a.strip() for a in avail_match.group(1).split(",")]
                if avail_match
                else ["ACTION1"]
            )
            first_act = avail_actions[0] if avail_actions else "ACTION1"
            generated_text = (
                f"Visual Inspection Analysis: I observe the colored grid board and identified affordances. "
                f"I decide to call step_action(action='{first_act}', reasoning='Navigating toward active objective')."
            )

        available_tools = self._extract_available_tools(llm_request)
        loaded_skills = self._extract_loaded_skills(llm_request)
        tc = self._detect_tool_call(
            generated_text, available_tools=available_tools, loaded_skills=loaded_skills
        )
        if tc:
            tool_name, tool_args = tc
            if not isinstance(tool_args, dict):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            part = Part.from_function_call(name=tool_name, args=tool_args)
            logger.info(f"[LocalQwenVL] Native Tool Call Detected (model): {tool_name}({tool_args})")
        else:
            part = Part.from_text(text=generated_text)

        response_content = Content(
            role="model",
            parts=[part],
        )
        yield LlmResponse(
            content=response_content,
            turn_complete=True,
        )
