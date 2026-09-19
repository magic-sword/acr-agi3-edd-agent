"""Google ADK 2.0 互換のローカル VLM (Vision-Language Model) アダプター.

Qwen2.5-VL 等のオープンマルチモーダルモデルをインプロセスでロードし、
ARC のグリッド画像とテキスト (SKILL.md やプロンプト) を統合して推論します。
インターネット接続なしの完全オフライン動作に対応しています。
"""

import inspect
import json
import logging
from pathlib import Path
import re
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def resolve_model_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Qwen2.5-VL モデル重みディレクトリを自動解決."""
    if custom_path and custom_path not in ("auto", "default", "mock"):
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

        try:
            import torch
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
            logger.info(f"Loaded Qwen2.5-VL model from {model_name_or_path} successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Qwen2.5-VL for '{model_name_or_path}': {e}. "
                "Fallback to mock/generate_fn mode."
            )

    @staticmethod
    def _declaration_to_schema(fd: Any) -> Dict[str, Any]:
        """ADK / Google GenAI の FunctionDeclaration を OpenAI / Qwen 互換の JSON Schema 辞書に変換."""
        name = getattr(fd, "name", "")
        desc = getattr(fd, "description", "") or ""
        params: Dict[str, Any] = {"type": "object", "properties": {}}

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
                    parts_text.append(
                        f"<tool_response>\n{json.dumps(fr.response, ensure_ascii=False, default=str)}\n</tool_response>"
                    )
                # 画像バイトデータまたは PIL Image の取得
                elif hasattr(part, "inline_data") and part.inline_data:
                    import io

                    raw_bytes = part.inline_data.data
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    images.append(img)

        prompt = "\n".join(parts_text)
        return prompt, images

    def _detect_tool_call(self, output: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        """出力からツール呼び出し (Function Call) を検出・パース."""
        if isinstance(output, Part) and hasattr(output, "function_call") and output.function_call:
            return output.function_call.name, dict(output.function_call.args or {})

        if isinstance(output, dict):
            if "name" in output and "args" in output:
                return output["name"], output.get("args") or {}
            if "tool_call" in output:
                tc = output["tool_call"]
                if isinstance(tc, str):
                    return tc, output.get("tool_args", {})
                if isinstance(tc, dict) and "name" in tc:
                    return tc["name"], tc.get("args") or tc.get("arguments") or {}

        if isinstance(output, str):
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
                        if name == "load_skill":
                            raw_sn = str(args.get("skill_name", "")).strip().upper()
                            if raw_sn in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "UP", "DOWN", "LEFT", "RIGHT", "RESET"]:
                                logger.info(f"[LocalQwenVL] Auto-correcting misrouted load_skill('{raw_sn}') to step_action(action='{raw_sn}')")
                                return "step_action", {"action": raw_sn, "reasoning": f"Executing {raw_sn}"}
                        return name, args
                except Exception:
                    pass

            # 2. ```json { "tool_call": ... } ```
            m_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
            if m_json:
                try:
                    data = json.loads(m_json.group(1))
                    if "tool_call" in data:
                        tc = data["tool_call"]
                        if isinstance(tc, str):
                            return tc, data.get("tool_args", {})
                        if isinstance(tc, dict) and "name" in tc:
                            return tc["name"], tc.get("args") or tc.get("arguments") or {}
                except Exception:
                    pass

        return None

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

            tc = self._detect_tool_call(generated)
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

            content_items: List[Dict[str, Any]] = []
            if images:
                content_items.append({"type": "image", "image": images[-1]})
            content_items.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content_items}]
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
                    max_new_tokens=256,
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

        tc = self._detect_tool_call(generated_text)
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
