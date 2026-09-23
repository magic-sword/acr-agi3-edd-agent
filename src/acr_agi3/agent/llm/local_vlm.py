"""Google ADK 2.0 互換のローカル VLM (Vision-Language Model) アダプター.

Qwen2.5-VL 等のオープンマルチモーダルモデルをインプロセスでロードし、
ARC のグリッド画像とテキスト (SKILL.md やプロンプト) を統合して推論します。
インターネット接続なしの完全オフライン動作に対応しています。
"""

import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai.types import Content, Part
from PIL import Image
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

# Logging destinations belong to the host; Dataset source directories are read-only.

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
    max_input_tokens: int = 24576
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _generate_fn: Optional[Callable[..., str]] = PrivateAttr(default=None)
    _inference_trace: list[dict] = PrivateAttr(default_factory=list)
    _last_output_error: str = PrivateAttr(default="")

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
            raise FileNotFoundError("Local VLM weights not found; refusing a mock benchmark run.")

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
                    "GPU (CUDA) が指定されていますが、torch.cuda.is_available() が False です。\n"
                    "CPU 推論は 30〜40 倍遅延し、重大な性能低下（評価に10時間以上）を招くため自動フォールバックは無効化されています。\n"
                    "NVML エラーや GPU パススルーを確認するか、意図的に CPU で動かす場合は allow_cpu_fallback=True または device='cpu' を指定してください。"
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
            logger.info(
                f"Loaded Qwen2.5-VL model from {model_name_or_path} successfully on {self._model.device}."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2.5-VL for '{model_name_or_path}': {e}.")
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
        """Use the same conversation and visual context for tests and local inference."""
        messages, images = self._build_qwen_messages(llm_request)
        text = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                text.append(content)
            else:
                text.extend(item["text"] for item in content if item["type"] == "text")
        return "\n".join(text), images

    def _detect_tool_call(
        self,
        output: Any,
        available_tools: Optional[Set[str]] = None,
        loaded_skills: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Decode one explicit function call, without inventing or changing its intent."""
        if isinstance(output, Part) and output.function_call:
            data = {
                "name": output.function_call.name,
                "args": dict(output.function_call.args or {}),
            }
        elif isinstance(output, dict):
            data = output
        elif isinstance(output, str):
            calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.DOTALL)
            if len(calls) != 1:
                return None
            try:
                data = json.loads(calls[0])
            except (ValueError, TypeError):
                return None
        else:
            return None
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        args = data.get("arguments", data.get("args", {}))
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        if available_tools and name not in available_tools:
            return None
        return name, args

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

    def _build_qwen_messages(
        self, llm_request: LlmRequest
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Construct official Qwen ChatML messages with system, user, assistant, and tool turns."""
        messages: List[Dict[str, Any]] = []
        all_images: List[Image.Image] = []
        image_labels: list[dict] = []
        latest_guidance = ""

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
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"<tool_call>\n{fc_str}\n</tool_call>",
                        }
                    )

                # B. Tool response
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response = fr.response
                    if isinstance(response, dict) and response.get("mode_guidance"):
                        latest_guidance = response["mode_guidance"]
                    if isinstance(response, dict) and "screen_observation" in response:
                        import base64
                        import io

                        obs = response["screen_observation"]
                        labels = []
                        all_images = []
                        for item in obs.get("images", []):
                            raw = base64.b64decode(item["data"])
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                            all_images.append(img)
                            labels.append({k: v for k, v in item.items() if k != "data"})
                        image_labels = labels
                        cleaned_response = dict(response)
                        cleaned_response["screen_observation"] = {
                            "current_frame_id": obs.get("current_frame_id"),
                            "images": labels,
                        }
                        response = cleaned_response
                    resp_str = json.dumps(response, ensure_ascii=False, default=str)
                    messages.append(
                        {
                            "role": "tool",
                            "name": fr.name,
                            "content": resp_str,
                        }
                    )

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

        # Keep the latest explicit observation together with its ordered labels.
        # Do not duplicate images already decoded by another conversion path.
        if all_images or latest_guidance:
            items = []
            for index, img in enumerate(all_images):
                label = image_labels[index] if index < len(image_labels) else {"image_index": index}
                items.append({"type": "text", "text": json.dumps(label)})
                items.append({"type": "image", "image": img})
            if latest_guidance:
                items.append(
                    {
                        "type": "text",
                        "text": (
                            "Current cognitive guidance (supersedes earlier mode instructions):\n"
                            + latest_guidance
                            + "\nCall one currently exposed tool to advance this task."
                        ),
                    }
                )
            messages.append({"role": "user", "content": items})

        return messages, all_images

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """マルチモーダル (画像 + テキスト) リクエストを推論処理."""
        self._last_output_error = ""
        prompt, images = self._extract_text_and_images(llm_request)

        # 1. カスタム生成関数 (モック・テスト用) の場合
        if self._generate_fn is not None:
            if inspect.iscoroutinefunction(self._generate_fn):
                generated = await self._generate_fn(prompt, images=images)
            else:
                generated = self._generate_fn(prompt, images=images)
            self._inference_trace.append({"image_count": len(images), "raw_output": str(generated)})

            available_tools = self._extract_available_tools(llm_request)
            tc = self._detect_tool_call(generated, available_tools=available_tools)
            if tc:
                tool_name, tool_args = tc
                if not isinstance(tool_args, dict):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}
                part = Part.from_function_call(name=tool_name, args=tool_args)
                logger.info(
                    f"[LocalQwenVL] Native Tool Call Detected (mock): {tool_name}({tool_args})"
                )
            elif isinstance(generated, Part) and not generated.function_call:
                part = generated
            else:
                self._last_output_error = (
                    "No tool executed. Use one currently exposed tool schema. "
                    f"Available tools: {sorted(available_tools)}. "
                    "To open a skill, call load_skill with its skill_name argument."
                )
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

            messages, vision_images = self._build_qwen_messages(llm_request)

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

            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens > self.max_input_tokens:
                raise ValueError(
                    f"Visual context exceeds input budget: {input_tokens} > {self.max_input_tokens}; "
                    "no observations or skill instructions were silently discarded."
                )
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
                self._inference_trace.append(
                    {
                        "image_count": len(vision_images),
                        "input_tokens": input_tokens,
                        "output_tokens": len(generated_ids_trimmed[0]),
                        "raw_output": generated_text,
                    }
                )
                logger.info(f"[LocalQwenVL] raw generated_text: {generated_text!r}")

            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise RuntimeError("No local model or explicit test generate_fn is configured.")

        available_tools = self._extract_available_tools(llm_request)
        tc = self._detect_tool_call(generated_text, available_tools=available_tools)
        if tc:
            tool_name, tool_args = tc
            if not isinstance(tool_args, dict):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            part = Part.from_function_call(name=tool_name, args=tool_args)
            logger.info(
                f"[LocalQwenVL] Native Tool Call Detected (model): {tool_name}({tool_args})"
            )
        else:
            self._last_output_error = (
                "No tool executed. Use one currently exposed tool schema. "
                f"Available tools: {sorted(available_tools)}. "
                "To open a skill, call load_skill with its skill_name argument."
            )
            part = Part.from_text(text=generated_text)

        response_content = Content(
            role="model",
            parts=[part],
        )
        yield LlmResponse(
            content=response_content,
            turn_complete=True,
        )
