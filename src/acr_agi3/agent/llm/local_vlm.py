"""Google ADK 2.0 互換のローカル VLM (Vision-Language Model) アダプター.

Qwen2.5-VL 等のオープンマルチモーダルモデルをインプロセスでロードし、
ARC のグリッド画像とテキスト (SKILL.md やプロンプト) を統合して推論します。
インターネット接続なしの完全オフライン動作に対応しています。
"""

import inspect
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai.types import Content, Part
from PIL import Image
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class LocalQwenVL(BaseLlm):
    """Google ADK 2.0 向けローカル Qwen2.5-VL マルチモーダル推論アダプター."""

    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _generate_fn: Optional[Callable[..., str]] = PrivateAttr(default=None)

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        generate_fn: Optional[Callable[..., str]] = None,
        device: str = "cuda",
        torch_dtype: Any = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        """初期化.

        Args:
            model_name_or_path: モデル名またはローカルの重みディレクトリパス
            generate_fn: テスト・モック用の生成関数 (fn(prompt, images=None) -> output_text)
            device: 実行デバイス ('cuda', 'cpu')
            torch_dtype: データ型 (torch.bfloat16, torch.float16等)
            load_in_8bit: 8bit量子化
            load_in_4bit: 4bit量子化
        """
        super().__init__(model=model_name_or_path, **kwargs)
        self._generate_fn = generate_fn

        if generate_fn is None and model_name_or_path != "mock":
            self._init_vlm(
                model_name_or_path=model_name_or_path,
                device=device,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

    def _init_vlm(
        self,
        model_name_or_path: str,
        device: str,
        torch_dtype: Any,
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> None:
        """Qwen2.5-VL モデルとプロセッサをオフライン初期化."""
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
            logger.info(f"Loaded Qwen2.5-VL model from {model_name_or_path} successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Qwen2.5-VL for '{model_name_or_path}': {e}. "
                "Fallback to mock/generate_fn mode."
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

        for content in llm_request.contents or []:
            for part in getattr(content, "parts", []):
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
                # 画像バイトデータまたは PIL Image の取得
                elif hasattr(part, "inline_data") and part.inline_data:
                    import io

                    raw_bytes = part.inline_data.data
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    images.append(img)

        prompt = "\n".join(parts_text)
        return prompt, images

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """マルチモーダル (画像 + テキスト) リクエストを推論処理."""
        prompt, images = self._extract_text_and_images(llm_request)

        # 1. カスタム生成関数 (モック) の場合
        if self._generate_fn is not None:
            if inspect.iscoroutinefunction(self._generate_fn):
                generated_text = await self._generate_fn(prompt, images=images)
            else:
                generated_text = self._generate_fn(prompt, images=images)
        # 2. Qwen2.5-VL モデルの場合
        elif self._model is not None and self._processor is not None:
            import torch
            from qwen_vl_utils import process_vision_info

            content_items: List[Dict[str, Any]] = []
            for img in images:
                content_items.append({"type": "image", "image": img})
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
                generated_ids = self._model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                generated_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
        else:
            generated_text = "```python\ndef transform(grid):\n    return grid.copy()\n```"

        response_content = Content(
            role="model",
            parts=[Part.from_text(text=generated_text)],
        )
        yield LlmResponse(
            content=response_content,
            turn_complete=True,
        )
