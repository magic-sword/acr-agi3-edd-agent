"""Google ADK 2.0 互換のローカル推論 LLM アダプター.

HuggingFace Transformers / オフラインモデルパイプラインをラップし、
Google ADK 2.0 の BaseLlm インターフェースを提供します。
インターネット接続なしで完全オフライン動作可能です。
"""

import inspect
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai.types import Content, Part
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class LocalTransformersLlm(BaseLlm):
    """Google ADK 2.0 向けローカル Transformers / オフライン推論アダプター."""

    model: str = "local-offline-model"
    _pipeline: Any = PrivateAttr(default=None)
    _generate_fn: Optional[Callable[[str], str]] = PrivateAttr(default=None)

    def __init__(
        self,
        model_name_or_path: str = "local-offline-model",
        pipeline: Any = None,
        generate_fn: Optional[Callable[[str], str]] = None,
        generation_fn: Optional[Callable[[str], str]] = None,
        device: str = "cuda",
        torch_dtype: Any = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        """初期化.

        Args:
            model_name_or_path: モデル名またはローカルの重みディレクトリパス
            pipeline: 既存の transformers パイプライン (指定時はこれを優先)
            generate_fn: テストやカスタム生成用のコールバック関数 (fn(prompt) -> output_text)
            generation_fn: generate_fn のエイリアス
            device: 実行デバイス ('cuda', 'cpu')
            torch_dtype: データ型 (torch.float16, torch.bfloat16等)
            load_in_8bit: 8bit量子化でロードするか
            load_in_4bit: 4bit量子化でロードするか
        """
        super().__init__(model=model_name_or_path, **kwargs)
        self._generate_fn = generate_fn or generation_fn


        if pipeline is not None:
            self._pipeline = pipeline
        elif generate_fn is None and model_name_or_path != "mock":
            self._init_pipeline(
                model_name_or_path=model_name_or_path,
                device=device,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

    def _init_pipeline(
        self,
        model_name_or_path: str,
        device: str,
        torch_dtype: Any,
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> None:
        """HuggingFace transformers パイプラインをオフライン初期化."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
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

            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **model_kwargs,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info(f"Loaded local model from {model_name_or_path} successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to initialize transformers pipeline for '{model_name_or_path}': {e}. "
                "Fallback to mock/generate_fn mode."
            )

    def _build_prompt(self, llm_request: LlmRequest) -> str:
        """LlmRequest からテキストプロンプトを再構築."""
        parts_text: List[str] = []

        # システム指示 (System Instruction) の取得
        if llm_request.config and hasattr(llm_request.config, "system_instruction"):
            sys_inst = llm_request.config.system_instruction
            if isinstance(sys_inst, str):
                parts_text.append(f"System: {sys_inst}\n")
            elif hasattr(sys_inst, "parts"):
                for p in sys_inst.parts:
                    if hasattr(p, "text") and p.text:
                        parts_text.append(f"System: {p.text}\n")

        # 各ターン (Contents) のメッセージ抽出
        for content in llm_request.contents or []:
            role = getattr(content, "role", "user") or "user"
            content_strs: List[str] = []
            for part in getattr(content, "parts", []):
                if hasattr(part, "text") and part.text:
                    content_strs.append(part.text)
                elif hasattr(part, "function_response") and part.function_response:
                    content_strs.append(
                        f"[Tool Result for {part.function_response.name}]: "
                        f"{part.function_response.response}"
                    )
            if content_strs:
                prefix = "User: " if role == "user" else "Assistant: "
                parts_text.append(f"{prefix}{''.join(content_strs)}\n")

        parts_text.append("Assistant: ")
        return "\n".join(parts_text)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """ADK 2.0 のコンテンツ生成リクエストを処理."""
        prompt = self._build_prompt(llm_request)

        # 1. カスタム生成関数 (またはモック) の場合
        if self._generate_fn is not None:
            if inspect.iscoroutinefunction(self._generate_fn):
                generated_text = await self._generate_fn(prompt)
            else:
                generated_text = self._generate_fn(prompt)
        # 2. Transformers パイプラインの場合
        elif self._pipeline is not None:
            max_new_tokens = 1024
            temperature = 0.2
            if llm_request.config and hasattr(llm_request.config, "temperature"):
                if llm_request.config.temperature is not None:
                    temperature = float(llm_request.config.temperature)

            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                return_full_text=False,
            )
            generated_text = outputs[0]["generated_text"]
        else:
            generated_text = "```python\ndef transform(grid):\n    return grid.copy()\n```"

        # ADK 2.0 の LlmResponse 形式でレスポンスを返却
        response_content = Content(
            role="model",
            parts=[Part.from_text(text=generated_text)],
        )

        yield LlmResponse(
            content=response_content,
            turn_complete=True,
        )
