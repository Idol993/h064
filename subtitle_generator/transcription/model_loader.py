import os
from typing import Optional

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    WhisperModel = None


class WhisperModelLoader:
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
    DEFAULT_MODEL = "base"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None
    ):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"不支持的模型大小: {model_size}，可用模型: {self.AVAILABLE_MODELS}"
            )

        if not HAS_FASTER_WHISPER:
            raise ImportError(
                "faster-whisper 未安装，请运行: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type, self.device)
        self.download_root = download_root
        self._model: Optional["WhisperModel"] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    @staticmethod
    def _resolve_compute_type(compute_type: str, device: str) -> str:
        if compute_type == "auto":
            if device == "cuda":
                return "float16"
            return "int8"
        return compute_type

    def load(self) -> "WhisperModel":
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
        return self._model

    def unload(self):
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
