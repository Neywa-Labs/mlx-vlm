from ..base import install_auto_processor_patch
from .config import AudioConfig, ModelConfig, TextConfig, VisionConfig
from .processing_wemm_video import WeMMVideoProcessor
from .wemm_video import Model, VisionModel

install_auto_processor_patch("wemm_video", WeMMVideoProcessor, trust_remote_code=False)
