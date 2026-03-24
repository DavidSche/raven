import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field

class LoggingFileConfig(BaseModel):
    enabled: bool = True
    path: str = "logs"
    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"

class LoggingPerformanceConfig(BaseModel):
    log_fps_interval: int = 30
    log_inference_time: bool = True

class LoggingConfig(BaseModel):
    level: str = "INFO"
    console_enabled: bool = True
    file: LoggingFileConfig = LoggingFileConfig()
    performance: LoggingPerformanceConfig = LoggingPerformanceConfig()

class ModelConfig(BaseModel):
    path: str
    imgsz: int = Field(ge=32, le=1920, description="推理图像尺寸，建议 320/640/1280")
    conf: float = Field(ge=0.01, le=1.0, description="检测置信度阈值")
    classes: List[int]
    half: bool
    segmentation: bool = True

class TrackerConfig(BaseModel):
    type: str
    conf: float = Field(ge=0.01, le=1.0, description="跟踪置信度阈值")
    persist: bool

class VerifierConfig(BaseModel):
    history_size: int = Field(ge=2, le=300, description="活体检测滑动窗口大小")
    threshold: float = Field(ge=0.0, description="位移方差阈值")

class SystemConfig(BaseModel):
    show_ui: bool
    source: str = "0" # Default to webcam

class PipelineRuntimeConfig(BaseModel):
    """Pipeline 运行时配置，对应 config/settings.yaml 中的 pipeline: 节。"""
    enable_detector: bool = True
    enable_tracker: bool = True
    enable_verifier: bool = True
    drop_frames: bool = True
    frame_queue_size: int = Field(ge=1, le=32, default=2)
    result_queue_size: int = Field(ge=1, le=16, default=1)
    reconnect: bool = True
    max_retries: int = -1  # -1 = 无限重试

class GlobalConfig(BaseModel):
    model: ModelConfig
    tracker: TrackerConfig
    verifier: VerifierConfig
    system: SystemConfig
    logging: LoggingConfig = LoggingConfig()
    pipeline: PipelineRuntimeConfig = PipelineRuntimeConfig()

class ConfigManager:
    _config: GlobalConfig = None

    @classmethod
    def load_config(cls, config_path: str = "config/settings.yaml") -> GlobalConfig:
        """
        Loads the YAML configuration and validates it using Pydantic.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"目录下没有对应的配置文件 settings.yaml，请检查: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        
        cls._config = GlobalConfig(**raw_config)
        return cls._config

    @classmethod
    def get_config(cls) -> GlobalConfig:
        if cls._config is None:
            return cls.load_config()
        return cls._config
