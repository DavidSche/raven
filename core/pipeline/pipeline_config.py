"""
core/pipeline/pipeline_config.py
每路 Pipeline 的运行时配置。

用法：
  1. 单路（从 settings.yaml 读取默认值）：
       cfg = PipelineConfig.from_global_config()

  2. 多路（每路独立覆盖）：
       cfg_cam1 = PipelineConfig(enable_verifier=True)
       cfg_cam2 = PipelineConfig(enable_tracker=False, enable_verifier=False)
"""
from dataclasses import dataclass
from core.config_manager import ConfigManager


@dataclass
class PipelineConfig:
    # ===== AI 能力开关 =====
    enable_detector: bool = True
    enable_tracker: bool = True
    enable_verifier: bool = True

    # ===== 帧队列策略 =====
    drop_frames: bool = True
    frame_queue_size: int = 2
    result_queue_size: int = 1

    # ===== RTSP 重连策略 =====
    reconnect: bool = True
    max_retries: int = -1  # -1 = 无限重试

    @staticmethod
    def from_global_config() -> "PipelineConfig":
        """从 config/settings.yaml 的 pipeline: 节读取默认配置。"""
        cfg = ConfigManager.get_config().pipeline
        return PipelineConfig(
            enable_detector=cfg.enable_detector,
            enable_tracker=cfg.enable_tracker,
            enable_verifier=cfg.enable_verifier,
            drop_frames=cfg.drop_frames,
            frame_queue_size=cfg.frame_queue_size,
            result_queue_size=cfg.result_queue_size,
            reconnect=cfg.reconnect,
            max_retries=cfg.max_retries,
        )
