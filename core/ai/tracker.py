"""
core/ai/tracker.py

跟踪器模块，负责处理视频流中的目标跟踪。
支持 ByteTrack 算法，保持一致的目标 ID 跨帧跟踪。
"""

from core.config_manager import ConfigManager
from core.logger import get_logger

log = get_logger("tracker")

class HumanTracker:
    """
    Multi-object tracker using ByteTrack algorithm.
    Maintains consistent object IDs across frames.
    """
    
    def __init__(self):
        log.info("[跟踪器] >>> 初始化 HumanTracker")
        self.cfg = ConfigManager.get_config().tracker
        self._frame_count = 0
        log.debug(f"[跟踪器] 参数: type={self.cfg.type}, conf={self.cfg.conf}, persist={self.cfg.persist}")
        log.success("[跟踪器] <<< 初始化完成")

    def update(self, detector, frame):
        """Updates the tracker with a new frame."""
        self._frame_count += 1
        log.trace(f"[跟踪执行] >>> 开始跟踪 | 帧计数={self._frame_count}")
        
        try:
            cfg = ConfigManager.get_config()
            results = detector.track(
                frame,
                persist=self.cfg.persist,
                classes=cfg.model.classes,
                conf=self.cfg.conf,
                tracker=self.cfg.type,
            )
            
            if results and len(results) > 0:
                track_count = len(results[0].boxes) if results[0].boxes is not None else 0
                log.trace(f"[跟踪执行] <<< 跟踪完成 | 跟踪目标数={track_count}")
            
            return results[0] if results else None
        except Exception as e:
            log.error(f"[跟踪执行] <<< 跟踪异常: {e}")
            return None
