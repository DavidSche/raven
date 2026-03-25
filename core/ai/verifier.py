"""
core/ai/verifier.py

活体验证器模块，负责处理视频流中的目标活体验证。
支持基于 bbox 中心点位移方差的轻量活体检测。
完全移除 MediaPipe Pose，用 bbox 中心点轨迹替代，降低 CPU 占用，无内部状态累积。

"""
import numpy as np
from collections import deque
from core.config_manager import ConfigManager
from core.logger import get_logger, TraceSampler

log = get_logger("verifier")

class HumanVerifier:
    """
    基于 bbox 中心点位移方差的轻量活体检测器。
    完全移除 MediaPipe Pose，用 bbox 中心点轨迹替代，CPU 占用极低，无内部状态累积。
    """

    def __init__(self, history_size=None, threshold=None):
        """
        Initializes the verifier with configurable history size and threshold.
        
        Args:
            history_size: Number of frames to track for variance calculation
            threshold: Variance threshold for liveness detection
        """
        log.info("[活体验证] >>> 初始化 HumanVerifier")
        
        self.cfg = ConfigManager.get_config().verifier
        self.history_size = history_size or self.cfg.history_size
        self.threshold = threshold or self.cfg.threshold

        self.history = {}
        self._last_seen = {}
        self._frame_count = 0
        self._cleanup_interval = 300
        
        log.debug(f"[活体验证] 参数: history_size={self.history_size}, threshold={self.threshold}")
        log.success("[活体验证] <<< 初始化完成")

    def verify(self, frame, track_id, bbox):
        """
        Verifies if a tracked object is a live human based on bbox movement variance.
        
        Args:
            frame: Current video frame
            track_id: Unique identifier for the tracked object
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (is_live: bool, variance: float)
        """
        should_trace = TraceSampler.get_instance().should_log("verifier")
        
        if should_trace:
            log.trace(f"[活体验证] >>> 验证目标 | track_id={track_id}")
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            log.warning(f"[活体验证] 帧尺寸无效: w={w}, h={h}")
            return False, 0

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h

        self._frame_count += 1

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_size)
            log.debug(f"[活体验证] 新目标加入跟踪 | track_id={track_id}")

        self.history[track_id].append((cx, cy))
        self._last_seen[track_id] = self._frame_count

        if self._frame_count % self._cleanup_interval == 0:
            stale = [tid for tid, last in self._last_seen.items()
                     if self._frame_count - last > self._cleanup_interval * 2]
            for tid in stale:
                self.history.pop(tid, None)
                self._last_seen.pop(tid, None)
            if stale:
                log.info(f"[活体验证] 清理过期目标 | 数量={len(stale)} | track_ids={stale}")

        if len(self.history[track_id]) < self.history_size:
            if should_trace:
                log.trace(f"[活体验证] <<< 历史帧不足 | track_id={track_id} | frames={len(self.history[track_id])}/{self.history_size}")
            return False, 0

        coords = np.array(self.history[track_id])
        variance = np.var(coords, axis=0).mean()
        
        is_live = variance > self.threshold
        status = "活体" if is_live else "静态"
        
        if should_trace:
            log.trace(f"[活体验证] <<< 验证完成 | track_id={track_id} | variance={variance:.6f} | 结果={status}")
        
        return is_live, variance

    def close(self):
        """Releases resources and clears internal state."""
        log.info("[活体验证] 关闭验证器")
        self.history.clear()
        self._last_seen.clear()
        log.success("[活体验证] 验证器已关闭")
