from core.ai.detector import HumanDetector
from core.logger import get_logger
import threading

log = get_logger("model-registry")

class ModelRegistry:
    """全局模型单例（GPU共享分配）
    解决多路路流引发的显存（VRAM）爆炸问题，所有的流共享一个底层推理引擎。
    """
    
    _detector = None
    _lock = threading.Lock()
    
    @classmethod
    def get_detector(cls):
        # 典型的 DCL (Double-Checked Locking) 单例模式
        if cls._detector is None:
            with cls._lock:
                if cls._detector is None:
                    log.info("[模型注册] 初始化全局共享 Detector (单实例)")
                    cls._detector = HumanDetector()
        return cls._detector
