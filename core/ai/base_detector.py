"""
core/ai/base_detector.py

检测器抽象基类，定义统一接口。
所有平台实现必须继承此类并实现 detect 方法。
"""

from abc import ABC, abstractmethod
import numpy as np

from core.ai.result import DetectionResult


class BaseDetector(ABC):
    """检测器抽象基类。"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        执行单帧检测。
        
        Args:
            frame: BGR 格式输入图像 (H, W, C)
            
        Returns:
            DetectionResult: 统一格式的检测结果
        """
        raise NotImplementedError
    
    @abstractmethod
    def warmup(self, iterations: int = 3) -> None:
        """模型预热。"""
        raise NotImplementedError

    @abstractmethod
    def track(self, frame: np.ndarray, persist: bool, classes: list, conf: float, tracker: str):
        """兼容跟踪器。"""
        raise NotImplementedError
        
    @property
    @abstractmethod
    def device_name(self) -> str:
        """返回设备名称（用于日志）。"""
        raise NotImplementedError
