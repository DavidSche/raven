"""
core/ai/base_tracker.py

跟踪器抽象基类，定义统一接口。
"""

from abc import ABC, abstractmethod
from typing import Any
from core.ai.base_detector import BaseDetector

class BaseTracker(ABC):
    """跟踪器抽象基类。"""
    
    @abstractmethod
    def update(self, detector: BaseDetector, frame: Any) -> Any:
        """
        根据当前帧更新跟踪器目标状态。
        
        Args:
            detector: 借用的检测器（在必要时 fallback 检测）
            frame: 当前视频流画面
            
        Returns:
            带有 Tracker ID 的 DetectionResult 集合
        """
        raise NotImplementedError
