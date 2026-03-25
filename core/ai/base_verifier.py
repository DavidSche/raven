"""
core/ai/base_verifier.py

验证器抽象基类，定义统一接口。
负责根据跟踪目标的边界框在时间序列上鉴别活体或网络状态。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseVerifier(ABC):
    """验证器抽象基类。"""
    
    @abstractmethod
    def verify(self, frame: Any, track_id: int, bbox: tuple) -> Tuple[bool, float]:
        """
        执行单次验证（活体、移动幅度等判定）。
        
        Args:
            frame: 当前帧图像
            track_id: 跟踪目标 ID
            bbox: (x1, y1, x2, y2)
            
        Returns:
            (is_live, confidence)
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, data: dict) -> dict:
        """
        （可选）批量处理接口供 Pipeline Stage 消费使用。
        """
        raise NotImplementedError
