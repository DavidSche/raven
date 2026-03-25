"""
core/rtsp/base_reader.py

视频流读取器抽象基类，定义统一接口。
所有平台实现必须继承此类。
"""

from abc import ABC, abstractmethod


class BaseReader(ABC):
    """视频流读取器抽象基类。"""
    
    @abstractmethod
    def start(self) -> None:
        """启动采流线程。"""
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        """停止采流，释放资源。"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def state(self) -> str:
        """返回当前状态。"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dropped_frames(self) -> int:
        """返回丢帧计数。"""
        raise NotImplementedError
