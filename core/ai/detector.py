"""
core/ai/detector.py

检测器模块，负责处理视频流中的目标检测。
支持 YOLO 基于模型，支持 .pt / .onnx / .engine 模型格式。
支持 GPU 加速推理。
如果运行在非NVIDIA GPU 上，将使用 CPU 推理,需要修改它的实现。

"""
import time

from ultralytics import YOLO
import torch
from core.config_manager import ConfigManager
from core.logger import get_logger

log = get_logger("detector")

class HumanDetector:
    """
    YOLO-based human detector.
    Supports .pt / .onnx / .engine model formats.
    """

    def __init__(self, model_path=None):
        log.info("[模型加载] >>> 初始化 HumanDetector")

        self.cfg = ConfigManager.get_config().model
        path = model_path or self.cfg.path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # .engine 是 TensorRT 编译产物，imgsz/half 在导出时已固定，推理时不能再传
        # .onnx 在 CPU 上不支持 half precision
        ext = path.lower().rsplit('.', 1)[-1]
        self._is_engine = (ext == 'engine')
        self._is_onnx = (ext == 'onnx')

        log.info(f"[模型加载] 设备: {self.device} | 格式: {ext} | 路径: {path}")

        try:
            # .onnx / .engine 不支持 .to(device)，只能在推理时传 device 参数
            if self._is_engine or self._is_onnx:
                self.model = YOLO(path)
            else:
                self.model = YOLO(path).to(self.device)
            log.success(f"[模型加载] <<< 模型加载完成: {path}")
        except Exception as e:
            log.error(f"[模型加载] <<< 模型加载失败: {e}")
            raise

    def _build_extra(self, include_half=False):
        """构建格式相关的额外推理参数。"""
        extra = {}
        if not self._is_engine:
            extra["imgsz"] = self.cfg.imgsz
        if include_half:
            extra["half"] = (self.cfg.half and self.device == 'cuda'
                             and not self._is_engine and not self._is_onnx)
        if self._is_engine or self._is_onnx:
            extra["device"] = self.device
        return extra

    def detect(self, frame):
        """执行单帧推理（用于预热和独立检测）。"""
        log.trace("[推理执行] >>> 开始推理")
        start = time.time()
        try:
            results = self.model.predict(
                frame,
                conf=self.cfg.conf,
                classes=self.cfg.classes,
                verbose=False,
                **self._build_extra(include_half=True)
            )
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            end = time.time()
            log.trace(f"耗时: {end - start}")
            log.trace(f"[推理执行] <<< 推理完成 | 检测数={num_detections}")
            return results[0]
        except Exception as e:
            log.error(f"[推理执行] <<< 推理异常: {e}")
            raise

    def track(self, frame, persist: bool, classes: list, conf: float, tracker: str):
        """封装 model.track()，屏蔽格式差异，tracker 只调用此方法。"""
        try:
            results = self.model.track(
                frame,
                persist=persist,
                classes=classes,
                conf=conf,
                tracker=tracker,
                verbose=False,
                **self._build_extra(include_half=False)
            )
            return results
        except Exception as e:
            log.error(f"[跟踪执行] <<< track 异常: {e}")
            raise
