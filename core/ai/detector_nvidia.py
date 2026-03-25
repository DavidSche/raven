"""
core/ai/detector_nvidia.py

NVIDIA GPU 平台检测器实现。
支持 .pt / .onnx / .engine 模型格式。
"""

import time
from typing import Optional
import numpy as np
import torch
from ultralytics import YOLO

from core.ai.base_detector import BaseDetector
from core.ai.result import DetectionResult
from core.config_manager import ConfigManager
from core.logger import get_logger, TraceSampler

log = get_logger("detector-nvidia")


class NvidiaDetector(BaseDetector):
    """NVIDIA GPU 平台检测器。"""
    
    def __init__(self, model_path: Optional[str] = None):
        log.info("[模型加载] >>> 初始化 NvidiaDetector")
        
        self.cfg = ConfigManager.get_config().model
        self.model_path = model_path or self.cfg.path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        ext = self.model_path.lower().rsplit('.', 1)[-1]
        self._is_engine = (ext == 'engine')
        self._is_onnx = (ext == 'onnx')
        
        log.info(f"[模型加载] 设备: {self.device} | 格式: {ext}")
        
        # .engine 仅能直接加载
        if self._is_engine or self._is_onnx:
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(self.model_path).to(self.device)
        log.success(f"[模型加载] <<< 模型加载完成")
    
    @property
    def device_name(self) -> str:
        return f"NVIDIA {self.device.upper()}"
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        should_trace = TraceSampler.get_instance().should_log("detector")
        
        start = time.time()
        
        extra = {}
        if not self._is_engine:
            extra["imgsz"] = self.cfg.imgsz
        if self._is_engine or self._is_onnx:
            extra["device"] = self.device

        results = self.model.predict(
            frame,
            conf=self.cfg.conf,
            classes=self.cfg.classes,
            verbose=False,
            **extra
        )
        
        inference_time = (time.time() - start) * 1000
        
        if should_trace:
            num_det = len(results[0].boxes) if results[0].boxes else 0
            log.trace(f"[推理] 完成 | 检测数={num_det}, 耗时={inference_time:.2f}ms")
        
        detections = self._parse_results(results[0])
        
        return DetectionResult(
            detections=detections,
            original_shape=frame.shape,
            inference_time=inference_time
        )
    
    def track(self, frame: np.ndarray, persist: bool, classes: list, conf: float, tracker: str):
        """封装 model.track()，兼容遗留的 Tracker"""
        try:
            extra = {}
            if not self._is_engine:
                extra["imgsz"] = self.cfg.imgsz
            if self._is_engine or self._is_onnx:
                extra["device"] = self.device

            results = self.model.track(
                frame,
                persist=persist,
                classes=classes,
                conf=conf,
                tracker=tracker,
                verbose=False,
                **extra
            )
            return results
        except Exception as e:
            log.error(f"[跟踪执行] <<< track 异常: {e}")
            raise
    
    def _parse_results(self, result) -> list:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        detections = []
        for i in range(len(result.boxes)):
            xyxy = result.boxes.xyxy[i].cpu().numpy()
            conf = float(result.boxes.conf[i].cpu().numpy())
            cls = int(result.boxes.cls[i].cpu().numpy())
            
            det = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls]
            if result.boxes.is_track and result.boxes.id is not None:
                det.append(int(result.boxes.id[i].cpu().numpy()))
            
            detections.append(det)
        
        return detections
    
    def warmup(self, iterations: int = 3) -> None:
        log.info(f"[预热] 开始 ({iterations} 次)")
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        log.success("[预热] 完成")
