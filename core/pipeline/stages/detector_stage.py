from core.pipeline.stages.base import PipelineStage
from core.ai.model_registry import ModelRegistry

class DetectorStage(PipelineStage):
    """
    负责目标检测（全流全局复用唯一检测引擎）
    """
    def __init__(self):
        self.detector = ModelRegistry.get_detector()

    def process(self, data: dict) -> dict:
        frame = data["frame"]
        
        # 将传入帧投喂给 detector
        results = self.detector.detect(frame)
        
        # 将 YOLOv8 results 对象挂入上下文
        data["results"] = results
        return data
