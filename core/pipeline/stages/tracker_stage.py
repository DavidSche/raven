from core.pipeline.stages.base import PipelineStage
from core.ai.tracker import HumanTracker
from core.ai.model_registry import ModelRegistry

class TrackerStage(PipelineStage):
    """
    负责目标跟踪。跟踪器本身状态(轨迹内存)不可混用，所以本实例为单路独享。
    但它需要借用探测器以做 fallback tracking，所以仍从全局引 detector 工具人实例。
    """
    def __init__(self):
        self.tracker = HumanTracker()
        self.detector = ModelRegistry.get_detector()

    def process(self, data: dict) -> dict:
        frame = data["frame"]
        
        # tracker 会在内部拦截并处理，必要时会 fallback 给 detector
        results = self.tracker.update(self.detector, frame)
        
        data["results"] = results
        return data
