from core.pipeline.stages.base import PipelineStage
from core.ai.verifier import HumanVerifier

def _nms_detections(detections: list, iou_threshold: float = 0.5) -> list:
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        suppressed = False
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept["bbox"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep


class VerifierStage(PipelineStage):
    """
    负责活体、异常验证（及解析 bbox 生成 processed 数据结构）和 NMS 过滤。
    """
    def __init__(self, enable_verifier=True):
        self.enable_verifier = enable_verifier
        self.verifier = HumanVerifier() if enable_verifier else None

    def process(self, data: dict) -> dict:
        frame = data["frame"]
        results = data.get("results")
        
        processed = []
        if results and getattr(results, "boxes", None) is not None:
            # 引入了检测头内的 mask 数据
            masks = (
                results.masks.xy
                if hasattr(results, 'masks') and results.masks is not None
                else None
            )

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else i
                mask_points = masks[i] if masks is not None and i < len(masks) else None

                # 执行活体验证（若被启用）
                if self.verifier is not None:
                    is_live, score = self.verifier.verify(
                        frame, track_id, [x1, y1, x2, y2]
                    )
                else:
                    is_live, score = False, 0.0

                processed.append({
                    "bbox": [x1, y1, x2, y2],
                    "id": track_id,
                    "conf": float(box.conf[0].cpu().numpy()),
                    "is_live": is_live,
                    "mask": mask_points,
                })
        
        # 去重
        processed = _nms_detections(processed, iou_threshold=0.5)
        
        # 丢弃原生 results 对象，转为标准化的 detections
        data["detections"] = processed
        return data
