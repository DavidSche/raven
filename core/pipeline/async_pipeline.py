"""
core/pipeline/async_pipeline.py

AsyncPipeline — 异步视频处理管道（重构版）

改动要点（相比 core/pipeline.py 旧版）：
  1. 接受 PipelineConfig 注入，按配置条件创建 detector / tracker / verifier
  2. 队列大小由 config 控制（不再硬编码为 2/1）
  3. _reconnect() 新增指数退避（1→2→4→...→30s），并记录连接状态
  4. _worker() 解耦：按能力开关条件调用各模块
  5. 其余接口（start/stop/get_results/get_results_nowait/get_performance_stats）完全不变
"""
import threading
import time
from queue import Queue, Empty

import numpy as np

from core.ai.detector import HumanDetector
from core.ai.tracker import HumanTracker
from core.ai.verifier import HumanVerifier
from core.config_manager import ConfigManager
from core.logger import get_logger
from core.pipeline.pipeline_config import PipelineConfig
from core.rtsp.reader import RtspReader

# ─────────────────────────────────────────────
# 辅助：IoU NMS 去重
# ─────────────────────────────────────────────

log = get_logger("async_pipeline")

def _nms_detections(detections: list, iou_threshold: float = 0.5) -> list:
    """
    对同一帧内的检测结果做 IoU 去重。
    tracker 因置信度波动产生重复轨迹时，保留置信度最高的框，丢弃高度重叠的冗余框。
    """
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

# ─────────────────────────────────────────────
# AsyncPipeline
# ─────────────────────────────────────────────


class AsyncPipeline:
    """
    异步视频处理管道（重构版）。

    Args:
        video_source: 视频源（0=摄像头，RTSP URL，视频文件路径）
        config:       PipelineConfig 实例；为 None 时从 settings.yaml 读取默认值
    """

    def __init__(self, video_source=0, config: PipelineConfig = None):
        log.info("[数据采集] >>> 初始化 AsyncPipeline")

        self.video_source = video_source
        self.cfg_runtime = config or PipelineConfig.from_global_config()

        # ── 队列（大小由配置控制）──
        self.frame_queue  = Queue(maxsize=self.cfg_runtime.frame_queue_size)
        self.result_queue = Queue(maxsize=self.cfg_runtime.result_queue_size)
        self.drop_frames  = self.cfg_runtime.drop_frames
        self.running      = False

        # ── 统计 ──
        self.dropped_frames      = 0
        self.processed_frames    = 0
        self._last_inference_time = 0.0
        self._inference_count    = 0

        # ── RTSP Reader 处理网络、解码与重连 ──
        self.reader = RtspReader(
            video_source=self.video_source,
            frame_queue=self.frame_queue,
            drop_frames=self.cfg_runtime.drop_frames,
            max_retries=self.cfg_runtime.max_retries
        )

        # ── 全局配置 ──
        ConfigManager.load_config()
        self.cfg = ConfigManager.get_config()

        # ── AI 模块（按能力开关条件创建）──
        log.info("[数据采集] 初始化核心组件...")
        self.detector = HumanDetector() if self.cfg_runtime.enable_detector else None
        self.tracker  = HumanTracker()  if self.cfg_runtime.enable_tracker  else None
        self.verifier = HumanVerifier() if self.cfg_runtime.enable_verifier else None

        enabled = [
            ("detector", self.detector is not None),
            ("tracker",  self.tracker  is not None),
            ("verifier", self.verifier is not None),
        ]
        status_str = ", ".join(n + ("=ON" if v else "=OFF") for n, v in enabled)
        log.success(f"[数据采集] 核心组件初始化完成 | {status_str}")

        log.success("[数据采集] <<< AsyncPipeline 初始化完成")



    # ─────────────────────────────────────────
    # 内部：Worker 线程
    # ─────────────────────────────────────────

    def _worker(self):
        """推理线程：按能力开关条件依次调用 detector / tracker / verifier。"""
        log.info("[推理处理] >>> 工作线程启动")

        # 模型预热（仅 detector 存在时需要）
        if self.detector is not None:
            log.info("[推理处理] 模型预热中...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detector.detect(dummy_frame)
            log.success("[推理处理] 模型预热完成")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                inference_start = time.perf_counter()
                log.debug("[推理处理] >>> 开始处理帧")

                # ── Step 1：推理（tracker 优先，否则 detect 降级）──
                results = None
                if self.detector is not None:
                    if self.tracker is not None:
                        results = self.tracker.update(self.detector, frame)
                    else:
                        results = self.detector.detect(frame)

                if results is None:
                    # 无 detector 或推理异常时跳过
                    log.debug("[推理处理] 推理结果为空，跳过本帧")
                    continue

                # ── Step 2：解析结果 ──
                cfg = ConfigManager.get_config()
                processed_data = []
                detection_count = 0

                if results.boxes is not None:
                    masks = (
                        results.masks.xy
                        if cfg.model.segmentation and results.masks is not None
                        else None
                    )

                    for i, box in enumerate(results.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 如果 tracker 未启用，box.id 可能为 None，分配临时 i 作为 track_id
                        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else i
                        mask_points = masks[i] if masks is not None and i < len(masks) else None

                        # ── Step 3：活体验证（可选）──
                        if self.verifier is not None:
                            is_live, score = self.verifier.verify(
                                frame, track_id, [x1, y1, x2, y2]
                            )
                        else:
                            is_live, score = False, 0.0

                        processed_data.append({
                            "bbox":    [x1, y1, x2, y2],
                            "id":      track_id,
                            "conf":    float(box.conf[0].cpu().numpy()),
                            "is_live": is_live,
                            "mask":    mask_points,
                        })
                        detection_count += 1

                # ── Step 4：IoU NMS 去重 ──
                processed_data = _nms_detections(processed_data, iou_threshold=0.5)

                # ── Step 5：写入结果队列 ──
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put((frame, processed_data))
                self.processed_frames += 1

                inference_time = (time.perf_counter() - inference_start) * 1000
                self._last_inference_time = inference_time
                self._inference_count += 1

                if inference_time > 200:
                    log.warning(
                        f"[推理处理] 推理超时告警 | 耗时={inference_time:.1f}ms > 200ms"
                    )

                if cfg.logging.performance.log_inference_time:
                    log.trace(
                        f"[推理处理] <<< 帧处理完成 | "
                        f"耗时={inference_time:.2f}ms | 检测数={detection_count}"
                    )
                else:
                    log.debug(f"[推理处理] <<< 帧处理完成 | 检测数={detection_count}")

            except Exception as e:
                log.error(f"[推理处理] 处理异常: {e}", exc_info=True)
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        log.warning("[推理处理] CUDA OOM，已清理显存缓存")
                    except Exception:
                        pass
                time.sleep(0.1)

        log.info("[推理处理] <<< 工作线程已停止")

    # ─────────────────────────────────────────
    # 公开接口（与旧版完全兼容）
    # ─────────────────────────────────────────

    def start(self):
        """启动 Reader 和 Worker 处理链。"""
        log.info("[系统控制] >>> 启动 Pipeline")
        self.running = True

        # 1. 启动专职 Reader
        self.reader.start()

        # 2. 启动 Worker 守护线程
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        log.success("[系统控制] <<< Pipeline 已启动")

    def stop(self):
        """停止所有线程并释放资源。"""
        log.info("[系统控制] >>> 停止 Pipeline")
        self.running = False
        
        self.reader.stop()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
        if self.verifier is not None:
            self.verifier.close()
            
        log.info(
            f"[系统控制] 统计信息: "
            f"已处理帧={self.processed_frames}, 丢弃帧={self.reader.dropped_frames}"
        )
        log.success("[系统控制] <<< Pipeline 已停止")

    def get_results(self):
        """阻塞式获取最新处理帧（超时 100ms）。"""
        try:
            return self.result_queue.get(timeout=0.1)
        except Empty:
            return None, None

    def get_results_nowait(self):
        """非阻塞式获取最新处理帧，供 async 环境使用。"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None, None

    def get_performance_stats(self) -> dict:
        """返回当前性能统计信息（含 CPU/GPU 内存）。"""
        stats = {
            "processed_frames":       self.processed_frames,
            "dropped_frames":         self.reader.dropped_frames if self.reader else 0,
            "last_inference_time_ms": round(self._last_inference_time, 2),
            "queue_size":             self.frame_queue.qsize(),
            "stream_state":           self.reader.state if self.reader else "unknown",
        }
        try:
            import psutil
            proc = psutil.Process()
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["ram_mb"]      = round(proc.memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            pass
            
        import torch
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024 / 1024, 1
            )
            stats["gpu_reserved_mb"] = round(
                torch.cuda.memory_reserved() / 1024 / 1024, 1
            )
        return stats
