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
import cv2
import threading
import time
import numpy as np
import os
from queue import Queue, Empty
from core.ai.detector import HumanDetector
from core.ai.tracker import HumanTracker
from core.ai.verifier import HumanVerifier
from core.config_manager import ConfigManager
from core.logger import get_logger, PerformanceLogger
from core.pipeline.pipeline_config import PipelineConfig

log = get_logger("pipeline")


# ─────────────────────────────────────────────
# 辅助：cudacodec 可用性检测
# ─────────────────────────────────────────────

def _check_cudacodec() -> bool:
    try:
        if not hasattr(cv2, 'cudacodec'):
            return False
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return False
        return True
    except Exception as e:
        log.debug(f"[系统检查] cudacodec 检测异常: {e}")
        return False


CUDACODEC_AVAILABLE = _check_cudacodec()


# ─────────────────────────────────────────────
# 辅助：IoU NMS 去重
# ─────────────────────────────────────────────

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
# 连接状态枚举
# ─────────────────────────────────────────────

class StreamState:
    CONNECTING   = "connecting"
    STREAMING    = "streaming"
    RECONNECTING = "reconnecting"
    FAILED       = "failed"


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

        # ── RTSP 状态机 ──
        self._stream_state = StreamState.CONNECTING
        self._retry_count  = 0

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

        # ── RTSP 低延迟参数 ──
        is_rtsp = isinstance(video_source, str) and video_source.startswith("rtsp://")
        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp"
                "|fflags;nobuffer+discardcorrupt"
                "|flags;low_delay"
                "|reorder_queue_size;0"
                "|allowed_media_types;video"
                "|stimeout;5000000"
                "|max_delay;0"
                "|buffer_size;65536"
            )
            log.info("[数据采集] RTSP 低延迟调优已启用")

        # ── 打开视频源 ──
        log.info(f"[数据采集] 打开视频源: {video_source}")
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            log.error(f"[数据采集] 无法打开视频源: {video_source}")
            raise Exception(f"Could not open video source: {video_source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        log.success("[数据采集] 视频源已连接")

        self._stream_state = StreamState.STREAMING

        # ── FPS / 文件源判断 ──
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_video_file = (
            isinstance(video_source, str)
            and not is_rtsp
            and self.fps > 0
        )
        self.frame_delay = 1.0 / self.fps if self.is_video_file else 0.001
        log.info(f"[数据采集] 视频参数: fps={self.fps:.2f}, is_video_file={self.is_video_file}")

        # ── GPU 硬件解码 ──
        self.cuda_reader = None
        if isinstance(video_source, str) and CUDACODEC_AVAILABLE:
            try:
                self.cuda_reader = cv2.cudacodec.createVideoReader(video_source)
                self.cuda_reader.set(cv2.cudacodec.ColorFormat_BGR)
                log.success("[数据采集] GPU 硬件解码已启用 (cudacodec)")
            except Exception as e:
                self.cuda_reader = None
                log.warning(f"[数据采集] cudacodec 初始化失败，回退到 CPU: {e}")
        elif isinstance(video_source, str):
            log.info("[数据采集] 使用 CPU 软件解码 (cudacodec 不可用)")

        log.success("[数据采集] <<< AsyncPipeline 初始化完成")

    # ─────────────────────────────────────────
    # 内部：重连（带指数退避）
    # ─────────────────────────────────────────

    def _reconnect(self):
        """RTSP 断线重连，使用指数退避策略（最大等待 30s）。"""
        self._stream_state = StreamState.RECONNECTING

        delay = min(2 ** self._retry_count, 30)
        log.warning(
            f"[数据采集] 等待 {delay}s 后重连 "
            f"(第 {self._retry_count + 1} 次) | source={self.video_source}"
        )
        time.sleep(delay)
        self._retry_count += 1

        # ── GPU 路径 ──
        if self.cuda_reader is not None:
            try:
                self.cuda_reader = cv2.cudacodec.createVideoReader(self.video_source)
                self.cuda_reader.set(cv2.cudacodec.ColorFormat_BGR)
                log.success("[数据采集] GPU 读取器重连成功")
                self._stream_state = StreamState.STREAMING
                self._retry_count = 0
            except Exception as e:
                log.error(f"[数据采集] GPU 重连失败，回退到 CPU: {e}")
                self.cuda_reader = None
                cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self.cap = cap
                if cap.isOpened():
                    self._stream_state = StreamState.STREAMING
                    self._retry_count = 0
        # ── CPU 路径 ──
        else:
            self.cap.release()
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap = cap
            if cap.isOpened():
                log.success("[数据采集] CPU 读取器重连成功")
                self._stream_state = StreamState.STREAMING
                self._retry_count = 0
            else:
                log.error("[数据采集] 重连后视频源仍无法打开")

        # 检查最大重试限制
        if (self.cfg_runtime.max_retries != -1
                and self._retry_count > self.cfg_runtime.max_retries):
            log.error(
                f"[数据采集] 已超过最大重试次数 ({self.cfg_runtime.max_retries})，"
                "标记为 FAILED"
            )
            self._stream_state = StreamState.FAILED

        # 重连后清空队列积压
        cleared = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                cleared += 1
            except Empty:
                break
        if cleared > 0:
            log.debug(f"[数据采集] 重连后清空队列积压: {cleared} 帧")

    # ─────────────────────────────────────────
    # 内部：Reader 线程
    # ─────────────────────────────────────────

    def _reader(self):
        """帧采集线程：GPU 解码优先，CPU 降级；RTSP 断线自动重连。"""
        log.info(
            f"[数据采集] >>> 读取线程启动 "
            f"(FPS 限制: {self.fps if self.is_video_file else 'None'})"
        )

        is_rtsp    = isinstance(self.video_source, str) and self.video_source.startswith("rtsp://")
        fail_count   = 0
        corrupt_count = 0

        while self.running:
            # 若已达 FAILED 状态且无限重试关闭，停止循环
            if self._stream_state == StreamState.FAILED:
                log.error("[数据采集] 流已进入 FAILED 状态，读取线程退出")
                break

            start_time = time.time()

            # ── GPU 解码路径 ──
            if self.cuda_reader is not None:
                try:
                    ret, gpu_frame = self.cuda_reader.nextFrame()
                    if not ret or gpu_frame is None:
                        fail_count += 1
                        if fail_count > 30:
                            log.warning(f"[数据采集] GPU 连续读取失败 {fail_count} 次，触发重连")
                            self._reconnect()
                            fail_count = 0
                            corrupt_count = 0
                        continue
                    fail_count = 0
                    frame = gpu_frame.download()
                except Exception as e:
                    log.error(f"[数据采集] GPU 解码错误: {e}，回退到 CPU")
                    self.cuda_reader = None
                    continue

            # ── CPU 解码路径 ──
            else:
                ret, frame = self.cap.read()
                if not ret:
                    fail_count += 1
                    if is_rtsp and fail_count > 30:
                        log.warning(f"[数据采集] CPU 连续读取失败 {fail_count} 次，触发重连")
                        self._reconnect()
                        fail_count = 0
                        corrupt_count = 0
                    elif self.is_video_file:
                        log.info("[数据采集] 视频文件播放完毕，重新开始")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.01)
                    continue
                fail_count = 0

            # ── RTSP 损坏帧检测（GPU/CPU 公用）──
            if is_rtsp:
                h, w = frame.shape[:2]
                sample = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
                std = cv2.meanStdDev(sample)[1][0][0]
                if std < 2.0:
                    corrupt_count += 1
                    if corrupt_count >= 5:
                        log.warning(f"[数据采集] 检测到 {corrupt_count} 帧损坏，触发重连")
                        self._reconnect()
                        corrupt_count = 0
                    continue
                else:
                    corrupt_count = 0

            # ── 帧丢弃策略 ──
            if self.frame_queue.full():
                if self.drop_frames or is_rtsp:
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except Empty:
                        pass
                else:
                    time.sleep(0.01)
                    continue

            self.frame_queue.put(frame)

            # 视频文件按原始 FPS 节流
            if self.is_video_file:
                elapsed = time.time() - start_time
                wait_time = max(0, self.frame_delay - elapsed)
                if wait_time > 0:
                    time.sleep(wait_time)
            elif not is_rtsp:
                time.sleep(0.001)

        log.info("[数据采集] <<< 读取线程已停止")

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
        """启动 Reader / Worker 两个 daemon 线程。"""
        log.info("[系统控制] >>> 启动 Pipeline")
        self.running = True
        self.threads = [
            threading.Thread(target=self._reader, daemon=True),
            threading.Thread(target=self._worker, daemon=True),
        ]
        for t in self.threads:
            t.start()
        log.success("[系统控制] <<< Pipeline 已启动")

    def stop(self):
        """停止所有线程并释放资源。"""
        log.info("[系统控制] >>> 停止 Pipeline")
        self.running = False
        for t in self.threads:
            t.join(timeout=2)
        self.cap.release()
        if self.verifier is not None:
            self.verifier.close()
        log.info(
            f"[系统控制] 统计信息: "
            f"已处理帧={self.processed_frames}, 丢弃帧={self.dropped_frames}"
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
        import torch
        stats = {
            "processed_frames":       self.processed_frames,
            "dropped_frames":         self.dropped_frames,
            "last_inference_time_ms": round(self._last_inference_time, 2),
            "queue_size":             self.frame_queue.qsize(),
            "stream_state":           self._stream_state,
        }
        try:
            import psutil
            proc = psutil.Process()
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["ram_mb"]      = round(proc.memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            pass
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024 / 1024, 1
            )
            stats["gpu_reserved_mb"] = round(
                torch.cuda.memory_reserved() / 1024 / 1024, 1
            )
        return stats
