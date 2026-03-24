import cv2
import threading
import time
import numpy as np
import os
from queue import Queue, Empty
from core.detector import HumanDetector
from core.tracker import HumanTracker
from core.verifier import HumanVerifier
from core.config_manager import ConfigManager
from core.logger import get_logger, PerformanceLogger

log = get_logger("pipeline")

# 检测 cv2.cudacodec 是否可用（需要 opencv-contrib + CUDA 编译版本）
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


def _nms_detections(detections: list, iou_threshold: float = 0.5) -> list:
    """
    对同一帧内的检测结果做 IoU 去重。
    当 tracker 因置信度波动产生重复轨迹时，保留置信度最高的框，丢弃高度重叠的冗余框。
    """
    if len(detections) <= 1:
        return detections

    # 按置信度降序排列
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        suppressed = False
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept["bbox"]
            # 计算交集
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

class AsyncPipeline:
    """
    Asynchronous video processing pipeline.
    Coordinates frame reading, detection, tracking, and verification across multiple threads.
    """
    
    def __init__(self, video_source=0, drop_frames=True):
        """
        Initializes the pipeline with video source and frame dropping policy.
        
        Args:
            video_source: Video source (0 for webcam, RTSP URL, or file path)
            drop_frames: Whether to drop frames when queue is full (recommended for RTSP)
        """
        log.info("[数据采集] >>> 初始化 AsyncPipeline")
        log.debug(f"[数据采集] 参数: video_source={video_source}, drop_frames={drop_frames}")
        
        self.video_source = video_source
        self.drop_frames = drop_frames
        
        # 队列保持极小：始终只处理/显示最新帧，避免积压导致抖动
        # frame_queue=2: reader 只超前 worker 1-2 帧，不积压旧帧
        # result_queue=1: 主线程永远取到最新结果，不显示过期帧
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=1)
        self.running = False
        
        # 帧丢弃统计
        self.dropped_frames = 0
        self.processed_frames = 0
        self._last_inference_time = 0.0
        self._inference_count = 0
        
        # Initialize components
        ConfigManager.load_config()
        self.cfg = ConfigManager.get_config()
        
        log.info("[数据采集] 初始化核心组件...")
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.verifier = HumanVerifier()
        log.success("[数据采集] 核心组件初始化完成")
        
        # FIX RTSP ARTIFACTS + LOW LATENCY: Force TCP, discard corrupt, minimize buffer
        if isinstance(video_source, str) and video_source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp"
                "|fflags;nobuffer+discardcorrupt"
                "|flags;low_delay"
                "|reorder_queue_size;0"
                "|allowed_media_types;video"
                "|stimeout;5000000"
                "|max_delay;0"          # 强制 FFmpeg demuxer 延迟为 0
                "|buffer_size;65536"    # 网络接收缓冲 64KB，减少 socket 层积压
            )
            log.info("[数据采集] RTSP 低延迟调优已启用")
        
        log.info(f"[数据采集] 打开视频源: {video_source}")
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            log.error(f"[数据采集] 无法打开视频源: {video_source}")
            raise Exception(f"Could not open video source: {video_source}")
        log.success("[数据采集] 视频源已连接")
        
        # Set buffer size to a safe minimum to reduce latency while preventing dropped packets
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Determine source FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # RTSP 流虽然有 FPS 值，但不应按文件模式节流，需单独区分
        is_rtsp_source = isinstance(video_source, str) and video_source.startswith("rtsp://")
        self.is_video_file = isinstance(video_source, str) and not is_rtsp_source and self.fps > 0
        self.frame_delay = 1.0 / self.fps if self.is_video_file else 0.001
        log.info(f"[数据采集] 视频参数: fps={self.fps:.2f}, is_video_file={self.is_video_file}")
        
        # 尝试启用 GPU 硬件解码（仅 RTSP/文件流，摄像头不支持）
        self.cuda_reader = None
        is_stream = isinstance(video_source, str)
        if is_stream and CUDACODEC_AVAILABLE:
            try:
                self.cuda_reader = cv2.cudacodec.createVideoReader(video_source)
                # 输出格式统一为 BGR，与 CPU 路径一致
                self.cuda_reader.set(cv2.cudacodec.ColorFormat_BGR)
                log.success("[数据采集] GPU 硬件解码已启用 (cudacodec)")
            except Exception as e:
                self.cuda_reader = None
                log.warning(f"[数据采集] cudacodec 初始化失败，回退到 CPU: {e}")
        elif is_stream:
            log.info("[数据采集] 使用 CPU 软件解码 (cudacodec 不可用)")
        
        log.success("[数据采集] <<< AsyncPipeline 初始化完成")

    def _reconnect(self):
        """Reconnects to the video source after connection loss."""
        log.warning(f"[数据采集] 开始重连: {self.video_source}")
        
        if self.cuda_reader is not None:
            try:
                self.cuda_reader = cv2.cudacodec.createVideoReader(self.video_source)
                self.cuda_reader.set(cv2.cudacodec.ColorFormat_BGR)
                log.success("[数据采集] GPU 读取器重连成功")
            except Exception as e:
                log.error(f"[数据采集] GPU 重连失败，回退到 CPU: {e}")
                self.cuda_reader = None
                cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self.cap = cap
        else:
            self.cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap = cap
            log.success("[数据采集] CPU 读取器重连成功")
        
        # 重连后清空队列积压，确保 worker 拿到的是最新帧
        cleared = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                cleared += 1
            except Empty:
                break
        if cleared > 0:
            log.debug(f"[数据采集] 重连后清空队列积压: {cleared} 帧")

    def _reader(self):
        """Thread for reading frames. GPU decoding if cudacodec available, else CPU fallback."""
        log.info(f"[数据采集] >>> 读取线程启动 (FPS 限制: {self.fps if self.is_video_file else 'None'})")
        
        is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith("rtsp://")
        fail_count = 0
        corrupt_count = 0

        while self.running:
            start_time = time.time()

            # --- GPU 解码路径 ---
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
                    # 从 GPU 内存下载到 CPU numpy（YOLO 推理需要 numpy）
                    frame = gpu_frame.download()
                except Exception as e:
                    log.error(f"[数据采集] GPU 解码错误: {e}，回退到 CPU")
                    self.cuda_reader = None
                    continue

            # --- CPU 解码路径 ---
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

            # RTSP 损坏帧检测（GPU/CPU 路径共用）
            # cv2.meanStdDev 无额外拷贝，比 np.std() 快且不产生 float64 临时内存
            if is_rtsp:
                h, w = frame.shape[:2]
                sample = frame[h//4:3*h//4, w//4:3*w//4]
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

            # 帧丢弃策略：队列满时丢弃旧帧
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

    def _worker(self):
        """Thread for processing detection and tracking."""
        log.info("[推理处理] >>> 工作线程启动")
        log.info("[推理处理] 模型预热中...")
        
        # Cold start for model
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
                
                # 1. Update Tracker
                results = self.tracker.update(self.detector, frame)

                # 重置帧或推理异常时 results 可能为 None，直接跳过本帧
                if results is None:
                    log.warning("[推理处理] 推理结果为空，跳过本帧")
                    continue

                # 2. Process results (Detection + Segmentation + Tracking)
                cfg = ConfigManager.get_config()
                processed_data = []
                if results.boxes is not None:
                    # 仅在 segmentation=true 且模型有 mask 输出时才处理
                    masks = (results.masks.xy
                             if cfg.model.segmentation and results.masks is not None
                             else None)
                    
                    detection_count = 0
                    for i, box in enumerate(results.boxes):
                        if box.id is None: continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id[0].cpu().numpy())
                        
                        # Get mask if available
                        mask_points = masks[i] if masks is not None and i < len(masks) else None
                        
                        is_live, score = self.verifier.verify(frame, track_id, [x1, y1, x2, y2])
                        processed_data.append({
                            "bbox": [x1, y1, x2, y2],
                            "id": track_id,
                            "conf": float(box.conf[0].cpu().numpy()),
                            "is_live": is_live,
                            "mask": mask_points
                        })
                        detection_count += 1
                
                # IoU 去重：同一帧内高度重叠的框只保留置信度最高的，防止 tracker 双框
                processed_data = _nms_detections(processed_data, iou_threshold=0.5)

                # 3. Put result
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
                    log.warning(f"[推理处理] 推理超时告警 | 耗时={inference_time:.1f}ms > 200ms")

                if cfg.logging.performance.log_inference_time:
                    log.trace(f"[推理处理] <<< 帧处理完成 | 耗时={inference_time:.2f}ms | 检测数={detection_count}")
                else:
                    log.debug(f"[推理处理] <<< 帧处理完成 | 检测数={detection_count}")
                
            except Exception as e:
                log.error(f"[推理处理] 处理异常: {e}", exc_info=True)
                # CUDA OOM 时清理显存，避免后续帧持续失败
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        log.warning("[推理处理] CUDA OOM，已清理显存缓存")
                    except Exception:
                        pass
                time.sleep(0.1)
                
        log.info("[推理处理] <<< 工作线程已停止")

    def start(self):
        """Starts the reader and worker threads."""
        log.info("[系统控制] >>> 启动 Pipeline")
        self.running = True
        self.threads = [
            threading.Thread(target=self._reader, daemon=True),
            threading.Thread(target=self._worker, daemon=True)
        ]
        for t in self.threads:
            t.start()
        log.success("[系统控制] <<< Pipeline 已启动")

    def stop(self):
        """Stops all threads and releases resources."""
        log.info("[系统控制] >>> 停止 Pipeline")
        self.running = False
        for t in self.threads:
            t.join(timeout=2)
        self.cap.release()
        self.verifier.close()
        log.info(f"[系统控制] 统计信息: 已处理帧={self.processed_frames}, 丢弃帧={self.dropped_frames}")
        log.success("[系统控制] <<< Pipeline 已停止")

    def get_results(self):
        """Returns the latest processed frame and detections."""
        try:
            return self.result_queue.get(timeout=0.1)
        except Empty:
            return None, None

    def get_results_nowait(self):
        """非阻塞版本，供 async 环境使用，避免阻塞 event loop。"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None, None

    def get_performance_stats(self):
        """Returns current performance statistics including CPU/GPU memory."""
        import torch
        stats = {
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "last_inference_time_ms": round(self._last_inference_time, 2),
            "queue_size": self.frame_queue.qsize(),
        }
        # CPU 内存（无需额外依赖，用 os 模块）
        try:
            import psutil
            proc = psutil.Process()
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["ram_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            pass
        # GPU 显存（torch 自带，无需 pynvml）
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
        return stats
