"""
core/rtsp/reader.py

RTSP/视频文件拉流与重连管理器

专职处理：
1. 本地视频文件、USB摄像头、RTSP 网络流的拉取。
2. 优先尝试 GPU cudacodec，降级使用 CPU OpenCV。
3. 网络断线时的指数退避重连机。
4. 将解码后的画面放入 frame_queue（解耦 AI 算力调度）。
"""

import cv2
import time
import os
import threading
from queue import Queue, Empty
from core.logger import get_logger

log = get_logger("rtsp-reader")

# ─────────────────────────────────────────────
# 连接状态枚举
# ─────────────────────────────────────────────

class StreamState:
    CONNECTING   = "connecting"
    STREAMING    = "streaming"
    RECONNECTING = "reconnecting"
    FAILED       = "failed"

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
# 核心类：RtspReader
# ─────────────────────────────────────────────

class RtspReader:
    def __init__(
        self, 
        video_source, 
        frame_queue: Queue, 
        drop_frames: bool = True,
        max_retries: int = -1
    ):
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.drop_frames = drop_frames
        self.max_retries = max_retries
        
        self.running = False
        self.thread = None
        self.dropped_frames = 0
        
        self.state = StreamState.CONNECTING
        self._retry_count = 0
        
        # ── RTSP 低延迟参数 ──
        self.is_rtsp = isinstance(video_source, str) and video_source.startswith("rtsp://")
        if self.is_rtsp:
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

        self.state = StreamState.STREAMING

        # ── FPS / 文件源判断 ──
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_video_file = (
            isinstance(video_source, str)
            and not self.is_rtsp
            and self.fps > 0
        )
        self.frame_delay = 1.0 / self.fps if self.is_video_file else 0.001
        
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

    def start(self):
        """启动采流线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止采流，释放资源"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def _reconnect(self):
        """RTSP 断线重连，使用指数退避策略（最大等待 30s）。"""
        self.state = StreamState.RECONNECTING
        delay = min(2 ** self._retry_count, 30)
        
        log.warning(
            f"[数据采集] 等待 {delay}s 后重连 "
            f"(第 {self._retry_count + 1} 次) | source={self.video_source}"
        )
        time.sleep(delay)
        self._retry_count += 1

        # ── GPU 路径重连 ──
        if hasattr(self, 'cuda_reader') and self.cuda_reader is not None:
            try:
                self.cuda_reader = cv2.cudacodec.createVideoReader(self.video_source)
                self.cuda_reader.set(cv2.cudacodec.ColorFormat_BGR)
                log.success("[数据采集] GPU 读取器重连成功")
                self.state = StreamState.STREAMING
                self._retry_count = 0
            except Exception as e:
                log.error(f"[数据采集] GPU 重连失败，回退到 CPU: {e}")
                self.cuda_reader = None
                
                # 回退 CPU 逻辑
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if self.cap.isOpened():
                    self.state = StreamState.STREAMING
                    self._retry_count = 0
        # ── CPU 路径重连 ──
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if self.cap.isOpened():
                log.success("[数据采集] CPU 读取器重连成功")
                self.state = StreamState.STREAMING
                self._retry_count = 0
            else:
                log.error("[数据采集] 重连后视频源仍无法打开")

        # Max retries check
        if self.max_retries != -1 and self._retry_count > self.max_retries:
            log.error(f"[数据采集] 已超过最大重试次数 ({self.max_retries})，标记为 FAILED")
            self.state = StreamState.FAILED

        # Clear queue
        cleared = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                cleared += 1
            except Empty:
                break
        if cleared > 0:
            log.debug(f"[数据采集] 重连后清空队列积压: {cleared} 帧")

    def _reader_loop(self):
        """核心拉流循环"""
        log.info(
            f"[数据采集] >>> 读取线程启动 "
            f"(FPS 限制: {self.fps if self.is_video_file else 'None'})"
        )

        fail_count = 0
        corrupt_count = 0

        while self.running:
            if self.state == StreamState.FAILED:
                log.error("[数据采集] 流已进入 FAILED 状态，读取线程退出")
                break

            start_time = time.time()
            frame = None

            # ── GPU 解码 ──
            if getattr(self, 'cuda_reader', None) is not None:
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
            # ── CPU 解码 ──
            else:
                ret, frame = self.cap.read()
                if not ret:
                    fail_count += 1
                    if self.is_rtsp and fail_count > 30:
                        log.warning(f"[数据采集] CPU 连续读取失败 {fail_count} 次，触发重连")
                        self._reconnect()
                        fail_count = 0
                        corrupt_count = 0
                    elif self.is_video_file:
                        log.info("[数据采集] 视频文件播毕，重新连播")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.01)
                    continue
                fail_count = 0

            # ── 损坏帧检测 ──
            if self.is_rtsp and frame is not None:
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

            # ── 丢帧策略与入列 ──
            if self.frame_queue.full():
                if self.drop_frames or self.is_rtsp:
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except Empty:
                        pass
                else:
                    time.sleep(0.01)
                    continue

            self.frame_queue.put(frame)

            # 节流
            if self.is_video_file:
                elapsed = time.time() - start_time
                wait_time = max(0, self.frame_delay - elapsed)
                if wait_time > 0:
                    time.sleep(wait_time)
            elif not self.is_rtsp:
                time.sleep(0.001)

        log.info("[数据采集] <<< 读取线程已停止")
