"""
core/pipeline/async_pipeline.py

AsyncPipeline — 异步视频处理管道（V2 终极架构重构版）
将原先的深度耦合结构，抽离为基于 Stage 插件扩展的数据流模型。
解决痛点：单例模型共享节约显存 + 5秒无帧软重连保护 + 未来功能级插拔。
"""
import threading
import time
from queue import Queue, Empty
from core.config_manager import ConfigManager, PipelineConfig
from core.rtsp.base_reader import BaseReader
from core.rtsp.reader_opencv import OpenCVReader

from core.pipeline.stages.detector_stage import DetectorStage
from core.pipeline.stages.tracker_stage import TrackerStage
from core.pipeline.stages.verifier_stage import VerifierStage

from core.logger import get_logger

log = get_logger("pipeline-v2")


class AsyncPipeline:
    """
    异步视频处理管道（V2 终极架构）。
    """

    def __init__(self, video_source=0, config: PipelineConfig = None):
        log.info("[数据采集] >>> 初始化 AsyncPipeline V2")

        self.video_source = video_source
        self.cfg_runtime = config or ConfigManager.get_config().pipeline

        # ── 队列（大小由配置控制）──
        self.frame_queue  = Queue(maxsize=self.cfg_runtime.frame_queue_size)
        self.result_queue = Queue(maxsize=self.cfg_runtime.result_queue_size)
        self.running      = False

        # ── 统计 ──
        self.processed_frames    = 0
        self._last_inference_time = 0.0

        # ── 读流器 ──
        self.reader: BaseReader = OpenCVReader(
            video_source=self.video_source,
            frame_queue=self.frame_queue,
            drop_frames=self.cfg_runtime.drop_frames,
            max_retries=self.cfg_runtime.max_retries
        )

        # ── 插件链 (Stages) ──
        self.stages = []

        if self.cfg_runtime.enable_detector:
            self.stages.append(DetectorStage())

        if self.cfg_runtime.enable_tracker:
            self.stages.append(TrackerStage())

        if self.cfg_runtime.enable_verifier:
            self.stages.append(VerifierStage(enable_verifier=True))
        else:
            self.stages.append(VerifierStage(enable_verifier=False))

        log.success(f"[数据采集] 核心组件 Stage 插件配置完成，数量={len(self.stages)}")
        log.success("[数据采集] <<< AsyncPipeline V2 初始化完成")

    # ─────────────────────────────────────────
    # 内部：Worker 线程 (V2版)
    # ─────────────────────────────────────────

    def _worker(self):
        """流转插件流线程。"""
        log.info("[推理处理] >>> 插件流转引擎启动")
        
        last_frame_time = time.time()

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                # 更新活跃时间
                last_frame_time = time.time()
            except Empty:
                # ── RTSP 卡死检测 ──
                if self.reader.state != "failed" and time.time() - last_frame_time > 5.0:
                    log.warning("[Pipeline-V2] 5秒无帧读出，疑似底层流卡死，强制触发 _reconnect()")
                    self.reader._reconnect()
                    last_frame_time = time.time()
                continue
            
            inference_start = time.perf_counter()
            data = {"frame": frame}

            try:
                for stage in self.stages:
                    data = stage.process(data)
                
                # 提取末端的最终标准化数据
                detections = data.get("detections", [])
                
                # ── 写入结果队列 ──
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put((frame, detections))
                self.processed_frames += 1

                inference_time = (time.perf_counter() - inference_start) * 1000
                self._last_inference_time = inference_time

            except Exception as e:
                log.error(f"[Pipeline-V2] Stage 流转执行异常: {e}")
                import traceback
                traceback.print_exc()

        log.info("[推理处理] <<< 插件流转引擎退出")

    # ─────────────────────────────────────────
    # 外部：控制与获取接口
    # ─────────────────────────────────────────

    def start(self):
        """启动 Pipeline"""
        log.info("[系统控制] >>> 启动 Pipeline V2")
        self.running = True

        # 1. 启动专职 Reader
        self.reader.start()

        # 2. 启动 Worker 守护线程
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        log.success("[系统控制] <<< Pipeline V2 已启动")

    def stop(self):
        """停止所有线程并释放资源。"""
        log.info("[系统控制] >>> 停止 Pipeline")
        self.running = False
        
        self.reader.stop()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
        log.info(
            f"[系统控制] 统计信息: "
            f"已处理帧={self.processed_frames}, 丢弃帧={self.reader.dropped_frames}"
        )
        log.success("[系统控制] <<< Pipeline 已停止")

    def get_results(self, timeout=None):
        """阻塞式获取结果。返回 (frame, processed_list)。"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None, None

    def get_results_nowait(self):
        """非阻塞式获取结果。返回 (frame, processed_list)。"""
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
