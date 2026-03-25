"""
core/pipeline/light_pipeline.py

轻量工业版 Pipeline（BM1684X / NVIDIA 通用）
特点：
- 单 Worker
- 无 Stage
- 低延迟
- 易维护
"""

import threading
import time
from queue import Queue, Empty

from core.config_manager import ConfigManager
from core.logger import get_logger
from core.ai.model_registry import ModelRegistry

log = get_logger("light-pipeline")


class LightPipeline:
    def __init__(self, video_source=0):
        log.info("[Pipeline] >>> 初始化轻量 Pipeline")

        cfg = ConfigManager.get_config()

        self.video_source = video_source
        self.frame_queue = Queue(maxsize=cfg.pipeline.frame_queue_size)

        self.running = False

        # ── Reader（自动适配平台）──
        from core.rtsp.reader_opencv import OpenCVReader
        self.reader = OpenCVReader(
            video_source=self.video_source,
            frame_queue=self.frame_queue,
            drop_frames=True
        )

        # ── 模型（单例）──
        self.detector = ModelRegistry.get_detector()

        # 可选模块
        from core.ai.tracker_bytetrack import ByteTracker
        from core.ai.verifier_default import DefaultVerifier

        self.tracker = ByteTracker() if cfg.pipeline.enable_tracker else None
        self.verifier = DefaultVerifier() if cfg.pipeline.enable_verifier else None

        # ── 状态 ──
        self.latest_result = None
        self.processed_frames = 0
        self.last_infer_time = 0

        log.success("[Pipeline] <<< 初始化完成")

    # ─────────────────────────────
    # Worker
    # ─────────────────────────────
    def _worker(self):
        log.info("[Pipeline] >>> Worker 启动")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            start = time.perf_counter()

            try:
                # 1️⃣ 检测
                result = self.detector.detect(frame)

                # 2️⃣ 跟踪
                if self.tracker:
                    result = self.tracker.update(result)

                # 3️⃣ 验证
                if self.verifier:
                    result = self.verifier.process(result)

                # 4️⃣ 保存最新结果（无 queue）
                self.latest_result = (frame, result)

                self.processed_frames += 1
                self.last_infer_time = (time.perf_counter() - start) * 1000

            except Exception as e:
                log.error(f"[Pipeline] 推理异常: {e}")

        log.info("[Pipeline] <<< Worker 退出")

    # ─────────────────────────────
    # 控制
    # ─────────────────────────────
    def start(self):
        log.info("[Pipeline] >>> 启动")
        self.running = True

        self.reader.start()

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        log.info("[Pipeline] >>> 停止")
        self.running = False

        self.reader.stop()

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

        log.success("[Pipeline] <<< 已停止")

    # ─────────────────────────────
    # 获取结果（最新帧）
    # ─────────────────────────────
    def get_latest(self):
        return self.latest_result

    # ─────────────────────────────
    # 性能统计
    # ─────────────────────────────
    def stats(self):
        return {
            "processed_frames": self.processed_frames,
            "last_infer_ms": round(self.last_infer_time, 2),
            "queue_size": self.frame_queue.qsize(),
            "dropped_frames": self.reader.dropped_frames,
            "state": self.reader.state
        }