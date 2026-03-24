"""
core/pipeline/multi_stream.py

MultiStreamPipeline — 多路 Pipeline 线程安全容器

职责：
  - 维护 stream_id → AsyncPipeline 映射
  - 线程安全地执行 add / remove / stop_all 操作
"""
import time

import threading
from typing import Dict, Optional
from core.pipeline.async_pipeline import AsyncPipeline
from core.config_manager import PipelineConfig
from core.logger import get_logger

log = get_logger("multi-stream")


class MultiStreamPipeline:
    """线程安全的多路 Pipeline 字典封装。"""

    def __init__(self):
        self.pipelines: Dict[str, AsyncPipeline] = {}
        self.lock = threading.Lock()

    def add_stream(
        self,
        stream_id: str,
        source,
        config: PipelineConfig,
    ) -> bool:
        """
        添加并启动一路新的 Pipeline。

        Returns:
            True  — 添加成功
            False — stream_id 已存在
        """
        with self.lock:
            if stream_id in self.pipelines:
                log.warning(f"[多流] stream 已存在，忽略重复添加: {stream_id}")
                return False

            log.info(f"[多流] 添加流: {stream_id} -> {source}")
            pipeline = AsyncPipeline(video_source=source, config=config)
            pipeline.start()
            self.pipelines[stream_id] = pipeline
            log.success(f"[多流] 流已启动: {stream_id}")
            return True

    def remove_stream(self, stream_id: str) -> bool:
        """
        停止并移除指定流。

        Returns:
            True  — 移除成功
            False — stream_id 不存在
        """
        with self.lock:
            pipeline = self.pipelines.get(stream_id)
            if pipeline is None:
                log.warning(f"[多流] stream 不存在，无法移除: {stream_id}")
                return False

            log.info(f"[多流] 移除流: {stream_id}")
            pipeline.stop()
            time.sleep(0.2)  # 等待底层资源完全释放
            del self.pipelines[stream_id]
            log.success(f"[多流] 流已移除: {stream_id}")
            return True

    def get_stream(self, stream_id: str) -> Optional[AsyncPipeline]:
        """返回指定 stream_id 对应的 AsyncPipeline，不存在则返回 None。"""
        return self.pipelines.get(stream_id)

    def list_streams(self) -> list:
        """返回当前所有流的 stream_id 列表（快照）。"""
        with self.lock:
            return list(self.pipelines.keys())

    def stop_all(self):
        """停止并清空所有流。"""
        log.info("[多流] 停止所有流...")
        with self.lock:
            for sid, pipeline in list(self.pipelines.items()):
                log.info(f"[多流] 停止: {sid}")
                pipeline.stop()

            time.sleep(0.2)
            self.pipelines.clear()
        log.success("[多流] 所有流已停止")
