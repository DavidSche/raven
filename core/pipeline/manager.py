"""
core/pipeline/manager.py

PipelineManager — 工业级Pipeline管理器

职责：
  - 生命周期管理（启动 / 停止 / 热重启）
  - 动态增删 RTSP 流
  - 状态查询（性能统计汇总）
  - 兼容单流模式（get_single_pipeline()）
"""
import threading
from typing import Dict, List, Optional
from core.pipeline.multi_stream import MultiStreamPipeline
from core.config_manager import PipelineConfig
from core.pipeline.async_pipeline import AsyncPipeline
from core.logger import get_logger

log = get_logger("pipeline-manager")


class PipelineManager:
    """
    工业级 Pipeline 管理器。

    单流使用（兼容 api_server.py / main.py 原有模式）：
        manager = PipelineManager()
        manager.add_rtsp("default", source, PipelineConfig.from_global_config())
        pipeline = manager.get_single_pipeline()

    多流使用：
        manager.add_rtsp("cam1", rtsp_url_1, cfg1)
        manager.add_rtsp("cam2", rtsp_url_2, cfg2)
        frame, dets = manager.get_frame("cam1")
    """

    def __init__(self):
        self.multi = MultiStreamPipeline()
        self.stream_configs: Dict[str, PipelineConfig] = {}
        self.stream_sources: Dict[str, str]            = {}
        self.stream_states: Dict[str, str]             = {}
        self.lock = threading.Lock()

    # ─────────────────────────────────────────
    # 流管理
    # ─────────────────────────────────────────

    def add_rtsp(
        self,
        stream_id: str,
        url,
        config: PipelineConfig,
    ) -> bool:
        """添加一路流并启动 Pipeline。"""
        with self.lock:
            ok = self.multi.add_stream(stream_id, url, config)
            if ok:
                self.stream_configs[stream_id] = config
                self.stream_sources[stream_id] = str(url)
                self.stream_states[stream_id]  = "CONNECTING"
                log.success(f"[管理器] 添加成功: {stream_id}")
            return ok

    def remove_rtsp(self, stream_id: str) -> bool:
        """停止并移除一路流。"""
        with self.lock:
            ok = self.multi.remove_stream(stream_id)
            if ok:
                self.stream_configs.pop(stream_id, None)
                self.stream_sources.pop(stream_id, None)
                self.stream_states.pop(stream_id, None)
                log.success(f"[管理器] 移除成功: {stream_id}")
            return ok

    def update_config(
        self,
        stream_id: str,
        new_config: PipelineConfig,
    ) -> bool:
        """
        热更新策略：停止旧 Pipeline → 用新配置重启。
        源地址从 stream_sources 中读取，无需外部传入。
        """
        with self.lock:
            source = self.stream_sources.get(stream_id)
            if source is None:
                log.error(f"[管理器] 更新失败，stream 不存在: {stream_id}")
                return False

            log.info(f"[管理器] 重启流以应用新配置: {stream_id}")
            self.multi.remove_stream(stream_id)
            ok = self.multi.add_stream(stream_id, source, new_config)
            if ok:
                self.stream_configs[stream_id] = new_config
                self.stream_states[stream_id]  = "CONNECTING"
                log.success(f"[管理器] 配置热更新完成: {stream_id}")
            return ok

    # ─────────────────────────────────────────
    # 帧获取
    # ─────────────────────────────────────────

    def get_frame(self, stream_id: str):
        """非阻塞获取指定流的最新帧和检测结果。"""
        pipeline = self.multi.get_stream(stream_id)
        if pipeline is None:
            return None, None
        return pipeline.get_results_nowait()

    def get_single_pipeline(self) -> Optional[AsyncPipeline]:
        """
        兼容单流模式：返回第一路 Pipeline 实例。
        供 api_server.py / main.py 使用，透明替代原先直接持有的 AsyncPipeline。
        """
        keys = self.multi.list_streams()
        if not keys:
            return None
        return self.multi.get_stream(keys[0])

    # ─────────────────────────────────────────
    # 状态查询
    # ─────────────────────────────────────────

    def list_streams(self) -> List[str]:
        """返回所有流 ID 列表。"""
        return self.multi.list_streams()

    def get_status(self) -> dict:
        """
        返回所有流的性能统计汇总。

        Returns:
            {stream_id: {processed_frames, dropped_frames, last_inference_time_ms,
                         queue_size, stream_state, ...}}
        """
        status = {}
        for sid in self.multi.list_streams():
            pipeline = self.multi.get_stream(sid)
            if pipeline is not None:
                stats = pipeline.get_performance_stats()
                stats["source"] = self.stream_sources.get(sid, "")
                self.stream_states[sid] = stats.get("stream_state", "unknown")
                stats["manager_state"] = self.stream_states[sid]
                status[sid] = stats
        return status

    # ─────────────────────────────────────────
    # 关闭
    # ─────────────────────────────────────────

    def shutdown(self):
        """停止并释放所有流资源。"""
        log.info("[管理器] 关闭所有流...")
        self.multi.stop_all()
        log.success("[管理器] 所有流已关闭")
