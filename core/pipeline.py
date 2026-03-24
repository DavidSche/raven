"""
core/pipeline.py — 向后兼容 shim

旧代码的 `from core.pipeline import AsyncPipeline` 仍可正常工作。
新代码请直接从 core.pipeline 包（core/pipeline/）中导入：
    from core.pipeline import AsyncPipeline, PipelineConfig, PipelineManager
"""
# noqa: F401
from core.pipeline.async_pipeline import AsyncPipeline
from core.pipeline.pipeline_config import PipelineConfig
from core.pipeline.multi_stream import MultiStreamPipeline
from core.pipeline.manager import PipelineManager

__all__ = ["AsyncPipeline", "PipelineConfig", "MultiStreamPipeline", "PipelineManager"]
