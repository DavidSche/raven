import cv2
import base64
import time
import torch
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.pipeline import AsyncPipeline
from core.pipeline.manager import PipelineManager
from core.pipeline.pipeline_config import PipelineConfig
from core.config_manager import ConfigManager
from core.logger import setup_logging, get_logger
import uvicorn
import asyncio

log = get_logger("api")

WEB_DIR = Path(__file__).parent / "web"

manager: PipelineManager = None
pipeline: AsyncPipeline = None   # 始终指向 manager 的第一路流，供旧接口使用

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理，处理启动和停止事件."""
    global pipeline
    
    log.info("[API服务] >>> 服务启动中...")
    
    # 主线程提前初始化 CUDA context，避免子线程竞争
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device='cuda')
        log.info(f"[API服务] CUDA 已初始化 | 设备: {torch.cuda.get_device_name(0)}")
    
    cfg = ConfigManager.load_config()
    # 日志设置
    setup_logging(
        level=cfg.logging.level,
        file_enabled=cfg.logging.file.enabled,
        console_enabled=cfg.logging.console_enabled,
        log_dir=cfg.logging.file.path,
        rotation=cfg.logging.file.rotation,
        retention=cfg.logging.file.retention,
        compression=cfg.logging.file.compression,
    )
    
    # 视频流
    source = cfg.system.source
    try:
        source = int(source)
    except:
        pass
    
    log.info(f"[API服务] 初始化 PipelineManager | source={source}")
    manager = PipelineManager()
    manager.add_rtsp("default", source, PipelineConfig.from_global_config())
    pipeline = manager.get_single_pipeline()
    
    log.success("[API服务] <<< 服务启动完成")
    
    yield
    
    log.info("[API服务] >>> 服务关闭中...")
    if manager:
        manager.shutdown()
        log.success("[API服务] PipelineManager 已关闭")
    log.success("[API服务] <<< 服务已关闭")

app = FastAPI(title="Human Detection API", version="2.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all HTTP requests with timing."""
    if request.url.path.startswith("/ws/"):
        return await call_next(request)
    
    start_time = time.perf_counter()
    
    log.info(f"[HTTP请求] >>> {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        log.info(f"[HTTP请求] <<< {request.method} {request.url.path} | status={response.status_code} | elapsed={elapsed:.2f}ms")
        
        return response
    except Exception as e:
        log.error(f"[HTTP请求] <<< {request.method} {request.url.path} | error={e}")
        raise

def _serialize_detections(detections):
    """将 detections 转为可 JSON 序列化的格式，去除 numpy mask。"""
    result = []
    for det in (detections or []):
        result.append({
            "id": det["id"],
            "bbox": [float(v) for v in det["bbox"]],
            "conf": float(det["conf"]),
            "is_live": bool(det["is_live"]),
            "mask": det["mask"].tolist() if det.get("mask") is not None else None,
        })
    return result

@app.get("/")
def read_root():
    """Serve the web viewer page."""
    return FileResponse(WEB_DIR / "index.html")

@app.websocket("/ws/results")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time results via WebSocket with video stream."""
    log.info("[WebSocket] >>> 客户端连接请求")
    
    await websocket.accept()
    log.success(f"[WebSocket] 客户端已连接 | client={websocket.client}")
    
    await websocket.send_json({"type": "connected", "message": "WebSocket connected successfully"})
    
    frame_count = 0
    # 目标发送间隔：约 25fps，与摄像头帧率匹配
    SEND_INTERVAL = 1.0 / 25
    last_send_time = 0.0

    try:
        while True:
            now = asyncio.get_event_loop().time()

            # 非阻塞取最新帧，通过封装接口访问，不直接操作内部队列
            frame, detections = pipeline.get_results_nowait()
            if frame is None or detections is None:
                await asyncio.sleep(0.005)
                continue
                await asyncio.sleep(0.005)
                continue

            # 控制发送频率，避免短时间内连续发多帧撑爆前端渲染队列
            elapsed = now - last_send_time
            if elapsed < SEND_INTERVAL:
                await asyncio.sleep(SEND_INTERVAL - elapsed)
            last_send_time = asyncio.get_event_loop().time()

            # 图像编码（在 executor 里跑，不阻塞 event loop）
            h, w = frame.shape[:2]
            _, buffer = await asyncio.get_event_loop().run_in_executor(
                None, lambda: cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            )

            # 图像和检测数据合并成一条消息发送，消除两条消息的时序错位
            img_b64 = base64.b64encode(buffer.tobytes()).decode('ascii')
            await websocket.send_json({
                "type": "frame",
                "frame_width": w,
                "frame_height": h,
                "image": img_b64,
                "detections": _serialize_detections(detections)
            })

            frame_count += 1
            if frame_count % 100 == 0:
                log.trace(f"[WebSocket] 已发送 {frame_count} 帧")

    except WebSocketDisconnect:
        log.info(f"[WebSocket] <<< 客户端断开连接 | client={websocket.client}")
    except Exception as e:
        log.error(f"[WebSocket] 连接异常: {e}")

def gen_frames():
    """MJPEG 视频流生成器。"""
    log.debug("[视频流] 开始生成 MJPEG 流")
    frame_count = 0
    
    while True:
        frame, detections = pipeline.get_results()
        if frame is None:
            continue

        for det in (detections or []):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if det['is_live'] else (0, 255, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_count += 1
        if frame_count % 100 == 0:
            log.trace(f"[视频流] 已发送 {frame_count} 帧")
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    """MJPEG 视频流端点。"""
    log.debug("[API端点] GET /video_feed")
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/change_source")
def change_video_source(new_source: str):
    """动态切换默认视频源（热重启 default 流）。"""
    global pipeline
    log.info(f"[API端点] POST /change_source | new_source={new_source}")
    try:
        final_source = int(new_source)
    except Exception:
        final_source = new_source
    ok = manager.update_config(
        "default",
        PipelineConfig.from_global_config(),
    )
    if not ok:
        # default 流不存在时直接新建
        manager.add_rtsp("default", final_source, PipelineConfig.from_global_config())
    else:
        # 已重建，但源地址以 new_source 为准 —— 直接 remove+add
        manager.remove_rtsp("default")
        manager.add_rtsp("default", final_source, PipelineConfig.from_global_config())
    pipeline = manager.get_single_pipeline()
    log.success(f"[API端点] 视频源切换完成: {final_source}")
    return {"status": "success", "new_source": str(final_source)}


# ─── 多流管理端点（新增）───────────────────────────────

class StreamAddRequest(BaseModel):
    stream_id: str
    url: str
    enable_tracker: bool  = True
    enable_verifier: bool = True


@app.post("/streams/add", summary="动态添加一路流")
def stream_add(req: StreamAddRequest):
    cfg = PipelineConfig(
        enable_tracker=req.enable_tracker,
        enable_verifier=req.enable_verifier,
    )
    ok = manager.add_rtsp(req.stream_id, req.url, cfg)
    if not ok:
        raise HTTPException(status_code=409, detail=f"stream_id '{req.stream_id}' 已存在")
    return {"status": "ok", "stream_id": req.stream_id}


@app.post("/streams/remove/{stream_id}", summary="移除一路流")
def stream_remove(stream_id: str):
    ok = manager.remove_rtsp(stream_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"stream_id '{stream_id}' 不存在")
    return {"status": "ok", "stream_id": stream_id}


@app.get("/streams/status", summary="所有流性能统计")
def streams_status():
    return {"streams": manager.get_status()}

@app.get("/results")
def get_latest_results():
    """获取最新检测结果（JSON）。"""
    _, detections = pipeline.get_results()
    log.trace(f"[API端点] GET /results | detections={len(detections) if detections else 0}")
    return {"detections": _serialize_detections(detections)}

@app.get("/health")
def health_check():
    """Health check endpoint with performance stats."""
    stats = pipeline.get_performance_stats() if pipeline else {}
    log.trace(f"[API端点] GET /health | processed={stats.get('processed_frames', 0)}")
    return {
        "status": "ok",
        "gpu_available": pipeline.detector.device == "cuda" if pipeline else False,
        **stats,
    }

if __name__ == "__main__":
    log.info("[API服务] 启动目标检测服务器")
    uvicorn.run(app, host="0.0.0.0", port=8000)
