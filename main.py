import cv2
import time
import numpy as np
import torch
from core.pipeline import AsyncPipeline
from core.config_manager import ConfigManager
from core.logger import setup_logging, get_logger

log = get_logger("main")

def _warmup_cuda():
    """在主线程提前初始化 CUDA context，避免子线程首次使用时竞争导致崩溃。"""
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device='cuda')
        log.info(f"[主程序] CUDA 已初始化 | 设备: {torch.cuda.get_device_name(0)}")

def main():
    # 1. Initialize Configuration
    log.info("[主程序] >>> 程序启动")
    _warmup_cuda()
    
    log.info("[主程序] 加载配置文件...")
    cfg = ConfigManager.load_config()
    
    setup_logging(
        level=cfg.logging.level,
        file_enabled=cfg.logging.file.enabled,
        console_enabled=cfg.logging.console_enabled,
        log_dir=cfg.logging.file.path,
        rotation=cfg.logging.file.rotation,
        retention=cfg.logging.file.retention,
        compression=cfg.logging.file.compression,
    )
    
    log.info(f"[主程序] 配置加载完成 | model={cfg.model.path} | imgsz={cfg.model.imgsz}")
    
    # 2. Initialize Pipeline
    log.info("[主程序] 初始化 Pipeline...")
    source = cfg.system.source
    # Handle int or str (RTSP/File)
    try:
        source = int(source)
    except:
        pass
        
    pipeline = AsyncPipeline(video_source=source, drop_frames=(not isinstance(source, int)))
    pipeline.start()
    
    log.success("Human Detection System v2 (Production Ready)")
    log.info(f"Model: {cfg.model.path} | Imgsz: {cfg.model.imgsz}")
    log.info("Press 'q' to quit (Resize window with mouse supported).")
    
    # 3. Create a resizable window
    win_name = "Human Detection System v2"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    has_started = False
    last_display_frame = None  # 重连期间冻结显示最后一帧
    stats_counter = 0
    log_fps_interval = cfg.logging.performance.log_fps_interval

    try:
        while True:
            # 1. Get next processed frame
            frame, detections = pipeline.get_results()
            
            if frame is not None:
                has_started = True
                fps_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # Process and draw
                display_frame = frame.copy()
                # mask_overlay 仅分割模式需要，按需创建避免无效内存拷贝
                mask_overlay = frame.copy() if cfg.model.segmentation else None
                has_masks = False

                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    is_live = det['is_live']
                    color = (0, 255, 0) if is_live else (0, 255, 255)
                    
                    # 1. DRAW MASK（仅 segmentation 模式）
                    if cfg.model.segmentation and det.get('mask') is not None:
                        has_masks = True
                        points = np.int32([det['mask']])
                        cv2.fillPoly(mask_overlay, points, color)                    
                    # 2. DRAW BBOX & LABEL
                    label = f"ID#{det['id']} {'LIVE' if is_live else 'SCAN'} ({det['conf']:.2f})"
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(display_frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Apply Alpha Blending if masks were drawn
                if has_masks:
                    alpha = 0.3
                    cv2.addWeighted(mask_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

                # Draw Status
                mode_label = "SEGMENTATION" if cfg.model.segmentation else "DETECTION"
                cv2.putText(display_frame, f"FPS: {fps:.1f} ({mode_label})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # 定期输出统计信息
                stats_counter += 1
                if stats_counter % log_fps_interval == 0:
                    stats = pipeline.get_performance_stats()
                    log.info(f"[性能统计] FPS: {fps:.1f} | 已处理: {stats['processed_frames']} | 丢弃: {stats['dropped_frames']} | 推理耗时: {stats['last_inference_time_ms']:.2f}ms")
                
                # 2. Render logic ONLY on new frame
                if cfg.system.show_ui:
                    # 窗口被关闭后重建时需重新设置为 WINDOW_NORMAL，否则无法调整大小
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(win_name, display_frame)
                last_display_frame = display_frame
            
            elif has_started and last_display_frame is not None:
                # 重连期间：冻结显示最后一帧，叠加 Reconnecting 提示，避免黑屏
                reconnect_frame = last_display_frame.copy()
                cv2.putText(reconnect_frame, "Reconnecting...", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, reconnect_frame)
                log.warning("[主程序] 等待重连...")

            elif not has_started:
                # Show loading screen only before the first frame arrives
                loading_screen = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(loading_screen, "Initializing Engine (Please wait)...", (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(win_name, loading_screen)
                
            # get_results() 内部已有 100ms timeout 阻塞，waitKey 只需最小值保证 UI 事件响应
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("[主程序] 用户请求退出")
                break
                
    except KeyboardInterrupt:
        log.info("[主程序] 收到中断信号")
    except Exception as e:
        log.error(f"[主程序] 运行异常: {e}", exc_info=True)
    finally:
        log.info("[主程序] >>> 开始清理资源")
        pipeline.stop()
        cv2.destroyAllWindows()
        log.success("[主程序] <<< 程序已退出")

if __name__ == "__main__":
    main()
