class HumanDetectionViewer {
    constructor() {
        this.ws = null;
        this.canvas = null;
        this.ctx = null;
        this.isConnected = false;
        this.frameCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;
        this.videoWidth = 1280;
        this.videoHeight = 720;

        // 渲染队列：只保留最新一帧，避免积压
        this.pendingFrame = null;   // { img, detections }
        this.rafId = null;
        this.resizeTimer = null;

        // canvas 实际绘制尺寸（与 CSS 尺寸解耦，避免频繁 reflow）
        this.drawWidth = 0;
        this.drawHeight = 0;
        this.scaleX = 1;
        this.scaleY = 1;

        // 目标列表缓存，避免无变化时重建 DOM
        this._lastTargetKey = '';

        this.initElements();
        this.initEventListeners();
        this.initCanvas();
        this.renderLoop();
    }

    initElements() {
        this.elements = {
            canvas: document.getElementById('detection-canvas'),
            connectionStatus: document.getElementById('connection-status'),
            fpsValue: document.getElementById('fps-value'),
            detectionCount: document.getElementById('detection-count'),
            liveCount: document.getElementById('live-count'),
            scanCount: document.getElementById('scan-count'),
            frameCount: document.getElementById('frame-count'),
            targetList: document.getElementById('target-list'),
            loadingOverlay: document.getElementById('loading-overlay'),
            errorOverlay: document.getElementById('error-overlay'),
            errorMessage: document.getElementById('error-message'),
            wsUrl: document.getElementById('ws-url'),
            videoWidth: document.getElementById('video-width'),
            videoHeight: document.getElementById('video-height'),
            connectBtn: document.getElementById('connect-btn'),
            disconnectBtn: document.getElementById('disconnect-btn'),
            reconnectBtn: document.getElementById('reconnect-btn')
        };
    }

    initEventListeners() {
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.reconnectBtn.addEventListener('click', () => {
            this.elements.errorOverlay.style.display = 'none';
            this.elements.loadingOverlay.style.display = 'flex';
            this.connect();
        });

        this.elements.videoWidth.addEventListener('change', () => {
            this.videoWidth = parseInt(this.elements.videoWidth.value) || 1280;
            this._scheduleResize();
        });

        this.elements.videoHeight.addEventListener('change', () => {
            this.videoHeight = parseInt(this.elements.videoHeight.value) || 720;
            this._scheduleResize();
        });

        // 防抖：窗口 resize 100ms 内只触发一次
        window.addEventListener('resize', () => this._scheduleResize());
    }

    _scheduleResize() {
        if (this.resizeTimer) clearTimeout(this.resizeTimer);
        this.resizeTimer = setTimeout(() => this._applyResize(), 100);
    }

    initCanvas() {
        this.canvas = this.elements.canvas;
        this.ctx = this.canvas.getContext('2d');
        // 初始化时立即计算一次尺寸
        this._applyResize();
    }

    // 只在真正需要时修改 canvas.width/height（触发 reflow 的操作）
    _applyResize() {
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight || 480;

        const aspectRatio = this.videoWidth / this.videoHeight;
        let w = containerWidth;
        let h = containerWidth / aspectRatio;

        if (h > containerHeight) {
            h = containerHeight;
            w = containerHeight * aspectRatio;
        }

        w = Math.floor(w);
        h = Math.floor(h);

        // canvas 尺寸没变就不写（避免触发 reflow），但 scale 仍需更新
        if (w !== this.drawWidth || h !== this.drawHeight) {
            this.drawWidth = w;
            this.drawHeight = h;
            this.canvas.width = w;
            this.canvas.height = h;
            this.drawPlaceholder();
        }

        this.scaleX = w / this.videoWidth;
        this.scaleY = h / this.videoHeight;
    }

    drawPlaceholder() {
        this.ctx.fillStyle = '#0f0f1a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#333';
        this.ctx.font = '16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('等待视频流...', this.canvas.width / 2, this.canvas.height / 2);
    }

    connect() {
        const url = this.elements.wsUrl.value.trim();
        if (!url) { this.showError('请输入 WebSocket URL'); return; }

        this.elements.loadingOverlay.style.display = 'flex';
        this.elements.errorOverlay.style.display = 'none';

        try {
            this.ws = new WebSocket(url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus(true);
                this.elements.connectBtn.disabled = true;
                this.elements.disconnectBtn.disabled = false;
            };

            this.ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    this._handleBinary(event.data);
                } else {
                    try {
                        this._handleJson(JSON.parse(event.data));
                    } catch (e) {
                        console.error('JSON parse error:', e);
                    }
                }
            };

            this.ws.onclose = (event) => {
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.elements.connectBtn.disabled = false;
                this.elements.disconnectBtn.disabled = true;
                if (event.code !== 1000) {
                    this.showError(`连接已断开 (code: ${event.code})`);
                }
            };

            this.ws.onerror = () => {
                this.showError('连接失败，请检查服务器是否运行');
                this.elements.loadingOverlay.style.display = 'none';
            };

        } catch (e) {
            this.showError('无效的 WebSocket URL');
            this.elements.loadingOverlay.style.display = 'none';
        }
    }

    disconnect() {
        if (this.ws) { this.ws.close(1000, 'User disconnect'); this.ws = null; }
        this.isConnected = false;
        this.updateConnectionStatus(false);
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;
        this.pendingFrame = null;
        this._updateTargetList([]);
        this.drawPlaceholder();
    }

    // 收到二进制消息（旧协议兼容）
    _handleBinary(data) {
        const blob = new Blob([data], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            if (!this.pendingFrame) this.pendingFrame = {};
            this.pendingFrame.img = img;
            URL.revokeObjectURL(url);
        };
        img.onerror = () => URL.revokeObjectURL(url);
        img.src = url;
    }

    // 收到 JSON 消息
    _handleJson(data) {
        if (data.type === 'connected') {
            this.elements.loadingOverlay.style.display = 'none';
            return;
        }

        // 新协议：图像 + 检测数据合并在一条消息，彻底消除时序错位
        if (data.type === 'frame') {
            const detections = data.detections || [];

            // 用服务端实际帧尺寸同步更新 scale，不依赖图像解码
            if (data.frame_width && data.frame_height) {
                const fw = data.frame_width;
                const fh = data.frame_height;
                if (fw !== this.videoWidth || fh !== this.videoHeight) {
                    this.videoWidth = fw;
                    this.videoHeight = fh;
                    this._applyResize();
                }
                this.scaleX = this.drawWidth / fw;
                this.scaleY = this.drawHeight / fh;
            }

            this.frameCount++;
            this._updateFps();
            this._updateStats(detections);
            this._updateTargetList(detections);

            // base64 解码图像，和 detections 打包成一帧提交渲染队列
            const img = new Image();
            img.onload = () => {
                // 覆盖 pendingFrame，始终只保留最新一帧
                this.pendingFrame = { img, detections };
            };
            img.src = 'data:image/jpeg;base64,' + data.image;
        }
    }

    // rAF 驱动的渲染循环，与 WebSocket 消息解耦，消除渲染抖动
    renderLoop() {
        this.rafId = requestAnimationFrame(() => this.renderLoop());

        const frame = this.pendingFrame;
        if (!frame || !frame.img || !frame.detections) return;

        // 消费帧
        this.pendingFrame = null;

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // 1. 绘制图像
        ctx.drawImage(frame.img, 0, 0, w, h);

        // 2. 绘制检测框
        frame.detections.forEach(det => this._drawDetection(det));

        // 3. HUD
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(6, 6, 200, 28);
        ctx.fillStyle = '#e94560';
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`FPS: ${this.fps}  Targets: ${frame.detections.length}`, 12, 25);
    }

    _updateFps() {
        const now = Date.now();
        const elapsed = now - this.lastFpsTime;
        if (elapsed >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / elapsed);
            this.elements.fpsValue.textContent = this.fps;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }

    _updateStats(detections) {
        let live = 0, scan = 0;
        detections.forEach(d => d.is_live ? live++ : scan++);
        this.elements.detectionCount.textContent = detections.length;
        this.elements.liveCount.textContent = live;
        this.elements.scanCount.textContent = scan;
        this.elements.frameCount.textContent = this.frameCount;
    }

    // 用 key 比较避免无变化时重建 DOM
    _updateTargetList(detections) {
        const key = detections.map(d => `${d.id}:${d.is_live}`).join(',');
        if (key === this._lastTargetKey) return;
        this._lastTargetKey = key;

        if (detections.length === 0) {
            this.elements.targetList.innerHTML = '<p class="no-targets">暂无检测目标</p>';
            return;
        }

        this.elements.targetList.innerHTML = detections.map(det => `
            <div class="target-item ${det.is_live ? 'live' : 'scan'}">
                <span class="target-id">ID #${det.id}</span>
                <span class="target-conf">${(det.conf * 100).toFixed(0)}%</span>
                <span class="target-status ${det.is_live ? 'live' : 'scan'}">${det.is_live ? 'LIVE' : 'SCAN'}</span>
            </div>
        `).join('');
    }

    _drawDetection(det) {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * this.scaleX, sy1 = y1 * this.scaleY;
        const sx2 = x2 * this.scaleX, sy2 = y2 * this.scaleY;
        const color = det.is_live ? '#00ff88' : '#ffd93d';

        // 绘制 mask（如果有）
        if (det.mask && det.mask.length > 0) {
            this.ctx.fillStyle = det.is_live ? 'rgba(0,255,136,0.25)' : 'rgba(255,217,61,0.25)';
            this.ctx.beginPath();
            this.ctx.moveTo(det.mask[0][0] * this.scaleX, det.mask[0][1] * this.scaleY);
            for (let i = 1; i < det.mask.length; i++) {
                this.ctx.lineTo(det.mask[i][0] * this.scaleX, det.mask[i][1] * this.scaleY);
            }
            this.ctx.closePath();
            this.ctx.fill();
        }

        // bbox
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

        // label
        const label = `ID #${det.id} ${det.is_live ? 'LIVE' : 'SCAN'} (${(det.conf * 100).toFixed(0)}%)`;
        this.ctx.font = 'bold 13px sans-serif';
        const tw = this.ctx.measureText(label).width;
        this.ctx.fillStyle = color;
        this.ctx.fillRect(sx1, sy1 - 20, tw + 10, 20);
        this.ctx.fillStyle = '#000';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(label, sx1 + 5, sy1 - 5);
    }

    updateConnectionStatus(connected) {
        const el = this.elements.connectionStatus;
        el.textContent = connected ? '已连接' : '未连接';
        el.className = 'status-value ' + (connected ? 'connected' : 'disconnected');
    }

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.loadingOverlay.style.display = 'none';
        this.elements.errorOverlay.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.viewer = new HumanDetectionViewer();
});
