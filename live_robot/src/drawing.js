// src/drawing.js
import { ApiClient } from './api.js';

/**
 * Canvas 绘图控制器
 */
export class DrawingController {
    constructor(canvasSelector, videoStream) {
        this.canvas = document.querySelector(canvasSelector);
        this.ctx = this.canvas.getContext('2d');
        this.videoStream = videoStream;
        this.isDrawing = false;
        this.startPos = { x: 0, y: 0 };

        this.initEventListeners();
    }

    initEventListeners() {
        window.addEventListener('load', () => this.resize());
        window.addEventListener('resize', () => this.resize());
        this.canvas.addEventListener('mousedown', e => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', e => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', e => this.onMouseUp(e));
        this.canvas.addEventListener('mouseleave', () => this.onMouseLeave());
    }

    resize() {
        const { clientWidth, clientHeight } = this.videoStream.getDimensions();
        this.canvas.width = clientWidth;
        this.canvas.height = clientHeight;
    }

    getMousePos(event) {
        const rect = this.canvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    onMouseDown(e) {
        this.isDrawing = true;
        this.startPos = this.getMousePos(e);
    }

    onMouseMove(e) {
        if (!this.isDrawing) return;
        const currentPos = this.getMousePos(e);
        this.clear();
        this.ctx.strokeStyle = 'lime';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(this.startPos.x, this.startPos.y, currentPos.x - this.startPos.x, currentPos.y - this.startPos.y);
    }

    onMouseUp(e) {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        const endPos = this.getMousePos(e);

        const rect = {
            x1: Math.min(this.startPos.x, endPos.x),
            y1: Math.min(this.startPos.y, endPos.y),
            x2: Math.max(this.startPos.x, endPos.x),
            y2: Math.max(this.startPos.y, endPos.y),
        };

        if (rect.x2 - rect.x1 < 5 || rect.y2 - rect.y1 < 5) {
            this.clear();
            return;
        }

        ApiClient.setTargetRegion(this.scaleCoordinates(rect))

        setTimeout(() => this.clear(), 500);
    }

    onMouseLeave() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.clear();
        }
    }

    scaleCoordinates(rect) {
        const { clientWidth, clientHeight, naturalWidth, naturalHeight } = this.videoStream.getDimensions();
        const scaleX = naturalWidth / clientWidth;
        const scaleY = naturalHeight / clientHeight;
        return {
            x1: Math.round(rect.x1 * scaleX),
            y1: Math.round(rect.y1 * scaleY),
            x2: Math.round(rect.x2 * scaleX),
            y2: Math.round(rect.y2 * scaleY),
        };
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawDetections(detections) {
        if (this.isDrawing) return;
        this.clear();

        const { clientWidth, clientHeight, naturalWidth, naturalHeight } = this.videoStream.getDimensions();
        if (naturalWidth === 0 || naturalHeight === 0) return;

        const scaleX = clientWidth / naturalWidth;
        const scaleY = clientHeight / naturalHeight;

        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            this.drawFocusBox(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
        });
    }

    drawFocusBox(x, y, width, height) {
        this.ctx.strokeStyle = 'lime';
        this.ctx.lineWidth = 3;
        const cornerLength = Math.min(width, height) * 0.2;

        this.ctx.beginPath();
        this.ctx.moveTo(x + cornerLength, y); this.ctx.lineTo(x, y); this.ctx.lineTo(x, y + cornerLength);
        this.ctx.moveTo(x + width - cornerLength, y); this.ctx.lineTo(x + width, y); this.ctx.lineTo(x + width, y + cornerLength);
        this.ctx.moveTo(x, y + height - cornerLength); this.ctx.lineTo(x, y + height); this.ctx.lineTo(x + cornerLength, y + height);
        this.ctx.moveTo(x + width - cornerLength, y + height); this.ctx.lineTo(x + width, y + height); this.ctx.lineTo(x + width, y + height - cornerLength);
        this.ctx.stroke();
    }
}
