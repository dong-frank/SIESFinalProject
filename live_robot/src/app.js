// src/app.js
import { ApiClient } from './api.js';
import { VideoStream } from './video.js';
import { StatusManager } from './status.js';
import { DrawingController } from './drawing.js';

/**
 * 主应用程序
 */
export class App {
    constructor() {
        this.videoStream = new VideoStream('#live-video');
        this.statusManager = new StatusManager(this.videoStream);
        this.drawingController = new DrawingController('#drawing-canvas', this.videoStream);

        document.querySelector('#cancel-tracker-btn').addEventListener('click', this.cancelTracking);
    }

    async cancelTracking() {
        try {
            const result = await ApiClient.cancelTracker();
        } catch (error) {
        }
    }

    startDetectionLoop(interval = 100) {
        setInterval(async () => {
            if (this.statusManager.status.camera_enabled) {
                try {
                    const detections = await ApiClient.getDetections();
                    this.drawingController.drawDetections(detections);
                } catch (error) {
                    // 静默处理错误
                }
            } else {
                this.drawingController.clear();
            }
        }, interval);
    }

    run() {
        this.statusManager.startPolling();
        this.startDetectionLoop();
    }
}
