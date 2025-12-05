// src/status.js
import { ApiClient } from './api.js';

/**
 * 状态管理器
 */
export class StatusManager {
    constructor(videoStream) {
        this.videoStream = videoStream;
        this.status = { camera_enabled: false };
        this.ui = {
            cameraBtn: document.querySelector('#toggle-camera-btn'),
            cameraStatus: document.querySelector('#camera-status'),
        };
        this.ui.cameraBtn.addEventListener('click', () => this.toggleCamera());
    }

    async toggleCamera() {
        try {
            await ApiClient.toggleCamera(!this.status.camera_enabled);
            await this.update();
        } catch (error) {
        }
    }

    async update() {
        try {
            this.status = await ApiClient.getStatus();
        } catch (error) {
            this.status = { camera_enabled: false };
        } finally {
            this.updateUI();
        }
    }

    updateUI() {
        const { camera_enabled } = this.status;
        this.ui.cameraStatus.textContent = camera_enabled ? '开启' : '关闭';
        this.ui.cameraStatus.classList.toggle('status-on', camera_enabled);
        this.ui.cameraStatus.classList.toggle('status-off', !camera_enabled);
        this.ui.cameraBtn.textContent = camera_enabled ? '关闭摄像头' : '开启摄像头';

        camera_enabled ? this.videoStream.start() : this.videoStream.stop();
    }

    startPolling(interval = 5000) {
        this.update();
        setInterval(() => this.update(), interval);
    }
}
