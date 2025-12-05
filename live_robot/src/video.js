// src/video.js
import { MJPEG_STREAM_URL, PLACEHOLDER_IMAGE } from './config.js';

/**
 * 视频流管理器
 */
export class VideoStream {
    constructor(selector) {
        this.element = document.querySelector(selector);
        this.element.onerror = () => this.showPlaceholder();
        this.showPlaceholder();
    }

    start() {
        if (this.element.src !== MJPEG_STREAM_URL) {
            this.element.src = MJPEG_STREAM_URL;
        }
    }

    stop() {
        this.showPlaceholder();
    }

    showPlaceholder() {
        if (this.element.src !== PLACEHOLDER_IMAGE) {
            this.element.src = PLACEHOLDER_IMAGE;
        }
    }

    getDimensions() {
        return {
            clientWidth: this.element.clientWidth,
            clientHeight: this.element.clientHeight,
            naturalWidth: this.element.naturalWidth,
            naturalHeight: this.element.naturalHeight,
        };
    }
}
