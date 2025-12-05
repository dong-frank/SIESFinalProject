// src/api.js
import { API_BASE_URL } from './config.js';

/**
 * API 客户端：封装所有与后端交互的 fetch 请求
 */
export class ApiClient {
    static async _request(endpoint, options = {}) {
        try {
            const response = await fetch(`${API_BASE_URL}/${endpoint}`, options);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.msg || `API 调用失败: ${response.statusText}`);
            }
            return data;
        } catch (error) {
            console.error(`API Error on ${endpoint}:`, error);
            throw error;
        }
    }

    static getStatus() {
        return this._request('get_status');
    }

    static getDetections() {
        return this._request('get_detections');
    }

    static toggleCamera(enable) {
        return this._request('toggle_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enable }),
        });
    }

    static setTargetRegion(roi) {
        return this._request('set_target_region', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(roi),
        });
    }

    static cancelTracker() {
        return this._request('cancel_tracker', { method: 'POST' });
    }
}
