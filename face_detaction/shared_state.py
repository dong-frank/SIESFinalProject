import threading

# --- 视频帧共享 ---
frame_buffer = None
buffer_lock = threading.Lock()

# --- 状态控制 ---
camera_enabled = True
state_lock = threading.Lock()

# --- 检测结果共享 (AI -> Web) ---
latest_detections = []
detection_lock = threading.Lock()

# --- 交互控制 (Web -> AI) ---
user_roi = None             # 用户画的框 (x1, y1, x2, y2)
roi_timeout_counter = 0     # ROI 搜索超时计数器
locked_track_id = None      # 当前锁定的目标 ID
interaction_lock = threading.Lock()