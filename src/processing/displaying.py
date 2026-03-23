import cv2
import numpy as np


_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188,
    0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
    0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
    0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500,
    1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
    0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000,
    0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
    0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571,
    0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0,
]).astype(np.float32).reshape(-1, 3)


def draw_detections(detections: list) -> np.ndarray:
    frame, _, _, results = detections
    orig_h, orig_w = frame.shape[:2]

    frame = frame.copy()
    for det in results:
        x1, y1, x2, y2 = det['bbox']
        scale_x = float(orig_w) / 640.0
        scale_y = float(orig_h) / 640.0
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        color = (_COLORS[det['class_id'] % len(_COLORS)] * 255).astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def display_multi_stream(all_detections: list, is_fullscreen: bool = False, fps: float = 15.0):
    n = len(all_detections)
    if n == 0:
        return is_fullscreen, None

    if n in (1, 2, 3, 5, 7):
        cols = n
        rows = 1
    else:
        cols = (n + 1) // 2
        rows = 2

    frames = []
    for detections in all_detections:
        frame = draw_detections(detections)
        frames.append(frame)

    h, w = frames[0].shape[:2]
    gap = 2
    total_h = rows * h + (rows + 1) * gap
    total_w = cols * w + (cols + 1) * gap

    canvas = np.full((total_h, total_w, 3), 40, dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        y = gap + r * (h + gap)
        x = gap + c * (w + gap)
        canvas[y:y+h, x:x+w] = frame

    window_name = "Multi-Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, total_w, total_h)

    cv2.imshow(window_name, canvas)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
    return is_fullscreen, key


def close_all_windows():
    cv2.destroyAllWindows()