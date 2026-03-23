import cv2
import numpy as np

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

_COLOR_CACHE = {
    i: tuple(int(c * 255) for c in _COLORS[i % len(_COLORS)]) for i in range(100)
}


class DisplayState:
    __slots__ = ("n", "cols", "rows", "h", "w", "gap", "total_h", "total_w", "canvas")

    def __init__(self):
        self.n = 0
        self.cols = 0
        self.rows = 0
        self.h = 0
        self.w = 0
        self.gap = 2
        self.total_h = 0
        self.total_w = 0
        self.canvas = None

    def update_layout(self, n: int, h: int, w: int):
        if self.n != n or self.h != h or self.w != w:
            self.n = n
            self.h = h
            self.w = w

            if n in (1, 2, 3, 5, 7):
                self.cols = n
                self.rows = 1
            else:
                self.cols = (n + 1) // 2
                self.rows = 2

            self.total_h = self.rows * h + (self.rows + 1) * self.gap
            self.total_w = self.cols * w + (self.cols + 1) * self.gap
            self.canvas = np.full((self.total_h, self.total_w, 3), 40, dtype=np.uint8)


_display_state = DisplayState()
_WINDOW_NAME = "Multi-Stream"


def draw_detections(
    frame: np.ndarray, orig_h: int, orig_w: int, results: list, infer_fps: float = None
) -> np.ndarray:
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        color = _COLOR_CACHE[det["class_id"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if infer_fps is not None:
        fps_text = f"FPS: {infer_fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return frame


def display_multi_stream(
    all_detections: list, is_fullscreen: bool = False, fps: float = 15.0
):
    n = len(all_detections)
    if n == 0:
        return is_fullscreen, None

    frames = []
    for detections in all_detections:
        frame, _, _, (results, infer_fps) = detections
        frame = draw_detections(frame.copy(), *frame.shape[:2], results, infer_fps)
        frames.append(frame)

    h, w = frames[0].shape[:2]

    _display_state.update_layout(n, h, w)

    _display_state.canvas[:] = 40
    gap = _display_state.gap
    cols = _display_state.cols

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        y = gap + r * (h + gap)
        x = gap + c * (w + gap)
        _display_state.canvas[y : y + h, x : x + w] = frame

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(
            _WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    else:
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW_NAME, _display_state.total_w, _display_state.total_h)

    cv2.imshow(_WINDOW_NAME, _display_state.canvas)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == ord("f"):
        is_fullscreen = not is_fullscreen
    return is_fullscreen, key


def close_all_windows():
    cv2.destroyAllWindows()
