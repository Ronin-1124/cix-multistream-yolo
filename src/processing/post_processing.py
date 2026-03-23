import numpy as np


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([])
    
    indices = scores.argsort()[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[indices[1:], 2] - boxes[indices[1:], 0]) * \
                    (boxes[indices[1:], 3] - boxes[indices[1:], 1])
        
        union = area_i + area_rest - intersection
        union = np.maximum(union, 1e-6)
        iou = intersection / union
        
        mask = iou <= iou_thr
        indices = indices[1:][mask]
    
    return np.array(keep)

def post_processing(
    pred: np.ndarray, 
    conf_thr: float, 
    iou_thr: float,
    class_names: list[str] = COCO_CLASSES
) -> list[dict]:
    
    if pred.ndim == 3:
        pred = pred[0]
    
    pred = pred.T
    
    boxes = pred[:, :4]
    scores = pred[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    scores = np.max(scores, axis=1)

    mask = scores > conf_thr
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    x_center, y_center, width, height = boxes.T
    
    x1 = (x_center - width/2)
    y1 = (y_center - height/2)
    x2 = (x_center + width/2)
    y2 = (y_center + height/2)
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    unique_classes = np.unique(class_ids)
    keep = []
    
    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        cls_keep = nms(cls_boxes, cls_scores, iou_thr)
        
        original_indices = np.where(cls_mask)[0][cls_keep]
        keep.extend(original_indices)
    
    keep = np.array(keep)
    keep = keep[scores[keep].argsort()[::-1]]
    
    results = []
    for idx in keep:
        result = {
            'bbox': boxes[idx].tolist(),
            'class_id': int(class_ids[idx]),
            'confidence': float(scores[idx]),
            'class_name': class_names[class_ids[idx]]
        }       
        results.append(result)

    return results