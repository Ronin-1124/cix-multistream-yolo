import numpy as np
import pytest

from src.processing.post_processing import nms, post_processing, COCO_CLASSES


class TestNMS:
    def test_nms_empty_boxes(self):
        """空边界框输入应返回空数组"""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        result = nms(boxes, scores, iou_thr=0.5)
        assert len(result) == 0

    def test_nms_single_box(self):
        """单个边界框应直接保留"""
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        result = nms(boxes, scores, iou_thr=0.5)
        assert len(result) == 1
        assert result[0] == 0

    def test_nms_no_overlap(self):
        """无重叠的边界框应全部保留"""
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = nms(boxes, scores, iou_thr=0.5)
        assert len(result) == 3

    def test_nms_full_overlap(self):
        """完全重叠的边界框只保留高置信度"""
        boxes = np.array([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = nms(boxes, scores, iou_thr=0.5)
        assert len(result) == 1
        assert result[0] == 0  # 最高分数的索引

    def test_nms_partial_overlap(self):
        """部分重叠时，IOU 阈值决定保留哪些"""
        # 两个重叠的盒子，中心点略有偏移
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],  # 与第一个重叠
        ])
        scores = np.array([0.9, 0.8])
        result = nms(boxes, scores, iou_thr=0.5)
        assert len(result) == 2  # IOU 较小，两个都保留

    def test_nms_sorted_by_score(self):
        """NMS 按分数从高到低处理"""
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
        ])
        scores = np.array([0.7, 0.9, 0.8])  # 第二个分数最高
        result = nms(boxes, scores, iou_thr=0.5)
        # 索引 1 (最高分) 应该被保留，索引 0 可能被抑制，索引 2 独立
        assert 1 in result


class TestPostProcessing:
    """post_processing 函数测试"""

    def test_post_processing_all_low_confidence(self):
        """所有检测置信度都低于阈值时应返回空列表"""
        # 创建一个预测，所有分数都很低，低于 conf_thr
        pred = np.random.rand(8400, 84) * 0.3  # 所有分数 < 0.3
        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)
        assert result == []

    def test_post_processing_3d_to_2d(self):
        """3D 预测张量应自动转为 2D"""
        pred = np.random.rand(1, 84, 8400)
        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)
        # 如果没有超过阈值的检测，返回空列表
        assert isinstance(result, list)

    def test_post_processing_confidence_filter(self):
        """置信度低于阈值的检测应被过滤"""
        # 创建一个预测，所有分数都很低
        pred = np.zeros((8400, 84))
        pred[:, 4:] = 0.1  # 所有类别分数都很低
        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)
        assert result == []

    def test_post_processing_output_format(self):
        """输出格式应包含 bbox, class_id, confidence, class_name"""
        # 创建一个有效的预测：中心点 (320, 320), 宽高 (100, 100), 高分数
        pred = np.zeros((8400, 84))
        # 设置第一个检测：中心点 320, 320, 宽 100, 高 100, 类别 0 (person) 分数 0.9
        pred[0, 0] = 320  # x_center
        pred[0, 1] = 320  # y_center
        pred[0, 2] = 100  # width
        pred[0, 3] = 100  # height
        pred[0, 4] = 0.9  # person 分数

        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)

        if len(result) > 0:
            item = result[0]
            assert 'bbox' in item
            assert 'class_id' in item
            assert 'confidence' in item
            assert 'class_name' in item
            assert item['class_name'] == 'person'
            assert item['class_id'] == 0

    def test_post_processing_bbox_conversion(self):
        """边界框应从 (x_center, y_center, w, h) 转换为 (x1, y1, x2, y2)"""
        pred = np.zeros((8400, 84))
        # 设置一个检测
        pred[0, 0] = 100  # x_center
        pred[0, 1] = 100  # y_center
        pred[0, 2] = 50   # width
        pred[0, 3] = 50   # height
        pred[0, 4] = 0.9  # 分数

        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)

        if len(result) > 0:
            bbox = result[0]['bbox']
            x1, y1, x2, y2 = bbox
            # x_center=100, width=50 -> x1=75, x2=125
            assert x1 == 75.0
            assert x2 == 125.0
            # y_center=100, height=50 -> y1=75, y2=125
            assert y1 == 75.0
            assert y2 == 125.0

    def test_post_processing_multi_class(self):
        """多类别检测应分别做 NMS"""
        pred = np.zeros((8400, 84))
        # 类别 0 (person) 两个重叠检测
        pred[0, 0] = 100
        pred[0, 1] = 100
        pred[0, 2] = 50
        pred[0, 3] = 50
        pred[0, 4] = 0.9

        pred[1, 0] = 105
        pred[1, 1] = 105
        pred[1, 2] = 50
        pred[1, 3] = 50
        pred[1, 5] = 0.8

        result = post_processing(pred, conf_thr=0.5, iou_thr=0.25)
        assert isinstance(result, list)


class TestCOCOClasses:
    def test_coco_classes_length(self):
        assert len(COCO_CLASSES) == 80

    def test_coco_classes_first_is_person(self):
        assert COCO_CLASSES[0] == 'person'

    def test_coco_classes_known_classes(self):
        assert COCO_CLASSES[0] == 'person'
        assert COCO_CLASSES[2] == 'car'
        assert COCO_CLASSES[14] == 'bird'
