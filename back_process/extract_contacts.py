import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
import os.path as osp
from extract_mask import ImageSegmenter
from extract_bboxes import ImageDetector


class TangentSolver:
    def __init__(self, masks: np.ndarray, composed_mask: np.ndarray, bboxes: List[List[int]]):
        self.masks = masks
        self.composed_mask = composed_mask
        self.bboxes = bboxes
        self.tangent_points = []

    def __call__(self, *args, **kwargs) -> List:
        for bbox in self.bboxes:
            self._find_bbox_direction(bbox)
            self._find_bbox_owner(bbox)
            tangent = self._find_tangent_point(bbox)
            self.tangent_points.append(tangent)
        return self.tangent_points

    def _find_bbox_owner(self, bbox: List[int]) -> None:

        center_x, center_y = self._find_center_point(bbox)

        for mask_id, mask in enumerate(self.masks):
            # 检查中心点是否在车辆掩码内部
            if mask[center_y, center_x] == 1:
                # 绘制检测框
                bbox.append(mask_id)
                break

    def _find_bbox_direction(self, bbox: List[int]) -> None:
        start_row = bbox[1]  # x1: wheel_box[0], x2: wheel_box[2], y1: wheel_box[1], y2: wheel_box[3]
        end_row = bbox[3]
        col_1 = bbox[0]
        col_2 = bbox[2]

        sum_line_1 = np.sum(self.composed_mask[start_row:end_row + 1, col_1])
        sum_line_2 = np.sum(self.composed_mask[start_row:end_row + 1, col_2])

        if sum_line_1 > sum_line_2:  # line1 为左侧竖线，line2 为右侧竖线
            bbox.append(0)  # 0代表左侧竖线为内侧，1代表左侧竖线为外侧
        else:
            bbox.append(1)

    def _find_tangent_point(self, bbox: List[int], threshold: int = 20) -> Optional[Tuple[int, int]]:
        """
        找到车轮检测框底边与车辆掩码的切点，作为车辆接地点
        Args:
            bbox: 车轮检测框
            threshold: 检测线长度阈值

        Returns: 切点坐标

        """
        # x1,y1为bbox左上角坐标，x2,y2为右下角坐标
        x1, y1, x2, y2, direction = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

        # 定义检测线长度,若检测框较窄，则其为整个底边长；否则为一半底边长
        bbox_width = x2 - x1
        if bbox_width > threshold:
            # 定义检测线段的起点,终点与起始高度
            detection_line_length = bbox_width // 2
        else:
            detection_line_length = bbox_width

        if direction == 1:
            detection_line_left_col = x1
            detection_line_right_col = detection_line_left_col + detection_line_length
        else:
            detection_line_left_col = x2 - detection_line_length
            detection_line_right_col = x2

        # 定义边界行
        top_line_row = y1  # 最高行号
        array_rows, _ = self.composed_mask.shape
        end_line_row = array_rows - 1  # 最低行号

        # 定义当前行号
        current_row = y2

        # 滑动窗口检测
        while top_line_row < current_row < end_line_row:
            # 三行窗口
            window = self.composed_mask[current_row - 1:current_row + 2,
                                        detection_line_left_col:detection_line_right_col + 1]

            # 如果第三行有1值，说明相交，向下移动窗口
            if np.any(window[2] == 1):
                current_row += 1
            # 如果第二行有多个1值，说明相切，找到切点
            elif np.any(window[1] == 1):
                cut_points = np.where(window[1] == 1)[0]
                if direction == 1:
                    closest_cut_point = cut_points[0]  # 左侧竖线为外侧，选择第一个相切点
                else:
                    closest_cut_point = cut_points[0]  # 右侧竖线为外侧，选择最后一个相切点
                tangent_point = (detection_line_left_col + closest_cut_point, current_row)
                return tangent_point
            # 如果没有相交和相切，则向上移动窗口
            else:
                current_row -= 1

        return None  # 如果没有找到相切点

    @staticmethod
    def _find_center_point(bbox: List[int]) -> Tuple[int, int]:
        """

        Args:
            bbox: 检测框，格式为[x1,y1,x2,y2]

        Returns: 检测框中心点(center_x, center_y)

        """
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        return int(center_x), int(center_y)



