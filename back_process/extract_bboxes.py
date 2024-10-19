import os
import os.path as osp
import numpy as np
import json

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.apis import DetInferencer, init_detector, inference_detector
from typing import List
from typing import Any
from mmdet.structures import DetDataSample


class ImageDetector:
    def __init__(self, model: str, weight: str, threshold: float = 0.7):
        """
        初始化图像检测器
        Args:
            model: 分割模型名或模型配置路径
            weight: 模型的权重文件
            threshold: 检测框阈值
        """
        self.model = model
        self.weight = weight
        self.threshold = threshold
        self.result = None  # 结果文件的路径

    def __call__(self, image: str, out_dir: str) -> None:
        """
        对图像进行检测并保存结果
        Args:
            image: 待检测的图片
            out_dir: 输出文件夹
        """
        base_name = osp.basename(image)
        name_no_ext = osp.splitext(base_name)[0]
        result_json_path = osp.join(out_dir, f'preds/{name_no_ext}.json')  # inference会将结果文件保存到preds文件夹下，文件名与图片名相同

        if not osp.exists(result_json_path):
            inference = DetInferencer(model=self.model, weights=self.weight)
            inference(image, out_dir=out_dir, no_save_pred=False)
            print('det_result is created for the first time.')
        else:
            print(f'det_result comes from {result_json_path}.')
        self.result = result_json_path

    @property
    def bboxes(self) -> List[List[int]]:
        """
        提取检测结果中的边界框
        Returns: 满足阈值条件的边界框列表
        """
        if not self.result:
            raise ValueError("No detection result available. Please call the detector first.")

        with open(self.result, 'r') as file:
            data = json.load(file)
        bbox_scores = data['scores']
        bboxes_points = data['bboxes']
        filtered_bboxes = []

        for score, bbox in zip(bbox_scores, bboxes_points):
            if score > self.threshold:
                # 使用列表推导将所有元素转换为整数
                bbox = [int(item) for item in bbox]
                filtered_bboxes.append(bbox)
        return filtered_bboxes
