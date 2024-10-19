import os
import os.path as osp
import numpy as np
import json

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.apis import DetInferencer, init_detector, inference_detector
from typing import Any
from mmdet.structures import DetDataSample


def image_detection(det_image: str, out_dir: str, model: str, weight: str) -> str:
    """

    Args:
        det_image: 待检测的图片
        out_dir: 输出文件夹
        model: 分割模型名或模型配置路径
        weight: 模型的权重文件

    Returns: 保存结果的json文件路径

    """
    det_inference = DetInferencer(model=model, weights=weight)
    det_inference(det_image, out_dir=out_dir, no_save_pred=False)

    base_name = osp.basename(det_image)
    name_no_ext = osp.splitext(base_name)[0]

    result_json = osp.join(out_dir, f'preds/{name_no_ext}.json')

    return result_json


def extract_bboxes(det_result: str, score_threshold: float = 0.7) -> Any:
    """
    从检测结果json文件中提取检测框
    Args:
        det_result: 检测结果的json文件路径
        score_threshold: 检测框阈值

    Returns: 检测框列表。类型检查无法识别到tolist()的返回类型，为了避免注解警告，将返回类型标注为Any

    """
    with open(det_result, 'r') as file:
        data = json.load(file)
    bbox_scores = data['scores']
    bboxes_points = data['bboxes']
    filtered_bboxes = []

    for score, bbox in zip(bbox_scores, bboxes_points):
        if score > score_threshold:
            filtered_bboxes.append(bbox)
    array_bboxes = np.array(filtered_bboxes, dtype=np.int32)

    final_bboxes = array_bboxes.tolist()
    return final_bboxes


if __name__ == '__main__':
    det_result_ = '../outputs/wdd/preds/car.json'
    print(type(extract_bboxes(det_result_)))
