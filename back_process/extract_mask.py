import os
import os.path as osp
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.apis import DetInferencer, init_detector, inference_detector
from mmdet.structures import DetDataSample


def image_segment(seg_image: str, out_dir: str, model: str, weight: str) -> dict:
    """
    Args:
        seg_image: 待分割的图片路径
        out_dir: 保存输出结果的目录路径
        model: 分割模型
        weight: 权重文件路径

    Returns: 推理结果

    """
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    seg_result_save_path = osp.join(out_dir, 'seg_result.pickle')
    if not osp.exists(seg_result_save_path):
        # 推理
        seg_inference = DetInferencer(model=model, weights=weight)
        seg_result = seg_inference(seg_image, out_dir=out_dir, no_save_pred=False, return_datasamples=True)

        # 存储推理结果
        with open(seg_result_save_path, 'wb') as file:
            pickle.dump(seg_result, file)
        print('seg_result is created for the first time.')
    else:
        # 加载推理结果
        with open(seg_result_save_path, 'rb') as file:
            seg_result = pickle.load(file)
        print(f'seg_result comes from {seg_result_save_path}.')
    return seg_result


def extract_masks(seg_result: dict, score_threshold: float = 0.7) -> np.ndarray:
    """

    Args:
        seg_result: 模型分割结果
        score_threshold: 掩码得分阈值

    Returns: 一组大于阈值的掩码

    """
    # 从字典中提取predictions的值
    predictions = seg_result['predictions']

    # 目前只取列表中的第一个元素，其为DetDataSample类型
    prediction = predictions[0]

    # 为InstanceData类型
    pred_instances = prediction.pred_instances

    # 提取预测掩码与对应得分
    masks = pred_instances.masks
    scores = pred_instances.scores

    # 获得阈值高于0.7的掩码索引
    mask_indices = [index for index, score in enumerate(scores) if score > score_threshold]

    # 将掩码转换为numpy类型，同时将布尔值转换为数字，此时masks中的值全为0/1
    masks = masks.cpu().numpy()[mask_indices, :, :] * 1

    return masks


if __name__ == '__main__':
    pass
