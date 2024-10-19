import os
import os.path as osp
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.apis import DetInferencer, init_detector, inference_detector
from mmdet.structures import DetDataSample


class ImageSegmenter:
    def __init__(self, model: str, weight: str, threshold: float = 0.7):
        """
        初始化分割器

        Args:
            model: 分割模型
            weight: 权重文件路径
            threshold: 掩码得分阈值
        """
        self.model = model
        self.weight = weight
        self.threshold = threshold
        self.result = None

    def __call__(self, image: str, out_dir: str) -> None:
        """
        对图像进行分割

        Args:
            image: 待分割的图片路径
            out_dir: 保存输出结果的目录路径

        Returns: 推理结果
        """
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        result_save_path = osp.join(out_dir, 'result.pickle')
        if not osp.exists(result_save_path):
            # 推理
            inference = DetInferencer(model=self.model, weights=self.weight)
            result = inference(image, out_dir=out_dir, no_save_pred=False, return_datasamples=True)

            # 存储推理结果
            with open(result_save_path, 'wb') as file:
                pickle.dump(result, file)
            print('result is created for the first time.')
        else:
            # 加载推理结果
            with open(result_save_path, 'rb') as file:
                result = pickle.load(file)
            print(f'result comes from {result_save_path}.')

        self.result = result

    @property
    def masks(self) -> np.ndarray:
        """
        提取高于阈值的掩码
        Returns:一组大于阈值的掩码

        """
        if self.result is None:
            raise ValueError("No segmentation result available. Please call the segmenter first.")

        # 从字典中提取predictions的值
        predictions = self.result['predictions']

        # 目前只取列表中的第一个元素，其为DetDataSample类型
        prediction = predictions[0]

        # 为InstanceData类型
        pred_instances = prediction.pred_instances

        # 提取预测掩码与对应得分
        masks = pred_instances.masks
        scores = pred_instances.scores

        # 获得阈值高于0.7的掩码索引
        mask_indices = [index for index, score in enumerate(scores) if score > self.threshold]

        # 将掩码转换为numpy类型，同时将布尔值转换为数字，此时masks中的值全为0/1
        masks = masks.cpu().numpy()[mask_indices, :, :] * 1

        return masks


if __name__ == '__main__':
    pass
