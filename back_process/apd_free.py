import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
import os.path as osp
from extract_mask import ImageSegmenter
from extract_bboxes import ImageDetector
from extract_contacts import TangentSolver


class FreeParkingDetector:
    model_config = {
        # 分割模型配置
        'seg_weight': '../checkpoints/solov2/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
        'seg_model': 'solov2_r50_fpn_1x_coco',

        # 检测模型配置
        'det_weight': '../work_dirs/yolox_l_8xb8-300e_wdd/epoch_300.pth',
        'det_model':  '../configs/yolox/yolox_l_8xb8-300e_wdd.py'
    }

    # 支持的图片文件扩展名
    img_extensions = {'.jpg', '.jpeg', '.png'}

    def __init__(self, origin_image_path: str, master_out_dir: str) -> None:

        self.master_out_dir = master_out_dir
        self.origin_image_path = origin_image_path
        # 汽车分割与车轮检测初始化
        self.car_segmenter = ImageSegmenter(self.model_config['seg_model'], self.model_config['seg_weight'])
        self.wheel_detector = ImageDetector(self.model_config['det_model'], self.model_config['det_weight'])

    def __call__(self, *args, **kwargs) -> None:

        if osp.isdir(self.origin_image_path):
            for image_path in self._get_image_path():
                self._process(image_path)
        else:
            self._process(self.origin_image_path)

    def _process(self, image_path: str) -> None:
        image_out_dir, car_seg_dir, wheel_det_dir = self._assign_out_dir(image_path)
        self.car_segmenter(image_path, car_seg_dir)
        self.wheel_detector(image_path, wheel_det_dir)

        car_masks, car_composed_mask = self.car_segmenter.masks
        wheel_bboxes = self.wheel_detector.bboxes

        contacts_solver = TangentSolver(car_masks, car_composed_mask, wheel_bboxes)
        wheel_tangent_points = contacts_solver()

        # 输出带有接地点标记的图片
        image = cv2.imread(image_path)
        for wheel_tangent_point in wheel_tangent_points:
            cv2.circle(image, wheel_tangent_point, radius=5, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(osp.join(image_out_dir, 'car_tangent_points.jpg'), image)

    def _get_image_path(self) -> str:
        for root, _, files in os.walk(self.origin_image_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.img_extensions:
                    yield os.path.join(root, file)

    def _assign_out_dir(self, origin_image_path: str) -> Tuple[str, str, str]:
        """
        根据主输出目录和图像路径创建分割输出目录和检测输出目录。

        :param origin_image_path: 输入图像的路径
        :return: (car_seg_dir, wheel_det_dir) 两个字符串，分别为车辆分割输出目录和轮胎检测输出目录
        """

        # 获取图像的基础名称和去掉扩展名的名称
        image_base_name = osp.basename(origin_image_path)
        ext_name = osp.splitext(image_base_name)[1].lower()

        # 检查是否为图片
        if ext_name not in self.img_extensions:
            raise ValueError("File is not a picture.")

        name_no_ext = osp.splitext(image_base_name)[0]

        image_out_dir = osp.join(self.master_out_dir, name_no_ext)
        if not osp.exists(image_out_dir):
            os.makedirs(image_out_dir)
            print(f"创建图片处理输出目录: {image_out_dir}")

        # 车辆分割输出目录和轮胎检测输出目录
        car_seg_dir = osp.join(image_out_dir, "car_seg", )
        wheel_det_dir = osp.join(image_out_dir, "wheel_det")

        # 创建车辆分割输出目录
        if not osp.exists(car_seg_dir):
            os.makedirs(car_seg_dir)
            print(f"创建车辆分割输出目录: {car_seg_dir}")

        # 创建轮胎检测输出目录
        if not osp.exists(wheel_det_dir):
            os.makedirs(wheel_det_dir)
            print(f"创建轮胎检测输出目录: {wheel_det_dir}")

        return image_out_dir, car_seg_dir, wheel_det_dir


if __name__ == '__main__':
    img_path = '../raw_images/side_car.jpg'
    apd_free_out_dir = '../apd_out/free'
    parking_detector = FreeParkingDetector(img_path, apd_free_out_dir)
    parking_detector()
