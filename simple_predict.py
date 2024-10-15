from mmdet.apis import DetInferencer

config_path = 'configs/yolox/yolox_x_8xb8-300e_wdd.py'

weight = 'work_dirs/yolox_x_8xb8-300e_wdd/epoch_280.pth'

out_dir = './wdd'

inference = DetInferencer(model=config_path, weights=weight)

inference('raw_images/car.jpg', show=True)
