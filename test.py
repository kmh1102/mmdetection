from mmengine.config import Config

cfg = Config.fromfile('configs/yolox/yolox_s_8xb8-300e_coco.py')


config = f'yolox_x_8xb8-300e_coco.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)

