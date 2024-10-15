from mmengine.config import Config

cfg = Config.fromfile('configs/yolox/yolox_s_8xb8-300e_coco.py')

# 数据集配置
cfg.metainfo = {
    'classes': ('wheel',),
    'palette': [
        (220, 20, 60),
    ]
}

# 数据集根目录
cfg.data_root = '/data/home/user12/dl-projects/datasets/WDD'

# 训练集配置
cfg.train_dataset.dataset.ann_file = 'annotations/train.json'
cfg.train_dataset.dataset.data_prefix.img = 'train/'
cfg.train_dataset.dataset.data_root = cfg.data_root

cfg.train_dataloader.dataset.dataset.ann_file = 'annotations/train.json'
cfg.train_dataloader.dataset.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.dataset.data_root = cfg.data_root

# 验证集配置
cfg.val_dataloader.dataset.ann_file = 'annotations/val.json'
cfg.val_dataloader.dataset.data_prefix.img = 'val/'
cfg.val_dataloader.dataset.data_root = cfg.data_root

cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'annotations/val.json'

# 测试集配置
cfg.test_dataloader.dataset.ann_file = 'annotations/test.json'
cfg.test_dataloader.dataset.data_prefix.img = 'test/'
cfg.test_dataloader.dataset.data_root = cfg.data_root

cfg.test_evaluator.ann_file = cfg.data_root + '/' + 'annotations/test.json'
# 模型配置
cfg.model.bbox_head.num_classes = 1

cfg.load_from = 'checkpoints/yolox/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'

# 日志配置
cfg.vis_backends.append({"type": 'TensorboardVisBackend'})
cfg.visualizer.vis_backends.append({"type": 'TensorboardVisBackend'})

# 保存配置
config = f'configs/yolox/yolox_x_8xb8-300e_wdd.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)
