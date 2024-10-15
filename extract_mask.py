import os
import cv2
from mmdet.apis import DetInferencer, init_detector, inference_detector


def get_image_paths(folder_path):
    # 常见的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_paths = []

    # 遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 获取文件的扩展名并转换为小写
            ext = os.path.splitext(file)[1].lower()
            # 如果文件扩展名是图片格式，则保存路径
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths


def save_binary_masks(result, score_threshold=0.7, output_dir='./output'):
    # 从result中提取pred_instances
    pred_instances = result.pred_instances

    # 提取masks和scores
    masks = pred_instances.masks
    scores = pred_instances.scores

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有masks和scores
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 仅保留score大于阈值的掩码
        if score > score_threshold:
            # 将mask转换为二值图像
            binary_mask = mask.byte().cpu().numpy() * 255

            # 保存二值图像
            output_path = os.path.join(output_dir, f'mask_{i}.png')
            cv2.imwrite(output_path, binary_mask)


if __name__ == '__main__':
    # 原始图片路径
    input_dir = './raw_images'

    # 输出结果路径
    output_dir = './outputs'

    # 获取所有图片路径
    img_paths = get_image_paths(input_dir)

    # 模型配置路径
    config_path = './configs/solov2/solov2_r50_fpn_1x_apd.py'

    # checkpoint文件路径
    checkpoint_path = './work_dirs/solov2_r50_fpn_1x_apd/epoch_160.pth'

    model = init_detector(config_path, checkpoint=checkpoint_path)

    img_path = 'raw_images/00513.jpg'

    result = inference_detector(model, img_path)

    save_binary_masks(result, output_dir=input_dir)
    # inference = DetInferencer(model=config_path, weights='./work_dirs/solov2_r50_fpn_1x_apd/epoch_160.pth')
    #
    # inference(img_paths, out_dir=output_dir, no_save_pred=False)
