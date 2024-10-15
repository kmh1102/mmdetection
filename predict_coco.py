from mmdet.apis import DetInferencer

checkpoints = 'checkpoints/solov2/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'
inference = DetInferencer(model='solov2_r50_fpn_1x_coco', weights=checkpoints)

inference('./raw_images/car1.jpg', out_dir='./')