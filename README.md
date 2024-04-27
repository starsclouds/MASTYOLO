# MATYOLO
A Repository for low light object detection.

## Training
Setp-1:
Pre-train MAST-YOLO model (36 epochs on 2 GPUs): 
```shell
CUDA_VISIBLE_DEVICES=0,1 PORT=12345 bash ./tools/dist_train_mat.sh configs/swin/yolov3_swin_mstrain-608_3x_coco.py 2
```
