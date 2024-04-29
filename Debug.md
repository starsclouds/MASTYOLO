## 1 训练中
### Resume model 恢复训练
```shell
 CUDA_VISIBLE_DEVICES=0,1 PORT=12345 bash ./tools/dist_train_mat.sh configs/swin/yolov3_swin_mstrain-608_3x_coco.py 2 --resume-from /root/autodl-tmp/work_dirs/latest.pth
```
