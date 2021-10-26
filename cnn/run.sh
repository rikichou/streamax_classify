#/usr/local/bin
python train_example.py \
  --batch_size 32 \
  --lr 0.006 \
  --beta 1.0 \
  --cutmix_prob 0.0 \
  --length 16 \
  --print_freq 100 \
  --opt_level 02 \
  --worker 16 \
  --epochs 500 \
  --net_type VGGNet \
  --dataset PlayPhone \
  --init_checkpoint /home/zyluo/RFBNet/weights/SSD_VGG_FPN_RFB_2020-08-31-19/SSD_VGG_FPN_RFB_VOC_epoches_55.pth \
