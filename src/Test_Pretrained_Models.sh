CONFIG_PATH=/home/cgusti/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py
NUM_GPUS=1
PRETRAINED_PATH=/home/cgusti/vitpose_small.pth

echo "CONFIG_PATH == ${CONFIG_PATH}"

echo "Executing function"
bash ViTPose/tools/dist_test.sh $CONFIG_PATH $PRETRAINED_PATH $NUM_GPUS
echo "Executed"
