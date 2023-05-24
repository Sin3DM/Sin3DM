GPU_ID=0
PROJ_DIR=$(pwd)
BLENDER_PATH=blender

# data
cd $PROJ_DIR/data
OBJ_PATH=$PROJ_DIR/data/towerruins/mesh/model.obj
DATA_PATH=$PROJ_DIR/data/towerruins/towerruins.npz
python mesh_sampler.py -s $OBJ_PATH -d $DATA_PATH --n_surf 5000000 --watertight
echo $DATA_PATH

cd $PROJ_DIR/rendering # render images for evaluation
$BLENDER_PATH -b -P blender_render_multiview.py -- -s $OBJ_PATH -o $(dirname $DATA_PATH)/renderings -g $GPU_ID


# # train
cd $PROJ_DIR/src
DATA_TAG=$(basename $DATA_PATH .npz)
EXP_DIR=checkpoints/$DATA_TAG
echo $EXP_DIR

python train.py \
    --tag $EXP_DIR \
    --data_path $DATA_PATH \
    --predict_xstart True \
    --enc_net_type skip \
    --enc_lr_decay 0.1 \
    --enc_lr_split 0.2 \
    --gpu_id $GPU_ID


# sample
cd $PROJ_DIR/src
python sample.py \
    --tag $EXP_DIR \
    --n_samples 50 \
    --n_faces 50000 \
    --output results50 \
    --gpu_id $GPU_ID

RESULT_DIR=$PROJ_DIR/src/$EXP_DIR/results50


# rendering
cd $PROJ_DIR/rendering
python mvrender_script.py -s $RESULT_DIR -g $GPU_ID -bl $BLENDER_PATH


# evaluation
cd $PROJ_DIR/evaluation
python eval_full.py -s $RESULT_DIR -r $(dirname $DATA_PATH) -g $GPU_ID
