GPU=0
DATA=${1}
MODEL_PATH=${2}
INFER_OUTPUT=${3}
CUDA_VISIBLE_DEVICES=${GPU} python -u ./src/main.py --cuda \
  --context_hops 3 --batch_size 512 --lr 0.0002 --dim 1024 --l2 0.0  --steps 10000000 \
  --cpu_num 128  --dataset WikiKG90Mv2 --neighbor_samples 10 \
  --neighbor_agg mean \
  --data_path ${DATA} \
  --use_ranking_loss True --margin 3 --gamma 3 \
  --infer \
  --infer_checkpoint ${MODEL_PATH} \
  --test_batch_size 1536 \
  --infer_path ${INFER_OUTPUT}
