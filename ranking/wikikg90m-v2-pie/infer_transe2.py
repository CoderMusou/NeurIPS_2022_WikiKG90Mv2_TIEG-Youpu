import os
import sys

os.system("cd ./dgl-ke-ogb-lsc/python;python3 -m pip install -e .")
os.system("pip list")
os.system("pip install dgl==0.4.* -U")
os.system("pip install ogb")

DATA_PATH = sys.argv[1]
SAVE_PATH = sys.argv[2]
VAL_CANDIDATE_PATH = sys.argv[3]
TEST_CANDIDATE_PATH = sys.argv[4]
CHECKPOINT = sys.argv[5]
NUM_PROC = 4

run_shell = "CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 dglke_train --model_name TransE \
  --hidden_dim 600 --gamma 10  --valid --test_dev --test_challenge -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
  --gpu 0 1 2 3 \
  --async_update --no_save_emb \
  --print_on_screen --encoder_model_name shallow --save_path {} \
  --data_path {} \
  --valid_dict {} \
  --test_challenge_dict {} \
  --only_valid --checkpoint {} \
  --neg_sample_size 8196 --batch_size 8196 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 20000000 --force_sync_interval 1000 --eval_interval 100000 \
  --LRE --LRE_rank 200".format(SAVE_PATH, DATA_PATH, VAL_CANDIDATE_PATH, TEST_CANDIDATE_PATH, CHECKPOINT)

print("="*5 + "run_shell" + "="*5)
print(run_shell)
os.system(run_shell)
