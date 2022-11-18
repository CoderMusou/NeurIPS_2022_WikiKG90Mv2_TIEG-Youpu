DATA_PATH=${1}
SAVA_FILE=${2}
E2R_SCORES_FILE=${3}

python -u get_candidate_dgl.py \
   --dataset 'wikikg90M' \
   --data_path ${DATA_PATH} \
   --test_challenge \
   --batch_size_eval 1 \
   --num_hops 3 \
   --num_proc 15 \
   --save_file ${SAVA_FILE} \
   --e2r_score_file ${E2R_SCORES_FILE}
