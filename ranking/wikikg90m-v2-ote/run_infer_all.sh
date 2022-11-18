DATA_PATH=${2}
INIT_PATH=${1}

python infer_all.py --model_path $INIT_PATH --data_path $DATA_PATH 
