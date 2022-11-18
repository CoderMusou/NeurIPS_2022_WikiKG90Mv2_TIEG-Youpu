DATA_PATH=${1}
sh ./train_scripts/run_ote_lrd5k.sh ${DATA_PATH}
sh ./train_scripts/run_ote20_neg1200.sh ${DATA_PATH}