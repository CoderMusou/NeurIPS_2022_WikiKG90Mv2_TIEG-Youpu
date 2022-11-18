# TIEG-Youpu's Solution for NeurIPS 2022 WikiKG90Mv2-LSC
This is the code of Team TIEG-Youpu in the WikiKG90Mv2-LSC track of OGB-LSC @ NeurIPS 2022.

Team Members: Feng Nie,  Zhixiu Ye, Sifa Xie, Shuang Wu, Xin Yuan, Liang Yao, Jiazhen Peng, Xu Cheng.

## Installation requirements
```
ogb >= 1.3.3
torch >= 1.7.0
dgl == 0.4.3
```

## Recall stage
In the recall stage, we used the following methods.

### 1. PIE recall
#### 1) Model training
Run script:
```shell
cd ./recall/entity_typing/; sh run_{0, 1, 2}.sh DATA
```

#### 2) Model inference
Run script:
```shell
cd ./recall/entity_typing/; sh run_infer.sh DATA MODEL_PATH INFER_OUTPUT
```

#### 3) Get valid candidate

Run script:
```shell
cd ./recall/candidate/; sh get_valid_candidate.sh DATA_PATH SAVA_FILE E2R_SCORES_FILE
```

#### 4) Get test challenge candidate
Run script:
```shell
cd ./recall/candidate/; sh get_test_candidate.sh DATA_PATH SAVA_FILE E2R_SCORES_FILE
```

### 2. Rule recall
It can be downloaded from this [google drive](https://drive.google.com/drive/folders/1tSVuP-FKHZcgUGvbzLv7hwPW4kwq5yw_).

### 3. Faiss retrieval recall
We retrieve potential tail entities in [faiss](https://github.com/facebookresearch/faiss) using text embeddings, run script:
```shell
cd ./recall/faiss_retrieval/; sh run.sh
```

### 4. Ensemble for recall
Run script:
```shell
cd ensemble; python recall_model_ensemble.py
```

## Ranking stage
We used 6 models in the ranking stage.

### 1. TransE1
Run the training script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./train_transe1.py DATA_PATH SAVE_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./infer_transe1.py DATA_PATH SAVE_PATH VAL_CANDIDATE_PATH TEST_CANDIDATE_PATH CHECKPOINT
```

### 2. TransE2
Run the training script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./train_transe2.py DATA_PATH SAVE_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./infer_transe2.py DATA_PATH SAVE_PATH VAL_CANDIDATE_PATH TEST_CANDIDATE_PATH CHECKPOINT
```

### 3. ComplEx
Run the training script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./train_complex.py DATA_PATH SAVE_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./infer_complex.py DATA_PATH SAVE_PATH VAL_CANDIDATE_PATH TEST_CANDIDATE_PATH CHECKPOINT
```


### 4. TransE with text embedding
Run the training script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./train_transe_concatv1.py DATA_PATH SAVE_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-pie; python -u ./infer_transe_concatv1.py DATA_PATH SAVE_PATH VAL_CANDIDATE_PATH TEST_CANDIDATE_PATH CHECKPOINT
```


### 5. OTE1 
Run the training script:
```shell
cd ranking/wikikg90m-v2-ote; sh ./train_scripts/run_ote_lrd5k.sh DATA_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-ote; python infer_all.py --model_path INIT_PATH --data_path DATA_PATH
```


### 6. OTE2
Run the training script:
```shell
cd ranking/wikikg90m-v2-ote; sh ./train_scripts/run_ote20_neg1200.sh DATA_PATH
```
Run the inference script:
```shell
cd ranking/wikikg90m-v2-ote; python infer_all.py --model_path INIT_PATH --data_path DATA_PATH
```

### 5. Ensemble for ranking
1. Modify the variable `cur_model_path` in `rank_model_ensemble.py`.
2. Run script `python rank_model_ensemble.py valid_candidate_path valid_correct_t_path test_candidate_path output_path`

### 6. Get the result of submission
Run script:
```shell
cd ensemble; python get_test_chl_submission.py t_pred_top10_file_path
```
