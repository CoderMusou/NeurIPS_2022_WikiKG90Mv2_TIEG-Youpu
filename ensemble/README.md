# Usage

## Ensemble for recall

### 1. Get the recall candidate for all models
See [candidate_data](../candidate_data/README.md).

### 2. Recall model ensemble
Run script:
```shell
python recall_model_ensemble.py
```

## Ensemble for ranking
1. Modify the variable `cur_model_path` in `rank_model_ensemble.py`.
2. Run script `python rank_model_ensemble.py valid_candidate_path valid_correct_t_path test_candidate_path output_path`

## Get the result of submission
Run script:
```shell
python get_test_chl_submission.py t_pred_top10_file
```

