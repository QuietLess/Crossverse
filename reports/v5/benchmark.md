# CrossVerse offline benchmark — v5

Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.
`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.

## Split

```json
{
  "train_interactions": 1158594,
  "train_users": 172604,
  "bridge_users": 120956,
  "holdout_movie_to_game_test": 2518,
  "holdout_game_to_movie_test": 2518,
  "holdout_movie_to_game_val": 2518,
  "holdout_game_to_movie_val": 2518,
  "val_within_movie": 3000,
  "val_within_game": 3000,
  "val_mixed": 3000,
  "val_cold_start": 3000,
  "val_movie_to_game": 2518,
  "val_game_to_movie": 2518,
  "test_within_movie": 3000,
  "test_within_game": 3000,
  "test_mixed": 3000,
  "test_cold_start": 3000,
  "test_movie_to_game": 2518,
  "test_game_to_movie": 2518
}
```

## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)

| task | vs baseline | n | ndcg@10 model | ndcg@10 baseline | lift | 95% CI (abs) | significant |
|---|---|---:|---:|---:|---:|---|:---:|
| within_movie | popularity | 3000 | 0.0387 | 0.0094 | +310.1% | [+0.0239, +0.0347] | yes |
| within_game | popularity | 3000 | 0.0627 | 0.0195 | +222.1% | [+0.0360, +0.0506] | yes |
| mixed | popularity | 3000 | 0.0403 | 0.0091 | +342.1% | [+0.0257, +0.0369] | yes |
| cold_start | popularity | 3000 | 0.0044 | 0.0084 | -47.5% | [-0.0064, -0.0014] | yes |
| movie_to_game | popularity | 2518 | 0.0324 | 0.0293 | +10.6% | [-0.0009, +0.0071] | no |
| game_to_movie | popularity | 2518 | 0.0251 | 0.0203 | +23.4% | [+0.0016, +0.0079] | yes |
| within_movie | rrf_fusion | 3000 | 0.0387 | 0.0305 | +27.0% | [+0.0039, +0.0130] | yes |
| within_game | rrf_fusion | 3000 | 0.0627 | 0.0617 | +1.6% | [-0.0043, +0.0060] | no |
| mixed | rrf_fusion | 3000 | 0.0403 | 0.0352 | +14.5% | [+0.0013, +0.0090] | yes |
| cold_start | rrf_fusion | 3000 | 0.0044 | 0.0033 | +34.3% | [+0.0002, +0.0021] | yes |
| movie_to_game | rrf_fusion | 2518 | 0.0324 | 0.0180 | +80.2% | [+0.0106, +0.0181] | yes |
| game_to_movie | rrf_fusion | 2518 | 0.0251 | 0.0205 | +22.4% | [+0.0019, +0.0074] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0094 | 0.0164 | 0.0273 | 0.0293 | 0.0004 | 0.6941 | 0.9998 |
| item_knn | 0.0345 | 0.0474 | 0.0626 | 0.0803 | **0.2642** | **0.7088** | 0.6455 |
| als_single_domain | 0.0260 | 0.0380 | 0.0569 | 0.0687 | 0.0279 | 0.6468 | 0.9901 |
| als_joint | 0.0258 | 0.0403 | 0.0606 | 0.0717 | 0.0227 | 0.6329 | 0.9914 |
| content | 0.0075 | 0.0125 | 0.0206 | 0.0250 | 0.2372 | 0.3611 | 0.6980 |
| two_tower | 0.0198 | 0.0308 | 0.0496 | 0.0577 | 0.0983 | 0.5544 | 0.9325 |
| rrf_fusion | 0.0305 | 0.0494 | 0.0747 | 0.0843 | 0.0480 | 0.6210 | 0.9822 |
| crossverse_ranker | 0.0376 | 0.0552 | 0.0825 | 0.0963 | 0.1442 | 0.6259 | 0.9167 |
| crossverse_ranker+diversity | **0.0387** | **0.0576** | **0.0835** | **0.0990** | 0.1383 | 0.6610 | 0.9247 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0338 | 0.0514 | 0.0510 | 0.0004 | 0.6065 | 0.9998 |
| item_knn | 0.0572 | 0.0876 | 0.1229 | 0.1197 | **0.1497** | 0.6522 | 0.7465 |
| als_single_domain | 0.0566 | 0.0936 | 0.1385 | 0.1277 | 0.0225 | 0.6382 | 0.9845 |
| als_joint | 0.0509 | 0.0839 | 0.1295 | 0.1153 | 0.0112 | 0.6264 | 0.9928 |
| content | 0.0161 | 0.0286 | 0.0453 | 0.0430 | 0.1493 | 0.3339 | 0.6592 |
| two_tower | 0.0459 | 0.0790 | 0.1196 | 0.1130 | 0.0613 | 0.5355 | 0.9423 |
| rrf_fusion | 0.0617 | 0.1075 | 0.1565 | 0.1483 | 0.0365 | 0.6054 | 0.9740 |
| crossverse_ranker | **0.0665** | **0.1101** | **0.1585** | **0.1510** | 0.0678 | 0.6059 | 0.9558 |
| crossverse_ranker+diversity | 0.0627 | 0.1005 | 0.1500 | 0.1397 | 0.0673 | **0.6631** | 0.9569 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0091 | 0.0145 | 0.0273 | 0.0290 | 0.0003 | **0.7348** | 0.9999 |
| item_knn | 0.0355 | 0.0502 | 0.0652 | 0.0850 | **0.2883** | 0.7150 | 0.6917 |
| als_joint | 0.0271 | 0.0401 | 0.0588 | 0.0757 | 0.0283 | 0.6706 | 0.9936 |
| content | 0.0075 | 0.0123 | 0.0216 | 0.0263 | 0.2514 | 0.3743 | 0.6918 |
| two_tower | 0.0220 | 0.0348 | 0.0584 | 0.0653 | 0.1037 | 0.5860 | 0.9565 |
| rrf_fusion | 0.0352 | 0.0569 | 0.0836 | 0.0993 | 0.0668 | 0.6278 | 0.9827 |
| crossverse_ranker | 0.0401 | 0.0606 | **0.0864** | 0.1083 | 0.1546 | 0.6501 | 0.9318 |
| crossverse_ranker+diversity | **0.0403** | **0.0611** | 0.0855 | **0.1093** | 0.1545 | 0.6525 | 0.9320 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0151** | **0.0245** | **0.0263** | 0.0003 | **0.7353** | 0.9999 |
| content | 0.0002 | 0.0006 | 0.0009 | 0.0010 | 0.0413 | 0.2453 | 0.4022 |
| rrf_fusion | 0.0033 | 0.0054 | 0.0087 | 0.0113 | **0.0771** | 0.5443 | 0.8077 |
| crossverse_ranker | 0.0044 | 0.0072 | 0.0124 | 0.0150 | 0.0699 | 0.5497 | 0.9076 |
| crossverse_ranker+diversity | 0.0044 | 0.0072 | 0.0120 | 0.0147 | 0.0708 | 0.5517 | 0.9067 |

### movie_to_game (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0293 | 0.0390 | 0.0563 | 0.1322 | 0.0002 | 0.6002 | 0.9998 |
| item_knn | 0.0037 | 0.0040 | 0.0066 | 0.0131 | 0.1433 | 0.7980 | 0.3780 |
| als_joint | 0.0201 | 0.0264 | 0.0408 | 0.0902 | 0.0097 | 0.7221 | 0.9949 |
| content | 0.0037 | 0.0056 | 0.0095 | 0.0191 | 0.0613 | 0.5063 | 0.5614 |
| copref | 0.0036 | 0.0039 | 0.0055 | 0.0119 | **0.1616** | 0.7782 | 0.3786 |
| two_tower | 0.0255 | 0.0347 | 0.0556 | 0.1140 | 0.0360 | 0.7158 | 0.9611 |
| rrf_fusion | 0.0180 | 0.0232 | 0.0373 | 0.0814 | 0.0169 | **0.8094** | 0.7771 |
| crossverse_ranker | **0.0336** | **0.0442** | **0.0684** | **0.1509** | 0.0197 | 0.6913 | 0.9925 |
| crossverse_ranker+diversity | 0.0324 | 0.0430 | 0.0640 | 0.1473 | 0.0193 | 0.7275 | 0.9922 |

### game_to_movie (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0203 | 0.0250 | 0.0377 | 0.1144 | 0.0002 | 0.6937 | 0.9998 |
| item_knn | 0.0015 | 0.0016 | 0.0027 | 0.0111 | 0.2286 | 0.8400 | 0.2820 |
| als_joint | 0.0161 | 0.0193 | 0.0290 | 0.0886 | 0.0128 | 0.7033 | 0.9953 |
| content | 0.0018 | 0.0027 | 0.0041 | 0.0131 | 0.1017 | 0.5010 | 0.5049 |
| copref | 0.0008 | 0.0008 | 0.0010 | 0.0064 | **0.2452** | **0.8435** | 0.3056 |
| two_tower | 0.0202 | 0.0259 | 0.0371 | 0.1041 | 0.0453 | 0.6857 | 0.9589 |
| rrf_fusion | 0.0205 | 0.0224 | 0.0315 | 0.1044 | 0.0232 | 0.7506 | 0.8946 |
| crossverse_ranker | **0.0252** | 0.0296 | **0.0439** | 0.1251 | 0.0247 | 0.6817 | 0.9926 |
| crossverse_ranker+diversity | 0.0251 | **0.0301** | 0.0420 | **0.1255** | 0.0243 | 0.7059 | 0.9928 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0044 (n=3000) | 0.0037 (n=996) | 0.0066 (n=889) | 0.0033 (n=1115) | 0.0045 (n=2041) |
| game_to_movie | 0.0251 (n=2518) | 0.0331 (n=334) | 0.0302 (n=299) | 0.0228 (n=1885) | 0.0251 (n=2518) |
| mixed | 0.0403 (n=3000) | 0.0358 (n=1113) | 0.0387 (n=889) | 0.0468 (n=998) | 0.0403 (n=3000) |
| movie_to_game | 0.0324 (n=2518) | 0.0406 (n=672) | 0.0426 (n=330) | 0.0266 (n=1516) | 0.0324 (n=2518) |
| within_game | 0.0627 (n=3000) | 0.0490 (n=1103) | 0.0606 (n=881) | 0.0793 (n=1016) | 0.0584 (n=2112) |
| within_movie | 0.0387 (n=3000) | 0.0343 (n=1207) | 0.0382 (n=895) | 0.0452 (n=898) | 0.0382 (n=2064) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 18,351 |
| n_sources | 14,998 |
| item_knn_z | 12,801 |
| log_rating_count | 11,834 |
| item_knn_rank | 10,772 |
| popularity_rank | 8,022 |
| item_knn_score | 7,657 |
| als_z | 7,298 |
| two_tower_score | 6,642 |
| popularity_score | 6,598 |
| two_tower_rank | 5,855 |
| popularity_z | 5,607 |
| max_sim_pos | 5,125 |
| bayes_rating | 4,972 |
| year_gap | 4,582 |