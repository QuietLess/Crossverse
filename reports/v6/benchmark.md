# CrossVerse offline benchmark — v6

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
| within_movie | popularity | 3000 | 0.0339 | 0.0094 | +258.7% | [+0.0193, +0.0295] | yes |
| within_game | popularity | 3000 | 0.0584 | 0.0195 | +199.9% | [+0.0318, +0.0461] | yes |
| mixed | popularity | 3000 | 0.0346 | 0.0091 | +278.9% | [+0.0207, +0.0304] | yes |
| cold_start | popularity | 3000 | 0.0045 | 0.0084 | -46.1% | [-0.0063, -0.0012] | yes |
| movie_to_game | popularity | 2518 | 0.0281 | 0.0293 | -4.2% | [-0.0050, +0.0026] | no |
| game_to_movie | popularity | 2518 | 0.0246 | 0.0203 | +21.4% | [+0.0013, +0.0074] | yes |
| within_movie | rrf_fusion | 3000 | 0.0339 | 0.0287 | +17.9% | [+0.0007, +0.0095] | yes |
| within_game | rrf_fusion | 3000 | 0.0584 | 0.0595 | -1.9% | [-0.0067, +0.0042] | no |
| mixed | rrf_fusion | 3000 | 0.0346 | 0.0311 | +11.1% | [-0.0008, +0.0074] | no |
| cold_start | rrf_fusion | 3000 | 0.0045 | 0.0033 | +37.8% | [+0.0002, +0.0023] | yes |
| movie_to_game | rrf_fusion | 2518 | 0.0281 | 0.0260 | +8.2% | [-0.0014, +0.0055] | no |
| game_to_movie | rrf_fusion | 2518 | 0.0246 | 0.0263 | -6.5% | [-0.0044, +0.0011] | no |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0094 | 0.0164 | 0.0273 | 0.0293 | 0.0004 | **0.6941** | 0.9998 |
| item_knn | **0.0354** | **0.0519** | 0.0734 | 0.0873 | 0.1149 | 0.6358 | 0.9418 |
| als_single_domain | 0.0260 | 0.0380 | 0.0569 | 0.0687 | 0.0279 | 0.6468 | 0.9901 |
| als_joint | 0.0258 | 0.0403 | 0.0606 | 0.0717 | 0.0227 | 0.6329 | 0.9914 |
| content | 0.0075 | 0.0125 | 0.0206 | 0.0250 | **0.2372** | 0.3611 | 0.6980 |
| two_tower | 0.0198 | 0.0308 | 0.0496 | 0.0577 | 0.0983 | 0.5544 | 0.9325 |
| rrf_fusion | 0.0287 | 0.0465 | 0.0722 | 0.0793 | 0.0401 | 0.6196 | 0.9864 |
| crossverse_ranker | 0.0337 | 0.0514 | **0.0749** | 0.0903 | 0.1373 | 0.6301 | 0.9240 |
| crossverse_ranker+diversity | 0.0339 | 0.0519 | 0.0717 | **0.0913** | 0.1316 | 0.6608 | 0.9313 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0338 | 0.0514 | 0.0510 | 0.0004 | 0.6065 | 0.9998 |
| item_knn | **0.0647** | 0.1022 | 0.1465 | 0.1410 | 0.0816 | 0.6244 | 0.9415 |
| als_single_domain | 0.0566 | 0.0936 | 0.1385 | 0.1277 | 0.0225 | 0.6382 | 0.9845 |
| als_joint | 0.0509 | 0.0839 | 0.1295 | 0.1153 | 0.0112 | 0.6264 | 0.9928 |
| content | 0.0161 | 0.0286 | 0.0453 | 0.0430 | **0.1493** | 0.3339 | 0.6592 |
| two_tower | 0.0459 | 0.0790 | 0.1196 | 0.1130 | 0.0613 | 0.5355 | 0.9423 |
| rrf_fusion | 0.0595 | 0.1038 | 0.1537 | 0.1430 | 0.0313 | 0.6027 | 0.9789 |
| crossverse_ranker | 0.0630 | **0.1053** | **0.1559** | **0.1443** | 0.0688 | 0.6196 | 0.9552 |
| crossverse_ranker+diversity | 0.0584 | 0.0936 | 0.1361 | 0.1287 | 0.0681 | **0.6762** | 0.9570 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0091 | 0.0145 | 0.0273 | 0.0290 | 0.0003 | **0.7348** | 0.9999 |
| item_knn | **0.0375** | **0.0543** | 0.0762 | 0.0947 | 0.1016 | 0.6501 | 0.9663 |
| als_joint | 0.0271 | 0.0401 | 0.0588 | 0.0757 | 0.0283 | 0.6706 | 0.9936 |
| content | 0.0075 | 0.0123 | 0.0216 | 0.0263 | **0.2514** | 0.3743 | 0.6918 |
| two_tower | 0.0220 | 0.0348 | 0.0584 | 0.0653 | 0.1037 | 0.5860 | 0.9565 |
| rrf_fusion | 0.0311 | 0.0497 | 0.0736 | 0.0890 | 0.0514 | 0.6483 | 0.9892 |
| crossverse_ranker | 0.0347 | 0.0542 | **0.0787** | **0.0963** | 0.1392 | 0.6575 | 0.9456 |
| crossverse_ranker+diversity | 0.0346 | 0.0539 | 0.0755 | 0.0960 | 0.1392 | 0.6601 | 0.9459 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0151** | **0.0245** | **0.0263** | 0.0003 | **0.7353** | 0.9999 |
| content | 0.0002 | 0.0006 | 0.0009 | 0.0010 | 0.0413 | 0.2453 | 0.4022 |
| rrf_fusion | 0.0033 | 0.0054 | 0.0087 | 0.0113 | **0.0771** | 0.5443 | 0.8077 |
| crossverse_ranker | 0.0046 | 0.0082 | 0.0119 | 0.0163 | 0.0697 | 0.5440 | 0.9022 |
| crossverse_ranker+diversity | 0.0045 | 0.0080 | 0.0115 | 0.0153 | 0.0704 | 0.5465 | 0.9013 |

### movie_to_game (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0293** | **0.0390** | 0.0563 | 0.1322 | 0.0002 | 0.6002 | 0.9998 |
| item_knn | 0.0141 | 0.0154 | 0.0198 | 0.0568 | **0.1215** | **0.7851** | 0.6771 |
| als_joint | 0.0201 | 0.0264 | 0.0408 | 0.0902 | 0.0097 | 0.7221 | 0.9949 |
| content | 0.0037 | 0.0056 | 0.0095 | 0.0191 | 0.0613 | 0.5063 | 0.5614 |
| copref | 0.0243 | 0.0315 | 0.0485 | 0.1084 | 0.0890 | 0.7296 | 0.9255 |
| two_tower | 0.0255 | 0.0347 | 0.0556 | 0.1140 | 0.0360 | 0.7158 | 0.9611 |
| rrf_fusion | 0.0260 | 0.0344 | 0.0490 | 0.1183 | 0.0155 | 0.7830 | 0.8906 |
| crossverse_ranker | 0.0286 | 0.0379 | **0.0645** | **0.1330** | 0.0172 | 0.6959 | 0.9936 |
| crossverse_ranker+diversity | 0.0281 | 0.0375 | 0.0612 | 0.1322 | 0.0169 | 0.7270 | 0.9933 |

### game_to_movie (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0203 | 0.0250 | 0.0377 | 0.1144 | 0.0002 | 0.6937 | 0.9998 |
| item_knn | 0.0184 | 0.0204 | 0.0261 | 0.0909 | **0.1376** | **0.7418** | 0.8392 |
| als_joint | 0.0161 | 0.0193 | 0.0290 | 0.0886 | 0.0128 | 0.7033 | 0.9953 |
| content | 0.0018 | 0.0027 | 0.0041 | 0.0131 | 0.1017 | 0.5010 | 0.5049 |
| copref | 0.0213 | 0.0237 | 0.0337 | 0.1068 | 0.1184 | 0.7148 | 0.9285 |
| two_tower | 0.0202 | 0.0259 | 0.0371 | 0.1041 | 0.0453 | 0.6857 | 0.9589 |
| rrf_fusion | **0.0263** | **0.0293** | 0.0410 | 0.1255 | 0.0185 | 0.7109 | 0.9663 |
| crossverse_ranker | 0.0248 | 0.0288 | **0.0423** | 0.1259 | 0.0207 | 0.6820 | 0.9946 |
| crossverse_ranker+diversity | 0.0246 | 0.0285 | 0.0415 | **0.1275** | 0.0208 | 0.6998 | 0.9948 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0045 (n=3000) | 0.0038 (n=996) | 0.0068 (n=889) | 0.0032 (n=1115) | 0.0049 (n=2041) |
| game_to_movie | 0.0246 (n=2518) | 0.0427 (n=334) | 0.0277 (n=299) | 0.0209 (n=1885) | 0.0246 (n=2518) |
| mixed | 0.0346 (n=3000) | 0.0295 (n=1113) | 0.0360 (n=889) | 0.0390 (n=998) | 0.0346 (n=3000) |
| movie_to_game | 0.0281 (n=2518) | 0.0378 (n=672) | 0.0265 (n=330) | 0.0241 (n=1516) | 0.0281 (n=2518) |
| within_game | 0.0584 (n=3000) | 0.0443 (n=1103) | 0.0572 (n=881) | 0.0746 (n=1016) | 0.0547 (n=2112) |
| within_movie | 0.0339 (n=3000) | 0.0332 (n=1207) | 0.0333 (n=895) | 0.0353 (n=898) | 0.0362 (n=2064) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 20,616 |
| item_knn_z | 19,397 |
| max_sim_pos | 14,188 |
| item_knn_rank | 14,087 |
| item_knn_score | 13,236 |
| log_rating_count | 12,575 |
| popularity_z | 12,529 |
| n_sources | 12,127 |
| popularity_score | 11,887 |
| bayes_rating | 11,379 |
| popularity_rank | 11,134 |
| als_z | 10,995 |
| two_tower_score | 10,285 |
| two_tower_rank | 9,474 |
| year_gap | 8,898 |