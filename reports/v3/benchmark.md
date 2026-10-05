# CrossVerse offline benchmark — v3

Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.
`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.

## Split

```json
{
  "train_interactions": 501964,
  "train_users": 58748,
  "bridge_users": 36124,
  "holdout_movie_to_game_test": 1609,
  "holdout_game_to_movie_test": 1609,
  "holdout_movie_to_game_val": 1609,
  "holdout_game_to_movie_val": 1609,
  "val_within_movie": 2000,
  "val_within_game": 2000,
  "val_mixed": 2000,
  "val_cold_start": 2000,
  "val_movie_to_game": 1609,
  "val_game_to_movie": 1609,
  "test_within_movie": 2000,
  "test_within_game": 2000,
  "test_mixed": 2000,
  "test_cold_start": 2000,
  "test_movie_to_game": 1609,
  "test_game_to_movie": 1609
}
```

## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)

| task | vs baseline | n | ndcg@10 model | ndcg@10 baseline | lift | 95% CI (abs) | significant |
|---|---|---:|---:|---:|---:|---|:---:|
| within_movie | popularity | 2000 | 0.0411 | 0.0093 | +342.4% | [+0.0252, +0.0391] | yes |
| within_game | popularity | 2000 | 0.0616 | 0.0240 | +156.8% | [+0.0291, +0.0466] | yes |
| mixed | popularity | 2000 | 0.0350 | 0.0126 | +177.7% | [+0.0155, +0.0292] | yes |
| cold_start | popularity | 2000 | 0.0048 | 0.0110 | -56.2% | [-0.0100, -0.0026] | yes |
| movie_to_game | popularity | 1609 | 0.0352 | 0.0340 | +3.5% | [-0.0032, +0.0057] | no |
| game_to_movie | popularity | 1609 | 0.0268 | 0.0194 | +38.2% | [+0.0039, +0.0106] | yes |
| within_movie | rrf_fusion | 2000 | 0.0411 | 0.0320 | +28.3% | [+0.0036, +0.0145] | yes |
| within_game | rrf_fusion | 2000 | 0.0616 | 0.0644 | -4.4% | [-0.0088, +0.0031] | no |
| mixed | rrf_fusion | 2000 | 0.0350 | 0.0330 | +6.0% | [-0.0028, +0.0070] | no |
| cold_start | rrf_fusion | 2000 | 0.0048 | 0.0042 | +15.1% | [-0.0005, +0.0017] | no |
| movie_to_game | rrf_fusion | 1609 | 0.0352 | 0.0292 | +20.8% | [+0.0020, +0.0098] | yes |
| game_to_movie | rrf_fusion | 1609 | 0.0268 | 0.0271 | -0.8% | [-0.0036, +0.0029] | no |

## Results (all users)

### within_movie (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0093 | 0.0162 | 0.0244 | 0.0325 | 0.0005 | **0.7414** | 0.9994 |
| item_knn | 0.0382 | 0.0566 | 0.0750 | 0.0900 | 0.1460 | 0.6609 | 0.9130 |
| als_single_domain | 0.0286 | 0.0460 | 0.0687 | 0.0820 | 0.0447 | 0.6716 | 0.9804 |
| als_joint | 0.0266 | 0.0434 | 0.0621 | 0.0775 | 0.0351 | 0.6570 | 0.9843 |
| content | 0.0107 | 0.0178 | 0.0270 | 0.0320 | **0.2693** | 0.3763 | 0.6431 |
| two_tower | 0.0223 | 0.0374 | 0.0595 | 0.0635 | 0.1131 | 0.5605 | 0.8942 |
| rrf_fusion | 0.0320 | 0.0533 | 0.0782 | 0.0895 | 0.0531 | 0.6431 | 0.9777 |
| crossverse_ranker | **0.0421** | **0.0648** | **0.0897** | **0.1060** | 0.1449 | 0.6236 | 0.9056 |
| crossverse_ranker+diversity | 0.0411 | 0.0622 | 0.0884 | 0.1010 | 0.1368 | 0.6607 | 0.9145 |

### within_game (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0240 | 0.0388 | 0.0631 | 0.0590 | 0.0005 | 0.6310 | 0.9998 |
| item_knn | **0.0664** | 0.1096 | 0.1492 | 0.1500 | 0.0975 | 0.6440 | 0.9408 |
| als_single_domain | 0.0576 | 0.0981 | 0.1436 | 0.1435 | 0.0373 | 0.6543 | 0.9811 |
| als_joint | 0.0570 | 0.0953 | 0.1419 | 0.1360 | 0.0224 | 0.6482 | 0.9901 |
| content | 0.0144 | 0.0256 | 0.0445 | 0.0390 | **0.1744** | 0.3524 | 0.6535 |
| two_tower | 0.0461 | 0.0841 | 0.1273 | 0.1210 | 0.0739 | 0.5610 | 0.9386 |
| rrf_fusion | 0.0644 | **0.1109** | **0.1575** | **0.1605** | 0.0479 | 0.6105 | 0.9762 |
| crossverse_ranker | 0.0646 | 0.1060 | 0.1535 | 0.1470 | 0.0904 | 0.6165 | 0.9434 |
| crossverse_ranker+diversity | 0.0616 | 0.1003 | 0.1441 | 0.1405 | 0.0888 | **0.6813** | 0.9470 |

### mixed (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0126 | 0.0183 | 0.0264 | 0.0380 | 0.0006 | **0.7269** | 0.9998 |
| item_knn | 0.0349 | 0.0517 | 0.0733 | 0.0995 | 0.1080 | 0.6713 | 0.9628 |
| als_joint | 0.0271 | 0.0416 | 0.0627 | 0.0830 | 0.0419 | 0.6873 | 0.9906 |
| content | 0.0066 | 0.0108 | 0.0183 | 0.0255 | **0.2674** | 0.3891 | 0.6650 |
| two_tower | 0.0210 | 0.0336 | 0.0543 | 0.0710 | 0.1192 | 0.6063 | 0.9419 |
| rrf_fusion | 0.0330 | 0.0525 | 0.0759 | **0.1045** | 0.0662 | 0.6676 | 0.9851 |
| crossverse_ranker | **0.0351** | **0.0526** | **0.0770** | 0.1015 | 0.1451 | 0.6571 | 0.9379 |
| crossverse_ranker+diversity | 0.0350 | 0.0524 | 0.0753 | 0.1015 | 0.1455 | 0.6619 | 0.9378 |

### cold_start (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0110** | **0.0172** | **0.0303** | **0.0330** | 0.0005 | **0.7267** | 0.9998 |
| content | 0.0003 | 0.0008 | 0.0016 | 0.0010 | 0.0599 | 0.2803 | 0.3811 |
| rrf_fusion | 0.0042 | 0.0078 | 0.0119 | 0.0195 | **0.1073** | 0.5511 | 0.8068 |
| crossverse_ranker | 0.0050 | 0.0092 | 0.0149 | 0.0220 | 0.0962 | 0.5723 | 0.9025 |
| crossverse_ranker+diversity | 0.0048 | 0.0086 | 0.0152 | 0.0220 | 0.0978 | 0.5813 | 0.9001 |

### movie_to_game (n=1609)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0340 | 0.0426 | 0.0569 | 0.1603 | 0.0004 | 0.6310 | 0.9998 |
| item_knn | 0.0198 | 0.0220 | 0.0265 | 0.0883 | **0.1278** | **0.7629** | 0.7566 |
| als_joint | 0.0210 | 0.0260 | 0.0413 | 0.1050 | 0.0187 | 0.7301 | 0.9934 |
| content | 0.0054 | 0.0069 | 0.0100 | 0.0267 | 0.0664 | 0.5099 | 0.5567 |
| copref | 0.0241 | 0.0298 | 0.0476 | 0.1212 | 0.0846 | 0.7262 | 0.9463 |
| two_tower | 0.0222 | 0.0287 | 0.0515 | 0.1187 | 0.0430 | 0.7153 | 0.9625 |
| rrf_fusion | 0.0292 | 0.0371 | 0.0573 | 0.1411 | 0.0256 | 0.7518 | 0.9158 |
| crossverse_ranker | **0.0372** | **0.0473** | **0.0734** | **0.1809** | 0.0212 | 0.6948 | 0.9914 |
| crossverse_ranker+diversity | 0.0352 | 0.0425 | 0.0675 | 0.1684 | 0.0211 | 0.7226 | 0.9912 |

### game_to_movie (n=1609)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0194 | 0.0246 | 0.0361 | 0.1237 | 0.0004 | 0.7427 | 0.9994 |
| item_knn | 0.0211 | 0.0219 | 0.0260 | 0.1131 | **0.1393** | **0.7450** | 0.8418 |
| als_joint | 0.0191 | 0.0219 | 0.0305 | 0.1181 | 0.0220 | 0.7153 | 0.9893 |
| content | 0.0044 | 0.0056 | 0.0066 | 0.0267 | 0.1026 | 0.5068 | 0.5009 |
| copref | 0.0217 | 0.0230 | 0.0305 | 0.1224 | 0.1271 | 0.7272 | 0.9152 |
| two_tower | 0.0180 | 0.0214 | 0.0312 | 0.1119 | 0.0500 | 0.7022 | 0.9381 |
| rrf_fusion | **0.0271** | 0.0283 | 0.0388 | 0.1541 | 0.0224 | 0.7181 | 0.9859 |
| crossverse_ranker | 0.0270 | 0.0301 | **0.0422** | 0.1541 | 0.0274 | 0.7000 | 0.9880 |
| crossverse_ranker+diversity | 0.0268 | **0.0303** | 0.0414 | **0.1554** | 0.0278 | 0.7232 | 0.9880 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0048 (n=2000) | 0.0051 (n=498) | 0.0059 (n=917) | 0.0028 (n=585) | 0.0043 (n=1335) |
| game_to_movie | 0.0268 (n=1609) | 0.0479 (n=161) | 0.0328 (n=381) | 0.0215 (n=1067) | 0.0268 (n=1609) |
| mixed | 0.0350 (n=2000) | 0.0238 (n=606) | 0.0352 (n=935) | 0.0496 (n=459) | 0.0350 (n=2000) |
| movie_to_game | 0.0352 (n=1609) | 0.0463 (n=348) | 0.0293 (n=419) | 0.0335 (n=842) | 0.0352 (n=1609) |
| within_game | 0.0616 (n=2000) | 0.0478 (n=529) | 0.0526 (n=906) | 0.0889 (n=565) | 0.0511 (n=1369) |
| within_movie | 0.0411 (n=2000) | 0.0288 (n=641) | 0.0417 (n=922) | 0.0578 (n=437) | 0.0445 (n=1448) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 17,650 |
| n_sources | 14,645 |
| item_knn_rank | 11,666 |
| item_knn_z | 9,430 |
| log_rating_count | 8,514 |
| max_sim_pos | 8,358 |
| als_z | 7,441 |
| popularity_z | 7,386 |
| item_knn_score | 7,339 |
| bayes_rating | 6,915 |
| two_tower_rank | 6,372 |
| popularity_score | 6,211 |
| year_gap | 5,353 |
| content_z | 5,197 |
| two_tower_score | 4,833 |