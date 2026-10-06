# CrossVerse offline benchmark — v9a

Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.
`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.

## Split

```json
{
  "train_interactions": 1158348,
  "train_users": 172254,
  "bridge_users": 119071,
  "holdout_movie_to_game_test": 2505,
  "holdout_game_to_movie_test": 2505,
  "holdout_movie_to_game_val": 2505,
  "holdout_game_to_movie_val": 2505,
  "val_within_movie": 3000,
  "val_within_game": 3000,
  "val_mixed": 3000,
  "val_cold_start": 3000,
  "val_movie_to_game": 2505,
  "val_game_to_movie": 2505,
  "test_within_movie": 3000,
  "test_within_game": 3000,
  "test_mixed": 3000,
  "test_cold_start": 3000,
  "test_movie_to_game": 2505,
  "test_game_to_movie": 2505
}
```

## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)

| task | vs baseline | n | ndcg@10 model | ndcg@10 baseline | lift | 95% CI (abs) | significant |
|---|---|---:|---:|---:|---:|---|:---:|
| within_movie | popularity | 3000 | 0.0424 | 0.0086 | +390.9% | [+0.0280, +0.0394] | yes |
| within_game | popularity | 3000 | 0.0549 | 0.0195 | +180.9% | [+0.0289, +0.0423] | yes |
| mixed | popularity | 3000 | 0.0387 | 0.0070 | +448.6% | [+0.0262, +0.0372] | yes |
| cold_start | popularity | 3000 | 0.0043 | 0.0084 | -48.8% | [-0.0068, -0.0015] | yes |
| movie_to_game | popularity | 2505 | 0.0334 | 0.0307 | +8.7% | [-0.0009, +0.0064] | no |
| game_to_movie | popularity | 2505 | 0.0302 | 0.0213 | +41.7% | [+0.0059, +0.0122] | yes |
| within_movie | rrf_fusion | 3000 | 0.0424 | 0.0314 | +35.0% | [+0.0063, +0.0156] | yes |
| within_game | rrf_fusion | 3000 | 0.0549 | 0.0567 | -3.2% | [-0.0066, +0.0032] | no |
| mixed | rrf_fusion | 3000 | 0.0387 | 0.0352 | +9.7% | [-0.0008, +0.0075] | no |
| cold_start | rrf_fusion | 3000 | 0.0043 | 0.0041 | +3.4% | [-0.0012, +0.0014] | no |
| movie_to_game | rrf_fusion | 2505 | 0.0334 | 0.0146 | +129.0% | [+0.0155, +0.0222] | yes |
| game_to_movie | rrf_fusion | 2505 | 0.0302 | 0.0210 | +43.8% | [+0.0062, +0.0120] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0086 | 0.0160 | 0.0247 | 0.0307 | 0.0004 | **0.7161** | 0.9998 |
| item_knn | 0.0338 | 0.0508 | 0.0666 | 0.0773 | **0.2644** | 0.7019 | 0.6564 |
| als_single_domain | 0.0268 | 0.0420 | 0.0614 | 0.0757 | 0.0278 | 0.6500 | 0.9903 |
| als_joint | 0.0268 | 0.0405 | 0.0601 | 0.0757 | 0.0226 | 0.6358 | 0.9916 |
| content | 0.0087 | 0.0133 | 0.0204 | 0.0233 | 0.2412 | 0.3620 | 0.6979 |
| two_tower | 0.0253 | 0.0421 | 0.0585 | 0.0760 | 0.0984 | 0.5459 | 0.9326 |
| rrf_fusion | 0.0314 | 0.0516 | 0.0776 | 0.0920 | 0.0451 | 0.6395 | 0.9855 |
| crossverse_ranker | 0.0421 | 0.0641 | **0.0906** | **0.1087** | 0.1407 | 0.6120 | 0.9229 |
| crossverse_ranker+diversity | **0.0424** | **0.0642** | 0.0876 | 0.1073 | 0.1369 | 0.6440 | 0.9287 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0328 | 0.0515 | 0.0457 | 0.0004 | **0.6727** | 0.9998 |
| item_knn | 0.0525 | 0.0851 | 0.1180 | 0.1157 | 0.1503 | 0.6551 | 0.7442 |
| als_single_domain | 0.0567 | 0.0940 | 0.1361 | 0.1263 | 0.0225 | 0.6454 | 0.9844 |
| als_joint | 0.0477 | 0.0838 | 0.1305 | 0.1107 | 0.0112 | 0.6342 | 0.9926 |
| content | 0.0123 | 0.0230 | 0.0419 | 0.0340 | **0.1527** | 0.3346 | 0.6600 |
| two_tower | 0.0420 | 0.0758 | 0.1211 | 0.1033 | 0.0606 | 0.5470 | 0.9433 |
| rrf_fusion | 0.0567 | 0.1016 | 0.1561 | **0.1390** | 0.0382 | 0.6019 | 0.9688 |
| crossverse_ranker | **0.0591** | **0.1049** | **0.1624** | 0.1383 | 0.0740 | 0.6072 | 0.9502 |
| crossverse_ranker+diversity | 0.0549 | 0.0933 | 0.1366 | 0.1253 | 0.0746 | 0.6591 | 0.9503 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0070 | 0.0110 | 0.0196 | 0.0213 | 0.0003 | **0.7371** | 0.9999 |
| item_knn | 0.0332 | 0.0525 | 0.0676 | 0.0833 | **0.2867** | 0.7104 | 0.7048 |
| als_joint | 0.0268 | 0.0424 | 0.0610 | 0.0740 | 0.0273 | 0.6740 | 0.9936 |
| content | 0.0086 | 0.0128 | 0.0190 | 0.0230 | 0.2504 | 0.3748 | 0.6933 |
| two_tower | 0.0218 | 0.0354 | 0.0561 | 0.0680 | 0.1013 | 0.5875 | 0.9577 |
| rrf_fusion | 0.0352 | 0.0566 | 0.0781 | 0.1007 | 0.0666 | 0.6303 | 0.9826 |
| crossverse_ranker | **0.0389** | **0.0635** | 0.0892 | **0.1047** | 0.1525 | 0.6533 | 0.9349 |
| crossverse_ranker+diversity | 0.0387 | 0.0630 | **0.0892** | 0.1043 | 0.1525 | 0.6556 | 0.9351 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0134** | **0.0219** | **0.0223** | 0.0003 | **0.7372** | 0.9999 |
| content | 0.0001 | 0.0001 | 0.0006 | 0.0007 | 0.0421 | 0.2495 | 0.3988 |
| rrf_fusion | 0.0041 | 0.0061 | 0.0106 | 0.0143 | **0.0782** | 0.5479 | 0.8090 |
| crossverse_ranker | 0.0046 | 0.0083 | 0.0117 | 0.0150 | 0.0723 | 0.5592 | 0.9101 |
| crossverse_ranker+diversity | 0.0043 | 0.0076 | 0.0112 | 0.0143 | 0.0731 | 0.5606 | 0.9091 |

### movie_to_game (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0307 | 0.0384 | 0.0574 | 0.1285 | 0.0002 | 0.6684 | 0.9998 |
| item_knn | 0.0023 | 0.0029 | 0.0044 | 0.0140 | 0.1426 | 0.7962 | 0.3547 |
| als_joint | 0.0206 | 0.0270 | 0.0437 | 0.0922 | 0.0093 | 0.7243 | 0.9953 |
| content | 0.0034 | 0.0045 | 0.0081 | 0.0136 | 0.0613 | 0.4979 | 0.5717 |
| copref | 0.0018 | 0.0021 | 0.0040 | 0.0116 | **0.1622** | 0.7732 | 0.3758 |
| two_tower | 0.0232 | 0.0310 | 0.0530 | 0.1114 | 0.0367 | 0.6997 | 0.9627 |
| rrf_fusion | 0.0146 | 0.0188 | 0.0333 | 0.0719 | 0.0148 | **0.8155** | 0.6386 |
| crossverse_ranker | **0.0341** | **0.0461** | **0.0765** | **0.1569** | 0.0196 | 0.6830 | 0.9915 |
| crossverse_ranker+diversity | 0.0334 | 0.0442 | 0.0696 | 0.1553 | 0.0188 | 0.7273 | 0.9914 |

### game_to_movie (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0213 | 0.0255 | 0.0358 | 0.1182 | 0.0002 | 0.7177 | 0.9998 |
| item_knn | 0.0018 | 0.0017 | 0.0027 | 0.0084 | 0.2326 | **0.8414** | 0.2803 |
| als_joint | 0.0182 | 0.0212 | 0.0321 | 0.0930 | 0.0132 | 0.7019 | 0.9954 |
| content | 0.0030 | 0.0036 | 0.0056 | 0.0164 | 0.1011 | 0.5009 | 0.4977 |
| copref | 0.0008 | 0.0010 | 0.0021 | 0.0044 | **0.2521** | 0.8396 | 0.3105 |
| two_tower | 0.0218 | 0.0262 | 0.0388 | 0.1094 | 0.0444 | 0.6920 | 0.9582 |
| rrf_fusion | 0.0210 | 0.0244 | 0.0367 | 0.1106 | 0.0181 | 0.7354 | 0.9732 |
| crossverse_ranker | **0.0306** | **0.0353** | **0.0485** | **0.1473** | 0.0225 | 0.6815 | 0.9929 |
| crossverse_ranker+diversity | 0.0302 | 0.0347 | 0.0484 | 0.1461 | 0.0219 | 0.7125 | 0.9930 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0043 (n=3000) | 0.0047 (n=990) | 0.0044 (n=916) | 0.0038 (n=1094) | 0.0041 (n=2018) |
| game_to_movie | 0.0302 (n=2505) | 0.0450 (n=379) | 0.0313 (n=332) | 0.0269 (n=1794) | 0.0302 (n=2505) |
| mixed | 0.0387 (n=3000) | 0.0337 (n=1087) | 0.0347 (n=899) | 0.0475 (n=1014) | 0.0387 (n=3000) |
| movie_to_game | 0.0334 (n=2505) | 0.0426 (n=629) | 0.0309 (n=384) | 0.0301 (n=1492) | 0.0334 (n=2505) |
| within_game | 0.0549 (n=3000) | 0.0520 (n=1036) | 0.0468 (n=933) | 0.0651 (n=1031) | 0.0529 (n=2128) |
| within_movie | 0.0424 (n=3000) | 0.0370 (n=1218) | 0.0325 (n=866) | 0.0589 (n=916) | 0.0439 (n=2016) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 23,612 |
| item_knn_z | 12,013 |
| item_knn_score | 8,146 |
| log_rating_count | 7,422 |
| item_knn_rank | 5,676 |
| two_tower_rank | 5,596 |
| n_sources | 5,517 |
| log_pop | 4,952 |
| als_z | 4,112 |
| popularity_z | 3,355 |
| max_sim_pos | 3,102 |
| bayes_rating | 3,037 |
| popularity_rank | 2,903 |
| two_tower_score | 2,642 |
| als_score | 2,408 |