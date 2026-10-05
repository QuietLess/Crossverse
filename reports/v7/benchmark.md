# CrossVerse offline benchmark — v7

Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.
`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.

## Split

```json
{
  "train_interactions": 1154941,
  "train_users": 172184,
  "bridge_users": 119016,
  "holdout_movie_to_game_test": 2504,
  "holdout_game_to_movie_test": 2504,
  "holdout_movie_to_game_val": 2504,
  "holdout_game_to_movie_val": 2504,
  "val_within_movie": 3000,
  "val_within_game": 3000,
  "val_mixed": 3000,
  "val_cold_start": 3000,
  "val_movie_to_game": 2504,
  "val_game_to_movie": 2504,
  "test_within_movie": 3000,
  "test_within_game": 3000,
  "test_mixed": 3000,
  "test_cold_start": 3000,
  "test_movie_to_game": 2504,
  "test_game_to_movie": 2504
}
```

## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)

| task | vs baseline | n | ndcg@10 model | ndcg@10 baseline | lift | 95% CI (abs) | significant |
|---|---|---:|---:|---:|---:|---|:---:|
| within_movie | popularity | 3000 | 0.0454 | 0.0097 | +370.7% | [+0.0300, +0.0419] | yes |
| within_game | popularity | 3000 | 0.0624 | 0.0203 | +207.5% | [+0.0351, +0.0491] | yes |
| mixed | popularity | 3000 | 0.0412 | 0.0062 | +562.5% | [+0.0296, +0.0402] | yes |
| cold_start | popularity | 3000 | 0.0066 | 0.0067 | -1.2% | [-0.0027, +0.0025] | no |
| movie_to_game | popularity | 2504 | 0.0338 | 0.0290 | +16.4% | [+0.0014, +0.0080] | yes |
| game_to_movie | popularity | 2504 | 0.0267 | 0.0216 | +23.7% | [+0.0022, +0.0083] | yes |
| within_movie | rrf_fusion | 3000 | 0.0454 | 0.0336 | +35.4% | [+0.0077, +0.0163] | yes |
| within_game | rrf_fusion | 3000 | 0.0624 | 0.0612 | +1.9% | [-0.0041, +0.0066] | no |
| mixed | rrf_fusion | 3000 | 0.0412 | 0.0332 | +24.2% | [+0.0043, +0.0120] | yes |
| cold_start | rrf_fusion | 3000 | 0.0066 | 0.0049 | +34.5% | [+0.0006, +0.0029] | yes |
| movie_to_game | rrf_fusion | 2504 | 0.0338 | 0.0131 | +158.4% | [+0.0173, +0.0242] | yes |
| game_to_movie | rrf_fusion | 2504 | 0.0267 | 0.0206 | +29.9% | [+0.0033, +0.0091] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0097 | 0.0186 | 0.0271 | 0.0313 | 0.0004 | **0.7145** | 0.9998 |
| item_knn | 0.0412 | 0.0566 | 0.0730 | 0.0897 | **0.2650** | 0.7020 | 0.6518 |
| als_single_domain | 0.0297 | 0.0447 | 0.0649 | 0.0717 | 0.0282 | 0.6478 | 0.9900 |
| als_joint | 0.0299 | 0.0444 | 0.0674 | 0.0760 | 0.0227 | 0.6332 | 0.9913 |
| content | 0.0114 | 0.0199 | 0.0269 | 0.0293 | 0.2374 | 0.3620 | 0.6971 |
| two_tower | 0.0260 | 0.0446 | 0.0653 | 0.0767 | 0.0982 | 0.5452 | 0.9294 |
| rrf_fusion | 0.0336 | 0.0536 | 0.0842 | 0.0847 | 0.0469 | 0.6210 | 0.9827 |
| crossverse_ranker | **0.0464** | **0.0700** | **0.0985** | **0.1117** | 0.1516 | 0.6093 | 0.9091 |
| crossverse_ranker+diversity | 0.0454 | 0.0682 | 0.0953 | 0.1080 | 0.1478 | 0.6420 | 0.9148 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0203 | 0.0367 | 0.0514 | 0.0513 | 0.0004 | 0.6084 | 0.9998 |
| item_knn | 0.0509 | 0.0834 | 0.1160 | 0.1113 | 0.1487 | 0.6487 | 0.7415 |
| als_single_domain | 0.0589 | 0.0978 | 0.1434 | 0.1290 | 0.0218 | 0.6412 | 0.9841 |
| als_joint | 0.0511 | 0.0853 | 0.1307 | 0.1127 | 0.0115 | 0.6266 | 0.9925 |
| content | 0.0159 | 0.0276 | 0.0445 | 0.0397 | **0.1513** | 0.3336 | 0.6578 |
| two_tower | 0.0427 | 0.0745 | 0.1164 | 0.1017 | 0.0625 | 0.5397 | 0.9377 |
| rrf_fusion | 0.0612 | 0.1030 | 0.1626 | 0.1363 | 0.0391 | 0.5914 | 0.9669 |
| crossverse_ranker | **0.0645** | **0.1082** | **0.1651** | **0.1437** | 0.0759 | 0.5979 | 0.9468 |
| crossverse_ranker+diversity | 0.0624 | 0.1030 | 0.1467 | 0.1373 | 0.0761 | **0.6517** | 0.9472 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0062 | 0.0113 | 0.0226 | 0.0213 | 0.0003 | **0.7361** | 0.9999 |
| item_knn | 0.0380 | 0.0536 | 0.0671 | 0.0907 | **0.2869** | 0.7097 | 0.6964 |
| als_joint | 0.0242 | 0.0398 | 0.0628 | 0.0693 | 0.0279 | 0.6671 | 0.9935 |
| content | 0.0091 | 0.0147 | 0.0225 | 0.0250 | 0.2500 | 0.3728 | 0.6917 |
| two_tower | 0.0233 | 0.0385 | 0.0580 | 0.0730 | 0.1051 | 0.5816 | 0.9542 |
| rrf_fusion | 0.0332 | 0.0524 | 0.0840 | 0.0887 | 0.0668 | 0.6227 | 0.9820 |
| crossverse_ranker | 0.0412 | **0.0644** | **0.0939** | **0.1080** | 0.1685 | 0.6459 | 0.9193 |
| crossverse_ranker+diversity | **0.0412** | **0.0644** | 0.0919 | **0.1080** | 0.1684 | 0.6482 | 0.9195 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0067** | **0.0129** | **0.0242** | **0.0220** | 0.0003 | **0.7361** | 0.9999 |
| content | 0.0003 | 0.0006 | 0.0014 | 0.0017 | 0.0416 | 0.2494 | 0.4061 |
| rrf_fusion | 0.0049 | 0.0086 | 0.0125 | 0.0150 | **0.0776** | 0.5467 | 0.8085 |
| crossverse_ranker | 0.0067 | 0.0116 | 0.0155 | 0.0200 | 0.0698 | 0.5551 | 0.9072 |
| crossverse_ranker+diversity | 0.0066 | 0.0115 | 0.0153 | 0.0200 | 0.0702 | 0.5559 | 0.9064 |

### movie_to_game (n=2504)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0290 | 0.0368 | 0.0531 | 0.1238 | 0.0002 | 0.6026 | 0.9998 |
| item_knn | 0.0022 | 0.0024 | 0.0042 | 0.0096 | 0.1399 | 0.7883 | 0.3453 |
| als_joint | 0.0165 | 0.0233 | 0.0411 | 0.0875 | 0.0097 | 0.7211 | 0.9952 |
| content | 0.0037 | 0.0056 | 0.0088 | 0.0192 | 0.0589 | 0.4964 | 0.5726 |
| copref | 0.0015 | 0.0023 | 0.0054 | 0.0088 | **0.1571** | 0.7725 | 0.3678 |
| two_tower | 0.0251 | 0.0344 | 0.0536 | 0.1198 | 0.0351 | 0.7053 | 0.9627 |
| rrf_fusion | 0.0131 | 0.0172 | 0.0267 | 0.0623 | 0.0151 | **0.8051** | 0.6825 |
| crossverse_ranker | **0.0354** | **0.0463** | **0.0718** | **0.1593** | 0.0193 | 0.6783 | 0.9918 |
| crossverse_ranker+diversity | 0.0338 | 0.0443 | 0.0680 | 0.1530 | 0.0188 | 0.7164 | 0.9916 |

### game_to_movie (n=2504)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0216 | 0.0259 | 0.0352 | 0.1174 | 0.0002 | 0.7162 | 0.9998 |
| item_knn | 0.0014 | 0.0021 | 0.0023 | 0.0116 | 0.2268 | 0.8426 | 0.2724 |
| als_joint | 0.0165 | 0.0192 | 0.0296 | 0.0863 | 0.0129 | 0.7015 | 0.9955 |
| content | 0.0022 | 0.0027 | 0.0044 | 0.0144 | 0.0970 | 0.5018 | 0.5041 |
| copref | 0.0007 | 0.0008 | 0.0014 | 0.0056 | **0.2459** | **0.8434** | 0.3048 |
| two_tower | 0.0188 | 0.0224 | 0.0329 | 0.0986 | 0.0450 | 0.6996 | 0.9573 |
| rrf_fusion | 0.0206 | 0.0237 | 0.0325 | 0.1078 | 0.0204 | 0.7433 | 0.9414 |
| crossverse_ranker | **0.0277** | **0.0337** | **0.0468** | **0.1434** | 0.0219 | 0.6792 | 0.9921 |
| crossverse_ranker+diversity | 0.0267 | 0.0323 | 0.0451 | 0.1386 | 0.0210 | 0.7095 | 0.9924 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0066 (n=3000) | 0.0070 (n=1017) | 0.0085 (n=851) | 0.0048 (n=1132) | 0.0055 (n=2009) |
| game_to_movie | 0.0267 (n=2504) | 0.0407 (n=311) | 0.0274 (n=345) | 0.0243 (n=1848) | 0.0267 (n=2504) |
| mixed | 0.0412 (n=3000) | 0.0341 (n=1134) | 0.0465 (n=833) | 0.0448 (n=1033) | 0.0412 (n=3000) |
| movie_to_game | 0.0338 (n=2504) | 0.0384 (n=660) | 0.0391 (n=363) | 0.0304 (n=1481) | 0.0338 (n=2504) |
| within_game | 0.0624 (n=3000) | 0.0536 (n=1076) | 0.0591 (n=897) | 0.0744 (n=1027) | 0.0560 (n=2139) |
| within_movie | 0.0454 (n=3000) | 0.0374 (n=1234) | 0.0430 (n=843) | 0.0584 (n=923) | 0.0476 (n=2001) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 19,335 |
| item_knn_z | 13,950 |
| two_tower_rank | 13,226 |
| n_sources | 11,884 |
| item_knn_score | 11,621 |
| log_rating_count | 11,165 |
| item_knn_rank | 6,979 |
| max_sim_pos | 6,119 |
| log_pop | 5,871 |
| popularity_z | 5,841 |
| als_z | 5,564 |
| popularity_score | 4,860 |
| popularity_rank | 4,729 |
| year_gap | 4,710 |
| two_tower_score | 4,702 |