# CrossVerse offline benchmark — v12

Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.
`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.

## Split

```json
{
  "train_interactions": 1155125,
  "train_users": 172235,
  "bridge_users": 119064,
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
| within_movie | popularity | 3000 | 0.0470 | 0.0078 | +504.2% | [+0.0333, +0.0451] | yes |
| within_game | popularity | 3000 | 0.0570 | 0.0187 | +204.4% | [+0.0317, +0.0447] | yes |
| mixed | popularity | 3000 | 0.0414 | 0.0053 | +688.2% | [+0.0310, +0.0415] | yes |
| cold_start | popularity | 3000 | 0.0046 | 0.0070 | -34.3% | [-0.0048, -0.0001] | yes |
| movie_to_game | popularity | 2505 | 0.0324 | 0.0307 | +5.4% | [-0.0025, +0.0059] | no |
| game_to_movie | popularity | 2505 | 0.0265 | 0.0212 | +24.9% | [+0.0020, +0.0086] | yes |
| within_movie | rrf_fusion | 3000 | 0.0470 | 0.0391 | +20.3% | [+0.0034, +0.0124] | yes |
| within_game | rrf_fusion | 3000 | 0.0570 | 0.0600 | -4.9% | [-0.0083, +0.0023] | no |
| mixed | rrf_fusion | 3000 | 0.0414 | 0.0355 | +16.7% | [+0.0022, +0.0096] | yes |
| cold_start | rrf_fusion | 3000 | 0.0046 | 0.0048 | -5.0% | [-0.0017, +0.0012] | no |
| movie_to_game | rrf_fusion | 2505 | 0.0324 | 0.0152 | +112.5% | [+0.0140, +0.0203] | yes |
| game_to_movie | rrf_fusion | 2505 | 0.0265 | 0.0207 | +28.4% | [+0.0029, +0.0086] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0078 | 0.0134 | 0.0218 | 0.0287 | 0.0004 | **0.7470** | 0.9998 |
| item_knn | 0.0435 | 0.0646 | 0.0805 | 0.1020 | 0.2406 | 0.6684 | 0.7660 |
| als_single_domain | 0.0326 | 0.0499 | 0.0700 | 0.0813 | 0.0294 | 0.6604 | 0.9897 |
| als_joint | 0.0292 | 0.0456 | 0.0673 | 0.0757 | 0.0237 | 0.6443 | 0.9912 |
| content | 0.0142 | 0.0211 | 0.0303 | 0.0343 | **0.2559** | 0.3706 | 0.6680 |
| two_tower | 0.0292 | 0.0486 | 0.0687 | 0.0797 | 0.1051 | 0.5322 | 0.9275 |
| rrf_fusion | 0.0391 | 0.0604 | 0.0895 | 0.1003 | 0.0599 | 0.6303 | 0.9763 |
| crossverse_ranker | **0.0478** | **0.0715** | **0.0973** | **0.1170** | 0.1435 | 0.6075 | 0.9176 |
| crossverse_ranker+diversity | 0.0470 | 0.0696 | 0.0947 | 0.1143 | 0.1379 | 0.6392 | 0.9249 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0187 | 0.0331 | 0.0529 | 0.0480 | 0.0004 | **0.6243** | 0.9998 |
| item_knn | 0.0535 | 0.0910 | 0.1258 | 0.1227 | 0.1133 | 0.5956 | 0.8491 |
| als_single_domain | 0.0509 | 0.0826 | 0.1242 | 0.1120 | 0.0229 | 0.5955 | 0.9842 |
| als_joint | 0.0464 | 0.0793 | 0.1228 | 0.1063 | 0.0111 | 0.5879 | 0.9927 |
| content | 0.0120 | 0.0231 | 0.0425 | 0.0313 | **0.1532** | 0.3015 | 0.6823 |
| two_tower | 0.0458 | 0.0823 | 0.1265 | 0.1130 | 0.0625 | 0.4875 | 0.9429 |
| rrf_fusion | 0.0600 | 0.1042 | 0.1505 | **0.1403** | 0.0469 | 0.5173 | 0.9561 |
| crossverse_ranker | **0.0601** | **0.1058** | **0.1546** | 0.1397 | 0.0707 | 0.5432 | 0.9526 |
| crossverse_ranker+diversity | 0.0570 | 0.0971 | 0.1422 | 0.1290 | 0.0695 | 0.6023 | 0.9549 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0053 | 0.0089 | 0.0187 | 0.0190 | 0.0004 | **0.7195** | 0.9999 |
| item_knn | 0.0403 | 0.0617 | 0.0818 | 0.0997 | 0.2623 | 0.6764 | 0.7967 |
| als_joint | 0.0245 | 0.0385 | 0.0616 | 0.0660 | 0.0290 | 0.6589 | 0.9934 |
| content | 0.0099 | 0.0161 | 0.0247 | 0.0280 | **0.2759** | 0.3677 | 0.6803 |
| two_tower | 0.0246 | 0.0405 | 0.0614 | 0.0693 | 0.1123 | 0.5646 | 0.9540 |
| rrf_fusion | 0.0355 | 0.0571 | 0.0895 | 0.0937 | 0.0781 | 0.6121 | 0.9773 |
| crossverse_ranker | **0.0416** | **0.0667** | **0.0967** | **0.1107** | 0.1552 | 0.6302 | 0.9337 |
| crossverse_ranker+diversity | 0.0414 | 0.0660 | 0.0939 | 0.1100 | 0.1554 | 0.6339 | 0.9338 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0070** | **0.0124** | **0.0211** | **0.0227** | 0.0004 | **0.7194** | 0.9999 |
| content | 0.0004 | 0.0009 | 0.0022 | 0.0020 | 0.0832 | 0.3408 | 0.4337 |
| rrf_fusion | 0.0048 | 0.0084 | 0.0131 | 0.0147 | **0.1431** | 0.4663 | 0.6832 |
| crossverse_ranker | 0.0051 | 0.0096 | 0.0165 | 0.0170 | 0.1254 | 0.4945 | 0.8790 |
| crossverse_ranker+diversity | 0.0046 | 0.0082 | 0.0117 | 0.0143 | 0.1305 | 0.5139 | 0.8696 |

### movie_to_game (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0307 | 0.0391 | 0.0584 | 0.1329 | 0.0002 | 0.6240 | 0.9998 |
| item_knn | 0.0051 | 0.0060 | 0.0081 | 0.0259 | 0.0515 | 0.7522 | 0.5223 |
| als_joint | 0.0206 | 0.0281 | 0.0444 | 0.0982 | 0.0098 | 0.6959 | 0.9951 |
| content | 0.0029 | 0.0044 | 0.0080 | 0.0160 | **0.0589** | 0.5064 | 0.5413 |
| copref | 0.0183 | 0.0233 | 0.0358 | 0.0826 | 0.0247 | 0.7062 | 0.7625 |
| two_tower | 0.0270 | 0.0360 | 0.0550 | 0.1250 | 0.0368 | 0.6756 | 0.9647 |
| rrf_fusion | 0.0152 | 0.0196 | 0.0325 | 0.0731 | 0.0140 | **0.7541** | 0.6318 |
| crossverse_ranker | **0.0334** | **0.0449** | **0.0718** | **0.1541** | 0.0199 | 0.6386 | 0.9918 |
| crossverse_ranker+diversity | 0.0324 | 0.0431 | 0.0672 | 0.1513 | 0.0192 | 0.6872 | 0.9916 |

### game_to_movie (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0212 | 0.0247 | 0.0360 | 0.1150 | 0.0002 | 0.7469 | 0.9998 |
| item_knn | 0.0076 | 0.0099 | 0.0154 | 0.0451 | **0.0992** | **0.7823** | 0.7954 |
| als_joint | 0.0169 | 0.0199 | 0.0295 | 0.0870 | 0.0130 | 0.7258 | 0.9954 |
| content | 0.0027 | 0.0036 | 0.0075 | 0.0172 | 0.0914 | 0.4957 | 0.5451 |
| copref | 0.0173 | 0.0204 | 0.0252 | 0.0846 | 0.0342 | 0.7394 | 0.9409 |
| two_tower | 0.0193 | 0.0220 | 0.0325 | 0.0970 | 0.0468 | 0.6907 | 0.9560 |
| rrf_fusion | 0.0207 | 0.0242 | 0.0341 | 0.1054 | 0.0110 | 0.7478 | 0.9822 |
| crossverse_ranker | **0.0271** | **0.0304** | **0.0438** | **0.1341** | 0.0217 | 0.6974 | 0.9918 |
| crossverse_ranker+diversity | 0.0265 | 0.0299 | 0.0428 | 0.1337 | 0.0210 | 0.7183 | 0.9924 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0046 (n=3000) | 0.0032 (n=1020) | 0.0071 (n=880) | 0.0038 (n=1100) | 0.0034 (n=2004) |
| game_to_movie | 0.0265 (n=2505) | 0.0506 (n=373) | 0.0285 (n=320) | 0.0212 (n=1812) | 0.0265 (n=2505) |
| mixed | 0.0414 (n=3000) | 0.0367 (n=1108) | 0.0425 (n=833) | 0.0454 (n=1059) | 0.0414 (n=3000) |
| movie_to_game | 0.0324 (n=2505) | 0.0403 (n=645) | 0.0247 (n=379) | 0.0309 (n=1481) | 0.0324 (n=2505) |
| within_game | 0.0570 (n=3000) | 0.0492 (n=1058) | 0.0575 (n=891) | 0.0646 (n=1051) | 0.0544 (n=2121) |
| within_movie | 0.0470 (n=3000) | 0.0391 (n=1239) | 0.0539 (n=856) | 0.0514 (n=905) | 0.0491 (n=1982) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 25,152 |
| item_knn_z | 22,233 |
| item_knn_score | 16,317 |
| log_rating_count | 12,440 |
| n_sources | 10,754 |
| two_tower_rank | 10,209 |
| popularity_score | 8,814 |
| semantic_score | 8,404 |
| semantic_z | 8,377 |
| popularity_z | 8,353 |
| item_knn_rank | 7,485 |
| max_sim_pos | 7,421 |
| year_gap | 7,110 |
| bayes_rating | 6,748 |
| als_z | 6,336 |