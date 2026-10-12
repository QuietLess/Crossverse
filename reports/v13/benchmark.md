# CrossVerse offline benchmark — v13

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
  "val_within_movie": 2000,
  "val_within_game": 2000,
  "val_mixed": 2000,
  "val_cold_start": 2000,
  "val_movie_to_game": 2000,
  "val_game_to_movie": 2000,
  "test_within_movie": 2000,
  "test_within_game": 2000,
  "test_mixed": 2000,
  "test_cold_start": 2000,
  "test_movie_to_game": 2000,
  "test_game_to_movie": 2000
}
```

## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)

| task | vs baseline | n | ndcg@10 model | ndcg@10 baseline | lift | 95% CI (abs) | significant |
|---|---|---:|---:|---:|---:|---|:---:|
| within_movie | popularity | 2000 | 0.0452 | 0.0076 | +493.1% | [+0.0311, +0.0447] | yes |
| within_game | popularity | 2000 | 0.0561 | 0.0180 | +212.1% | [+0.0301, +0.0460] | yes |
| mixed | popularity | 2000 | 0.0415 | 0.0058 | +615.6% | [+0.0290, +0.0426] | yes |
| cold_start | popularity | 2000 | 0.0042 | 0.0073 | -42.9% | [-0.0060, -0.0000] | yes |
| movie_to_game | popularity | 2000 | 0.0347 | 0.0325 | +6.6% | [-0.0023, +0.0069] | no |
| game_to_movie | popularity | 2000 | 0.0267 | 0.0212 | +26.1% | [+0.0016, +0.0092] | yes |
| within_movie | rrf_fusion | 2000 | 0.0452 | 0.0397 | +14.0% | [+0.0008, +0.0106] | yes |
| within_game | rrf_fusion | 2000 | 0.0561 | 0.0593 | -5.3% | [-0.0093, +0.0035] | no |
| mixed | rrf_fusion | 2000 | 0.0415 | 0.0352 | +18.0% | [+0.0016, +0.0109] | yes |
| cold_start | rrf_fusion | 2000 | 0.0042 | 0.0049 | -14.5% | [-0.0023, +0.0008] | no |
| movie_to_game | rrf_fusion | 2000 | 0.0347 | 0.0200 | +73.2% | [+0.0110, +0.0183] | yes |
| game_to_movie | rrf_fusion | 2000 | 0.0267 | 0.0207 | +29.0% | [+0.0030, +0.0090] | yes |

## Results (all users)

### within_movie (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0076 | 0.0134 | 0.0218 | 0.0285 | 0.0004 | **0.7474** | 0.9998 |
| item_knn | 0.0435 | 0.0639 | 0.0832 | 0.1040 | 0.1947 | 0.6696 | 0.7675 |
| als_single_domain | 0.0326 | 0.0506 | 0.0719 | 0.0825 | 0.0269 | 0.6614 | 0.9898 |
| als_joint | 0.0297 | 0.0479 | 0.0704 | 0.0770 | 0.0222 | 0.6442 | 0.9912 |
| content | 0.0151 | 0.0227 | 0.0326 | 0.0350 | **0.2016** | 0.3703 | 0.6735 |
| two_tower | 0.0290 | 0.0464 | 0.0696 | 0.0755 | 0.0908 | 0.5338 | 0.9286 |
| rrf_fusion | 0.0397 | 0.0615 | 0.0895 | 0.1010 | 0.0506 | 0.6306 | 0.9762 |
| crossverse_ranker | **0.0461** | **0.0703** | 0.0950 | **0.1135** | 0.1075 | 0.6095 | 0.9310 |
| crossverse_ranker+diversity | 0.0452 | 0.0686 | **0.0966** | 0.1125 | 0.1046 | 0.6415 | 0.9359 |

### within_game (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0180 | 0.0326 | 0.0513 | 0.0480 | 0.0004 | **0.6242** | 0.9998 |
| item_knn | 0.0541 | 0.0927 | 0.1271 | 0.1225 | 0.0994 | 0.5957 | 0.8499 |
| als_single_domain | 0.0498 | 0.0824 | 0.1211 | 0.1115 | 0.0222 | 0.5961 | 0.9842 |
| als_joint | 0.0448 | 0.0784 | 0.1216 | 0.1045 | 0.0107 | 0.5871 | 0.9927 |
| content | 0.0122 | 0.0233 | 0.0435 | 0.0315 | **0.1322** | 0.3013 | 0.6824 |
| two_tower | 0.0446 | 0.0786 | 0.1246 | 0.1075 | 0.0570 | 0.4874 | 0.9434 |
| rrf_fusion | 0.0593 | **0.1046** | **0.1485** | **0.1390** | 0.0403 | 0.5161 | 0.9593 |
| crossverse_ranker | **0.0595** | 0.1003 | 0.1474 | 0.1320 | 0.0580 | 0.5426 | 0.9551 |
| crossverse_ranker+diversity | 0.0561 | 0.0913 | 0.1287 | 0.1200 | 0.0574 | 0.5980 | 0.9571 |

### mixed (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0058 | 0.0092 | 0.0189 | 0.0205 | 0.0004 | **0.7198** | 0.9999 |
| item_knn | 0.0413 | 0.0607 | 0.0816 | 0.1010 | 0.2120 | 0.6750 | 0.7940 |
| als_joint | 0.0246 | 0.0387 | 0.0600 | 0.0675 | 0.0268 | 0.6591 | 0.9934 |
| content | 0.0099 | 0.0160 | 0.0240 | 0.0280 | **0.2180** | 0.3674 | 0.6803 |
| two_tower | 0.0254 | 0.0402 | 0.0651 | 0.0705 | 0.0973 | 0.5635 | 0.9531 |
| rrf_fusion | 0.0352 | 0.0565 | 0.0862 | 0.0940 | 0.0660 | 0.6148 | 0.9773 |
| crossverse_ranker | **0.0415** | **0.0630** | **0.0869** | **0.1075** | 0.1183 | 0.6377 | 0.9425 |
| crossverse_ranker+diversity | 0.0415 | 0.0625 | 0.0852 | 0.1065 | 0.1181 | 0.6406 | 0.9427 |

### cold_start (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0073** | **0.0128** | **0.0204** | **0.0230** | 0.0004 | **0.7197** | 0.9999 |
| content | 0.0003 | 0.0009 | 0.0032 | 0.0015 | 0.0731 | 0.3396 | 0.4316 |
| rrf_fusion | 0.0049 | 0.0089 | 0.0123 | 0.0150 | **0.1207** | 0.4661 | 0.6839 |
| crossverse_ranker | 0.0052 | 0.0104 | 0.0170 | 0.0180 | 0.1043 | 0.4997 | 0.8804 |
| crossverse_ranker+diversity | 0.0042 | 0.0078 | 0.0116 | 0.0145 | 0.1085 | 0.5163 | 0.8690 |

### movie_to_game (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0325 | 0.0408 | 0.0610 | 0.1400 | 0.0002 | 0.6239 | 0.9998 |
| item_knn | 0.0056 | 0.0065 | 0.0083 | 0.0285 | 0.0493 | 0.7532 | 0.5190 |
| als_joint | 0.0217 | 0.0296 | 0.0457 | 0.1040 | 0.0096 | 0.6968 | 0.9951 |
| content | 0.0029 | 0.0040 | 0.0071 | 0.0155 | **0.0544** | 0.5074 | 0.5391 |
| copref | 0.0199 | 0.0254 | 0.0372 | 0.0890 | 0.0242 | 0.7063 | 0.7638 |
| two_tower | 0.0259 | 0.0343 | 0.0568 | 0.1205 | 0.0344 | 0.6789 | 0.9649 |
| rrf_fusion | 0.0200 | 0.0257 | 0.0410 | 0.0955 | 0.0147 | **0.7693** | 0.6874 |
| crossverse_ranker | **0.0353** | **0.0462** | **0.0724** | **0.1640** | 0.0172 | 0.6356 | 0.9921 |
| crossverse_ranker+diversity | 0.0347 | 0.0460 | 0.0685 | 0.1625 | 0.0164 | 0.6870 | 0.9920 |

### game_to_movie (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0212 | 0.0242 | 0.0361 | 0.1110 | 0.0002 | 0.7473 | 0.9998 |
| item_knn | 0.0084 | 0.0109 | 0.0159 | 0.0490 | **0.0949** | **0.7821** | 0.7962 |
| als_joint | 0.0172 | 0.0202 | 0.0298 | 0.0870 | 0.0126 | 0.7261 | 0.9954 |
| content | 0.0029 | 0.0038 | 0.0079 | 0.0200 | 0.0815 | 0.4944 | 0.5474 |
| copref | 0.0175 | 0.0205 | 0.0251 | 0.0845 | 0.0335 | 0.7396 | 0.9411 |
| two_tower | 0.0194 | 0.0222 | 0.0322 | 0.0975 | 0.0426 | 0.6913 | 0.9559 |
| rrf_fusion | 0.0207 | 0.0236 | 0.0333 | 0.1030 | 0.0105 | 0.7469 | 0.9819 |
| crossverse_ranker | **0.0272** | **0.0282** | 0.0418 | **0.1235** | 0.0200 | 0.6951 | 0.9916 |
| crossverse_ranker+diversity | 0.0267 | 0.0274 | **0.0418** | 0.1215 | 0.0194 | 0.7191 | 0.9921 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0042 (n=2000) | 0.0036 (n=678) | 0.0050 (n=596) | 0.0040 (n=726) | 0.0029 (n=1335) |
| game_to_movie | 0.0267 (n=2000) | 0.0501 (n=291) | 0.0314 (n=243) | 0.0213 (n=1466) | 0.0267 (n=2000) |
| mixed | 0.0415 (n=2000) | 0.0386 (n=763) | 0.0389 (n=557) | 0.0467 (n=680) | 0.0415 (n=2000) |
| movie_to_game | 0.0347 (n=2000) | 0.0445 (n=519) | 0.0340 (n=299) | 0.0305 (n=1182) | 0.0347 (n=2000) |
| within_game | 0.0561 (n=2000) | 0.0456 (n=705) | 0.0626 (n=594) | 0.0612 (n=701) | 0.0482 (n=1412) |
| within_movie | 0.0452 (n=2000) | 0.0361 (n=837) | 0.0537 (n=588) | 0.0499 (n=575) | 0.0482 (n=1326) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 16,097 |
| item_knn_z | 12,033 |
| n_sources | 10,849 |
| log_rating_count | 7,140 |
| popularity_score | 6,750 |
| item_knn_score | 6,366 |
| item_knn_rank | 4,376 |
| two_tower_rank | 4,106 |
| semantic_z | 3,949 |
| semantic_score | 3,354 |
| als_rank | 3,275 |
| max_sim_pos | 3,049 |
| year_gap | 3,012 |
| als_z | 2,998 |
| popularity_z | 2,913 |