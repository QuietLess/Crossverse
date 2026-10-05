# CrossVerse offline benchmark — v2

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
| within_movie | popularity | 2000 | 0.0448 | 0.0093 | +383.0% | [+0.0285, +0.0427] | yes |
| within_game | popularity | 2000 | 0.0627 | 0.0240 | +161.3% | [+0.0303, +0.0473] | yes |
| mixed | popularity | 2000 | 0.0402 | 0.0126 | +218.5% | [+0.0208, +0.0344] | yes |
| cold_start | popularity | 2000 | 0.0056 | 0.0110 | -49.2% | [-0.0092, -0.0017] | yes |
| movie_to_game | popularity | 1609 | 0.0347 | 0.0340 | +2.1% | [-0.0034, +0.0048] | no |
| game_to_movie | popularity | 1609 | 0.0269 | 0.0194 | +38.5% | [+0.0039, +0.0111] | yes |
| within_movie | rrf_fusion | 2000 | 0.0448 | 0.0343 | +30.7% | [+0.0048, +0.0162] | yes |
| within_game | rrf_fusion | 2000 | 0.0627 | 0.0651 | -3.7% | [-0.0085, +0.0041] | no |
| mixed | rrf_fusion | 2000 | 0.0402 | 0.0361 | +11.4% | [-0.0006, +0.0090] | no |
| cold_start | rrf_fusion | 2000 | 0.0056 | 0.0042 | +33.7% | [+0.0003, +0.0025] | yes |
| movie_to_game | rrf_fusion | 1609 | 0.0347 | 0.0232 | +49.7% | [+0.0068, +0.0162] | yes |
| game_to_movie | rrf_fusion | 1609 | 0.0269 | 0.0225 | +19.7% | [+0.0011, +0.0077] | yes |

## Results (all users)

### within_movie (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0093 | 0.0162 | 0.0244 | 0.0325 | 0.0005 | **0.7414** | 0.9994 |
| item_knn | 0.0392 | 0.0522 | 0.0666 | 0.0835 | **0.2820** | 0.7125 | 0.6643 |
| als_single_domain | 0.0286 | 0.0460 | 0.0687 | 0.0820 | 0.0447 | 0.6716 | 0.9804 |
| als_joint | 0.0266 | 0.0434 | 0.0621 | 0.0775 | 0.0351 | 0.6570 | 0.9843 |
| content | 0.0107 | 0.0178 | 0.0270 | 0.0320 | 0.2693 | 0.3763 | 0.6431 |
| two_tower | 0.0223 | 0.0374 | 0.0595 | 0.0635 | 0.1131 | 0.5605 | 0.8942 |
| rrf_fusion | 0.0343 | 0.0555 | 0.0786 | 0.0965 | 0.0614 | 0.6393 | 0.9726 |
| crossverse_ranker | **0.0451** | 0.0662 | **0.0897** | **0.1130** | 0.1641 | 0.6317 | 0.8849 |
| crossverse_ranker+diversity | 0.0448 | **0.0668** | 0.0851 | 0.1110 | 0.1604 | 0.6683 | 0.8919 |

### within_game (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0240 | 0.0388 | 0.0631 | 0.0590 | 0.0005 | 0.6310 | 0.9998 |
| item_knn | 0.0561 | 0.0948 | 0.1268 | 0.1285 | 0.1734 | 0.6621 | 0.7673 |
| als_single_domain | 0.0576 | 0.0981 | 0.1436 | 0.1435 | 0.0373 | 0.6543 | 0.9811 |
| als_joint | 0.0570 | 0.0953 | 0.1419 | 0.1360 | 0.0224 | 0.6482 | 0.9901 |
| content | 0.0144 | 0.0256 | 0.0445 | 0.0390 | **0.1744** | 0.3524 | 0.6535 |
| two_tower | 0.0461 | 0.0841 | 0.1273 | 0.1210 | 0.0739 | 0.5610 | 0.9386 |
| rrf_fusion | 0.0651 | 0.1112 | **0.1636** | **0.1605** | 0.0540 | 0.6049 | 0.9697 |
| crossverse_ranker | **0.0657** | **0.1136** | 0.1635 | 0.1540 | 0.0918 | 0.6183 | 0.9424 |
| crossverse_ranker+diversity | 0.0627 | 0.1056 | 0.1451 | 0.1425 | 0.0913 | **0.6830** | 0.9452 |

### mixed (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0126 | 0.0183 | 0.0264 | 0.0380 | 0.0006 | **0.7269** | 0.9998 |
| item_knn | 0.0383 | 0.0506 | 0.0646 | 0.0950 | **0.2927** | 0.7160 | 0.7329 |
| als_joint | 0.0271 | 0.0416 | 0.0627 | 0.0830 | 0.0419 | 0.6873 | 0.9906 |
| content | 0.0066 | 0.0108 | 0.0183 | 0.0255 | 0.2674 | 0.3891 | 0.6650 |
| two_tower | 0.0210 | 0.0336 | 0.0543 | 0.0710 | 0.1192 | 0.6063 | 0.9419 |
| rrf_fusion | 0.0361 | 0.0572 | 0.0823 | 0.1130 | 0.0867 | 0.6416 | 0.9759 |
| crossverse_ranker | **0.0404** | **0.0586** | **0.0852** | **0.1180** | 0.1669 | 0.6673 | 0.9194 |
| crossverse_ranker+diversity | 0.0402 | 0.0577 | 0.0827 | 0.1175 | 0.1672 | 0.6717 | 0.9195 |

### cold_start (n=2000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0110** | **0.0172** | **0.0303** | **0.0330** | 0.0005 | **0.7267** | 0.9998 |
| content | 0.0003 | 0.0008 | 0.0016 | 0.0010 | 0.0599 | 0.2803 | 0.3811 |
| rrf_fusion | 0.0042 | 0.0078 | 0.0119 | 0.0195 | **0.1073** | 0.5511 | 0.8068 |
| crossverse_ranker | 0.0055 | 0.0096 | 0.0179 | 0.0235 | 0.0968 | 0.5759 | 0.9032 |
| crossverse_ranker+diversity | 0.0056 | 0.0090 | 0.0165 | 0.0230 | 0.0984 | 0.5825 | 0.9009 |

### movie_to_game (n=1609)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0340 | 0.0426 | 0.0569 | 0.1603 | 0.0004 | 0.6310 | 0.9998 |
| item_knn | 0.0048 | 0.0052 | 0.0069 | 0.0186 | 0.1585 | **0.7941** | 0.3831 |
| als_joint | 0.0210 | 0.0260 | 0.0413 | 0.1050 | 0.0187 | 0.7301 | 0.9934 |
| content | 0.0054 | 0.0069 | 0.0100 | 0.0267 | 0.0664 | 0.5099 | 0.5567 |
| copref | 0.0034 | 0.0047 | 0.0077 | 0.0199 | **0.1682** | 0.7845 | 0.3974 |
| two_tower | 0.0222 | 0.0287 | 0.0515 | 0.1187 | 0.0430 | 0.7153 | 0.9625 |
| rrf_fusion | 0.0232 | 0.0292 | 0.0399 | 0.1119 | 0.0333 | 0.7820 | 0.7548 |
| crossverse_ranker | **0.0383** | **0.0485** | **0.0712** | **0.1840** | 0.0196 | 0.6913 | 0.9933 |
| crossverse_ranker+diversity | 0.0347 | 0.0426 | 0.0663 | 0.1722 | 0.0193 | 0.7204 | 0.9925 |

### game_to_movie (n=1609)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0194 | 0.0246 | 0.0361 | 0.1237 | 0.0004 | 0.7427 | 0.9994 |
| item_knn | 0.0029 | 0.0034 | 0.0048 | 0.0180 | 0.2362 | **0.8304** | 0.3152 |
| als_joint | 0.0191 | 0.0219 | 0.0305 | 0.1181 | 0.0220 | 0.7153 | 0.9893 |
| content | 0.0044 | 0.0056 | 0.0066 | 0.0267 | 0.1026 | 0.5068 | 0.5009 |
| copref | 0.0012 | 0.0018 | 0.0029 | 0.0099 | **0.2495** | 0.8302 | 0.2895 |
| two_tower | 0.0180 | 0.0214 | 0.0312 | 0.1119 | 0.0500 | 0.7022 | 0.9381 |
| rrf_fusion | 0.0225 | 0.0247 | 0.0351 | 0.1274 | 0.0289 | 0.7348 | 0.9626 |
| crossverse_ranker | **0.0274** | **0.0304** | 0.0426 | **0.1523** | 0.0249 | 0.6913 | 0.9896 |
| crossverse_ranker+diversity | 0.0269 | 0.0296 | **0.0431** | 0.1516 | 0.0253 | 0.7180 | 0.9892 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0056 (n=2000) | 0.0063 (n=498) | 0.0066 (n=917) | 0.0034 (n=585) | 0.0044 (n=1335) |
| game_to_movie | 0.0269 (n=1609) | 0.0400 (n=161) | 0.0329 (n=381) | 0.0227 (n=1067) | 0.0269 (n=1609) |
| mixed | 0.0402 (n=2000) | 0.0342 (n=606) | 0.0393 (n=935) | 0.0497 (n=459) | 0.0402 (n=2000) |
| movie_to_game | 0.0347 (n=1609) | 0.0375 (n=348) | 0.0341 (n=419) | 0.0339 (n=842) | 0.0347 (n=1609) |
| within_game | 0.0627 (n=2000) | 0.0514 (n=529) | 0.0571 (n=906) | 0.0822 (n=565) | 0.0554 (n=1369) |
| within_movie | 0.0448 (n=2000) | 0.0320 (n=641) | 0.0450 (n=922) | 0.0633 (n=437) | 0.0474 (n=1448) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 20,484 |
| n_sources | 13,219 |
| item_knn_score | 11,029 |
| item_knn_z | 10,921 |
| log_rating_count | 9,263 |
| als_z | 8,826 |
| item_knn_rank | 8,175 |
| popularity_score | 7,844 |
| max_sim_pos | 7,312 |
| bayes_rating | 6,978 |
| two_tower_rank | 6,447 |
| two_tower_score | 6,145 |
| popularity_z | 5,989 |
| content_score | 5,346 |
| year_gap | 5,315 |