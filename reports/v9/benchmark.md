# CrossVerse offline benchmark — v9

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
| within_movie | popularity | 3000 | 0.0420 | 0.0086 | +386.8% | [+0.0280, +0.0391] | yes |
| within_game | popularity | 3000 | 0.0562 | 0.0195 | +187.7% | [+0.0297, +0.0437] | yes |
| mixed | popularity | 3000 | 0.0385 | 0.0070 | +446.1% | [+0.0261, +0.0369] | yes |
| cold_start | popularity | 3000 | 0.0048 | 0.0084 | -42.2% | [-0.0063, -0.0007] | yes |
| movie_to_game | popularity | 2505 | 0.0334 | 0.0307 | +9.0% | [-0.0010, +0.0064] | no |
| game_to_movie | popularity | 2505 | 0.0286 | 0.0213 | +34.1% | [+0.0044, +0.0106] | yes |
| within_movie | rrf_fusion | 3000 | 0.0420 | 0.0322 | +30.5% | [+0.0051, +0.0146] | yes |
| within_game | rrf_fusion | 3000 | 0.0562 | 0.0571 | -1.6% | [-0.0061, +0.0042] | no |
| mixed | rrf_fusion | 3000 | 0.0385 | 0.0350 | +10.1% | [-0.0006, +0.0077] | no |
| cold_start | rrf_fusion | 3000 | 0.0048 | 0.0040 | +21.1% | [-0.0005, +0.0022] | no |
| movie_to_game | rrf_fusion | 2505 | 0.0334 | 0.0145 | +130.2% | [+0.0156, +0.0225] | yes |
| game_to_movie | rrf_fusion | 2505 | 0.0286 | 0.0214 | +33.8% | [+0.0042, +0.0102] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0086 | 0.0160 | 0.0247 | 0.0307 | 0.0004 | **0.7504** | 0.9998 |
| item_knn | 0.0338 | 0.0508 | 0.0666 | 0.0773 | **0.2644** | 0.7167 | 0.6564 |
| als_single_domain | 0.0268 | 0.0420 | 0.0614 | 0.0757 | 0.0278 | 0.6782 | 0.9903 |
| als_joint | 0.0268 | 0.0405 | 0.0601 | 0.0757 | 0.0226 | 0.6637 | 0.9916 |
| content | 0.0070 | 0.0113 | 0.0200 | 0.0197 | 0.2493 | 0.3832 | 0.6803 |
| two_tower | 0.0254 | 0.0421 | 0.0599 | 0.0773 | 0.0978 | 0.5637 | 0.9333 |
| rrf_fusion | 0.0322 | 0.0530 | 0.0768 | 0.0963 | 0.0448 | 0.6706 | 0.9855 |
| crossverse_ranker | 0.0419 | 0.0641 | **0.0918** | 0.1063 | 0.1517 | 0.6444 | 0.9121 |
| crossverse_ranker+diversity | **0.0420** | **0.0642** | 0.0915 | **0.1067** | 0.1469 | 0.6774 | 0.9193 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0328 | 0.0515 | 0.0457 | 0.0004 | **0.6769** | 0.9998 |
| item_knn | 0.0525 | 0.0851 | 0.1180 | 0.1157 | 0.1503 | 0.6670 | 0.7442 |
| als_single_domain | 0.0567 | 0.0940 | 0.1361 | 0.1263 | 0.0225 | 0.6577 | 0.9844 |
| als_joint | 0.0477 | 0.0838 | 0.1305 | 0.1107 | 0.0112 | 0.6453 | 0.9926 |
| content | 0.0133 | 0.0237 | 0.0421 | 0.0343 | **0.1559** | 0.3514 | 0.6573 |
| two_tower | 0.0426 | 0.0755 | 0.1196 | 0.1043 | 0.0607 | 0.5570 | 0.9438 |
| rrf_fusion | 0.0571 | 0.1041 | 0.1545 | 0.1420 | 0.0385 | 0.6133 | 0.9691 |
| crossverse_ranker | **0.0604** | **0.1071** | **0.1551** | **0.1447** | 0.0803 | 0.6192 | 0.9421 |
| crossverse_ranker+diversity | 0.0562 | 0.0962 | 0.1418 | 0.1307 | 0.0805 | 0.6725 | 0.9433 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0070 | 0.0110 | 0.0196 | 0.0213 | 0.0003 | **0.7505** | 0.9999 |
| item_knn | 0.0332 | 0.0525 | 0.0676 | 0.0833 | **0.2867** | 0.7250 | 0.7048 |
| als_joint | 0.0268 | 0.0424 | 0.0610 | 0.0740 | 0.0273 | 0.6935 | 0.9936 |
| content | 0.0086 | 0.0129 | 0.0192 | 0.0233 | 0.2648 | 0.3974 | 0.6829 |
| two_tower | 0.0227 | 0.0372 | 0.0572 | 0.0720 | 0.1006 | 0.6061 | 0.9582 |
| rrf_fusion | 0.0350 | 0.0554 | 0.0779 | 0.0977 | 0.0670 | 0.6529 | 0.9825 |
| crossverse_ranker | **0.0386** | **0.0640** | **0.0909** | **0.1057** | 0.1582 | 0.6697 | 0.9292 |
| crossverse_ranker+diversity | 0.0385 | 0.0637 | 0.0888 | 0.1050 | 0.1585 | 0.6720 | 0.9291 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0134** | **0.0219** | **0.0223** | 0.0003 | **0.7506** | 0.9999 |
| content | 0.0003 | 0.0002 | 0.0008 | 0.0017 | 0.0579 | 0.2826 | 0.3990 |
| rrf_fusion | 0.0040 | 0.0065 | 0.0097 | 0.0147 | **0.1093** | 0.5392 | 0.7574 |
| crossverse_ranker | 0.0049 | 0.0082 | 0.0112 | 0.0173 | 0.1002 | 0.5617 | 0.8713 |
| crossverse_ranker+diversity | 0.0048 | 0.0080 | 0.0111 | 0.0170 | 0.1011 | 0.5643 | 0.8701 |

### movie_to_game (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0307 | 0.0384 | 0.0574 | 0.1285 | 0.0002 | 0.6733 | 0.9998 |
| item_knn | 0.0023 | 0.0029 | 0.0044 | 0.0140 | 0.1426 | 0.8046 | 0.3547 |
| als_joint | 0.0206 | 0.0270 | 0.0437 | 0.0922 | 0.0093 | 0.7345 | 0.9953 |
| content | 0.0033 | 0.0050 | 0.0077 | 0.0144 | 0.0653 | 0.5223 | 0.5624 |
| copref | 0.0018 | 0.0021 | 0.0040 | 0.0116 | **0.1622** | 0.7827 | 0.3758 |
| two_tower | 0.0234 | 0.0318 | 0.0537 | 0.1150 | 0.0364 | 0.7086 | 0.9633 |
| rrf_fusion | 0.0145 | 0.0191 | 0.0326 | 0.0731 | 0.0150 | **0.8199** | 0.6378 |
| crossverse_ranker | **0.0358** | **0.0492** | **0.0745** | **0.1653** | 0.0175 | 0.6844 | 0.9924 |
| crossverse_ranker+diversity | 0.0334 | 0.0448 | 0.0693 | 0.1577 | 0.0173 | 0.7223 | 0.9925 |

### game_to_movie (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0213 | 0.0255 | 0.0358 | 0.1182 | 0.0002 | 0.7519 | 0.9998 |
| item_knn | 0.0018 | 0.0017 | 0.0027 | 0.0084 | 0.2326 | **0.8481** | 0.2803 |
| als_joint | 0.0182 | 0.0212 | 0.0321 | 0.0930 | 0.0132 | 0.7318 | 0.9954 |
| content | 0.0033 | 0.0040 | 0.0058 | 0.0188 | 0.1059 | 0.5200 | 0.4935 |
| copref | 0.0008 | 0.0010 | 0.0021 | 0.0044 | **0.2521** | 0.8467 | 0.3105 |
| two_tower | 0.0220 | 0.0267 | 0.0372 | 0.1126 | 0.0439 | 0.7161 | 0.9579 |
| rrf_fusion | 0.0214 | 0.0249 | 0.0364 | 0.1126 | 0.0183 | 0.7635 | 0.9733 |
| crossverse_ranker | **0.0294** | **0.0334** | 0.0507 | **0.1381** | 0.0175 | 0.7129 | 0.9945 |
| crossverse_ranker+diversity | 0.0286 | 0.0322 | **0.0512** | 0.1333 | 0.0174 | 0.7387 | 0.9948 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0048 (n=3000) | 0.0057 (n=990) | 0.0037 (n=916) | 0.0051 (n=1094) | 0.0045 (n=2018) |
| game_to_movie | 0.0286 (n=2505) | 0.0476 (n=379) | 0.0267 (n=332) | 0.0249 (n=1794) | 0.0286 (n=2505) |
| mixed | 0.0385 (n=3000) | 0.0319 (n=1087) | 0.0354 (n=899) | 0.0482 (n=1014) | 0.0385 (n=3000) |
| movie_to_game | 0.0334 (n=2505) | 0.0445 (n=629) | 0.0291 (n=384) | 0.0299 (n=1492) | 0.0334 (n=2505) |
| within_game | 0.0562 (n=3000) | 0.0479 (n=1036) | 0.0508 (n=933) | 0.0695 (n=1031) | 0.0538 (n=2128) |
| within_movie | 0.0420 (n=3000) | 0.0353 (n=1218) | 0.0362 (n=866) | 0.0566 (n=916) | 0.0425 (n=2016) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 30,310 |
| item_knn_z | 17,092 |
| item_knn_score | 13,902 |
| log_rating_count | 12,241 |
| item_knn_rank | 8,853 |
| two_tower_rank | 8,314 |
| max_sim_pos | 7,866 |
| n_sources | 7,196 |
| popularity_z | 7,060 |
| bayes_rating | 6,878 |
| als_z | 6,496 |
| popularity_rank | 5,563 |
| year_gap | 5,545 |
| two_tower_score | 5,454 |
| log_pop | 5,351 |