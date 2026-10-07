# CrossVerse offline benchmark — v11

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
| within_movie | popularity | 3000 | 0.0403 | 0.0086 | +366.7% | [+0.0263, +0.0370] | yes |
| within_game | popularity | 3000 | 0.0544 | 0.0195 | +178.1% | [+0.0280, +0.0417] | yes |
| mixed | popularity | 3000 | 0.0379 | 0.0070 | +437.8% | [+0.0257, +0.0358] | yes |
| cold_start | popularity | 3000 | 0.0044 | 0.0084 | -47.3% | [-0.0068, -0.0011] | yes |
| movie_to_game | popularity | 2505 | 0.0318 | 0.0307 | +3.7% | [-0.0024, +0.0048] | no |
| game_to_movie | popularity | 2505 | 0.0292 | 0.0213 | +36.8% | [+0.0045, +0.0112] | yes |
| within_movie | rrf_fusion | 3000 | 0.0403 | 0.0328 | +22.9% | [+0.0030, +0.0118] | yes |
| within_game | rrf_fusion | 3000 | 0.0544 | 0.0578 | -6.0% | [-0.0084, +0.0013] | no |
| mixed | rrf_fusion | 3000 | 0.0379 | 0.0354 | +7.1% | [-0.0014, +0.0064] | no |
| cold_start | rrf_fusion | 3000 | 0.0044 | 0.0041 | +7.3% | [-0.0011, +0.0017] | no |
| movie_to_game | rrf_fusion | 2505 | 0.0318 | 0.0142 | +124.3% | [+0.0142, +0.0210] | yes |
| game_to_movie | rrf_fusion | 2505 | 0.0292 | 0.0238 | +22.6% | [+0.0026, +0.0081] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0086 | 0.0160 | 0.0247 | 0.0307 | 0.0004 | **0.7504** | 0.9998 |
| item_knn | 0.0359 | 0.0543 | 0.0715 | 0.0870 | 0.2352 | 0.6991 | 0.7703 |
| als_single_domain | 0.0268 | 0.0420 | 0.0614 | 0.0757 | 0.0278 | 0.6782 | 0.9903 |
| als_joint | 0.0268 | 0.0405 | 0.0601 | 0.0757 | 0.0226 | 0.6637 | 0.9916 |
| content | 0.0070 | 0.0113 | 0.0200 | 0.0197 | **0.2493** | 0.3832 | 0.6803 |
| two_tower | 0.0254 | 0.0421 | 0.0599 | 0.0773 | 0.0978 | 0.5637 | 0.9333 |
| rrf_fusion | 0.0328 | 0.0540 | 0.0827 | 0.0940 | 0.0571 | 0.6506 | 0.9782 |
| crossverse_ranker | 0.0402 | 0.0630 | 0.0891 | **0.1060** | 0.1350 | 0.6447 | 0.9218 |
| crossverse_ranker+diversity | **0.0403** | **0.0637** | **0.0903** | 0.1047 | 0.1307 | 0.6748 | 0.9285 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0328 | 0.0515 | 0.0457 | 0.0004 | **0.6769** | 0.9998 |
| item_knn | 0.0562 | 0.0936 | 0.1290 | 0.1263 | 0.1133 | 0.6585 | 0.8523 |
| als_single_domain | 0.0567 | 0.0940 | 0.1361 | 0.1263 | 0.0225 | 0.6577 | 0.9844 |
| als_joint | 0.0477 | 0.0838 | 0.1305 | 0.1107 | 0.0112 | 0.6453 | 0.9926 |
| content | 0.0133 | 0.0237 | 0.0421 | 0.0343 | **0.1559** | 0.3514 | 0.6573 |
| two_tower | 0.0426 | 0.0755 | 0.1196 | 0.1043 | 0.0607 | 0.5570 | 0.9438 |
| rrf_fusion | 0.0578 | **0.1055** | **0.1621** | **0.1427** | 0.0457 | 0.5963 | 0.9579 |
| crossverse_ranker | **0.0584** | 0.1004 | 0.1579 | 0.1350 | 0.0760 | 0.6131 | 0.9467 |
| crossverse_ranker+diversity | 0.0544 | 0.0881 | 0.1354 | 0.1217 | 0.0733 | 0.6677 | 0.9500 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0070 | 0.0110 | 0.0196 | 0.0213 | 0.0003 | **0.7505** | 0.9999 |
| item_knn | 0.0351 | 0.0567 | 0.0733 | 0.0920 | 0.2567 | 0.7118 | 0.8016 |
| als_joint | 0.0268 | 0.0424 | 0.0610 | 0.0740 | 0.0273 | 0.6935 | 0.9936 |
| content | 0.0086 | 0.0129 | 0.0192 | 0.0233 | **0.2648** | 0.3974 | 0.6829 |
| two_tower | 0.0227 | 0.0372 | 0.0572 | 0.0720 | 0.1006 | 0.6061 | 0.9582 |
| rrf_fusion | 0.0354 | 0.0569 | 0.0822 | 0.1003 | 0.0731 | 0.6526 | 0.9794 |
| crossverse_ranker | 0.0377 | 0.0636 | **0.0877** | 0.1050 | 0.1369 | 0.6677 | 0.9427 |
| crossverse_ranker+diversity | **0.0379** | **0.0641** | 0.0866 | **0.1057** | 0.1366 | 0.6712 | 0.9430 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0134** | **0.0219** | **0.0223** | 0.0003 | **0.7506** | 0.9999 |
| content | 0.0003 | 0.0002 | 0.0008 | 0.0017 | 0.0579 | 0.2826 | 0.3990 |
| rrf_fusion | 0.0041 | 0.0069 | 0.0091 | 0.0147 | **0.1252** | 0.5300 | 0.7106 |
| crossverse_ranker | 0.0046 | 0.0075 | 0.0113 | 0.0153 | 0.1124 | 0.5577 | 0.8617 |
| crossverse_ranker+diversity | 0.0044 | 0.0071 | 0.0108 | 0.0153 | 0.1144 | 0.5693 | 0.8572 |

### movie_to_game (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0307 | 0.0384 | 0.0574 | 0.1285 | 0.0002 | 0.6733 | 0.9998 |
| item_knn | 0.0051 | 0.0059 | 0.0091 | 0.0271 | 0.0515 | **0.8208** | 0.5859 |
| als_joint | 0.0206 | 0.0270 | 0.0437 | 0.0922 | 0.0093 | 0.7345 | 0.9953 |
| content | 0.0033 | 0.0050 | 0.0077 | 0.0144 | **0.0653** | 0.5223 | 0.5624 |
| copref | 0.0183 | 0.0236 | 0.0357 | 0.0862 | 0.0243 | 0.7740 | 0.7963 |
| two_tower | 0.0234 | 0.0318 | 0.0537 | 0.1150 | 0.0364 | 0.7086 | 0.9633 |
| rrf_fusion | 0.0142 | 0.0182 | 0.0357 | 0.0683 | 0.0137 | 0.8176 | 0.6373 |
| crossverse_ranker | **0.0322** | **0.0427** | **0.0714** | 0.1505 | 0.0187 | 0.6836 | 0.9917 |
| crossverse_ranker+diversity | 0.0318 | 0.0427 | 0.0659 | **0.1537** | 0.0181 | 0.7216 | 0.9916 |

### game_to_movie (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0213 | 0.0255 | 0.0358 | 0.1182 | 0.0002 | 0.7519 | 0.9998 |
| item_knn | 0.0067 | 0.0079 | 0.0133 | 0.0383 | 0.0982 | **0.8002** | 0.7772 |
| als_joint | 0.0182 | 0.0212 | 0.0321 | 0.0930 | 0.0132 | 0.7318 | 0.9954 |
| content | 0.0033 | 0.0040 | 0.0058 | 0.0188 | **0.1059** | 0.5200 | 0.4935 |
| copref | 0.0159 | 0.0191 | 0.0278 | 0.0874 | 0.0333 | 0.7547 | 0.9335 |
| two_tower | 0.0220 | 0.0267 | 0.0372 | 0.1126 | 0.0439 | 0.7161 | 0.9579 |
| rrf_fusion | 0.0238 | 0.0258 | 0.0342 | 0.1126 | 0.0109 | 0.7624 | 0.9780 |
| crossverse_ranker | **0.0300** | **0.0339** | **0.0466** | **0.1429** | 0.0206 | 0.7082 | 0.9920 |
| crossverse_ranker+diversity | 0.0292 | 0.0323 | 0.0451 | 0.1361 | 0.0199 | 0.7270 | 0.9925 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0044 (n=3000) | 0.0058 (n=990) | 0.0024 (n=916) | 0.0049 (n=1094) | 0.0040 (n=2018) |
| game_to_movie | 0.0292 (n=2505) | 0.0474 (n=379) | 0.0217 (n=332) | 0.0267 (n=1794) | 0.0292 (n=2505) |
| mixed | 0.0379 (n=3000) | 0.0299 (n=1087) | 0.0348 (n=899) | 0.0492 (n=1014) | 0.0379 (n=3000) |
| movie_to_game | 0.0318 (n=2505) | 0.0380 (n=629) | 0.0315 (n=384) | 0.0293 (n=1492) | 0.0318 (n=2505) |
| within_game | 0.0544 (n=3000) | 0.0480 (n=1036) | 0.0472 (n=933) | 0.0672 (n=1031) | 0.0503 (n=2128) |
| within_movie | 0.0403 (n=3000) | 0.0320 (n=1218) | 0.0321 (n=866) | 0.0591 (n=916) | 0.0387 (n=2016) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| item_knn_z | 28,044 |
| two_tower_z | 24,272 |
| two_tower_rank | 13,657 |
| log_rating_count | 13,272 |
| n_sources | 12,420 |
| item_knn_score | 11,856 |
| semantic_z | 9,424 |
| max_sim_pos | 8,122 |
| log_pop | 8,093 |
| bayes_rating | 7,930 |
| popularity_score | 7,806 |
| semantic_score | 7,706 |
| year_gap | 6,760 |
| item_knn_rank | 6,648 |
| als_z | 6,625 |