# CrossVerse offline benchmark — v10

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
| within_movie | popularity | 3000 | 0.0427 | 0.0086 | +394.3% | [+0.0286, +0.0395] | yes |
| within_game | popularity | 3000 | 0.0567 | 0.0195 | +190.1% | [+0.0299, +0.0445] | yes |
| mixed | popularity | 3000 | 0.0363 | 0.0070 | +414.5% | [+0.0240, +0.0343] | yes |
| cold_start | popularity | 3000 | 0.0045 | 0.0084 | -46.8% | [-0.0068, -0.0012] | yes |
| movie_to_game | popularity | 2505 | 0.0334 | 0.0307 | +8.7% | [-0.0008, +0.0061] | no |
| game_to_movie | popularity | 2505 | 0.0281 | 0.0213 | +31.7% | [+0.0036, +0.0103] | yes |
| within_movie | rrf_fusion | 3000 | 0.0427 | 0.0318 | +34.2% | [+0.0060, +0.0158] | yes |
| within_game | rrf_fusion | 3000 | 0.0567 | 0.0557 | +1.8% | [-0.0038, +0.0057] | no |
| mixed | rrf_fusion | 3000 | 0.0363 | 0.0340 | +6.8% | [-0.0020, +0.0066] | no |
| cold_start | rrf_fusion | 3000 | 0.0045 | 0.0040 | +11.5% | [-0.0007, +0.0017] | no |
| movie_to_game | rrf_fusion | 2505 | 0.0334 | 0.0128 | +160.1% | [+0.0171, +0.0240] | yes |
| game_to_movie | rrf_fusion | 2505 | 0.0281 | 0.0236 | +19.2% | [+0.0016, +0.0076] | yes |

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
| rrf_fusion | 0.0318 | 0.0519 | 0.0779 | 0.0947 | 0.0435 | 0.6671 | 0.9859 |
| crossverse_ranker | **0.0431** | **0.0690** | **0.0916** | **0.1120** | 0.1457 | 0.6378 | 0.9177 |
| crossverse_ranker+diversity | 0.0427 | 0.0667 | 0.0887 | 0.1103 | 0.1412 | 0.6720 | 0.9242 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0328 | 0.0515 | 0.0457 | 0.0004 | 0.6769 | 0.9998 |
| item_knn | 0.0562 | 0.0936 | 0.1290 | 0.1263 | 0.1133 | 0.6585 | 0.8523 |
| als_single_domain | 0.0567 | 0.0940 | 0.1361 | 0.1263 | 0.0225 | 0.6577 | 0.9844 |
| als_joint | 0.0477 | 0.0838 | 0.1305 | 0.1107 | 0.0112 | 0.6453 | 0.9926 |
| content | 0.0133 | 0.0237 | 0.0421 | 0.0343 | **0.1559** | 0.3514 | 0.6573 |
| two_tower | 0.0426 | 0.0755 | 0.1196 | 0.1043 | 0.0607 | 0.5570 | 0.9438 |
| rrf_fusion | 0.0557 | **0.1033** | 0.1536 | **0.1410** | 0.0345 | 0.6176 | 0.9654 |
| crossverse_ranker | **0.0596** | 0.1022 | **0.1574** | 0.1353 | 0.0688 | 0.6230 | 0.9550 |
| crossverse_ranker+diversity | 0.0567 | 0.0944 | 0.1411 | 0.1277 | 0.0685 | **0.6785** | 0.9558 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0070 | 0.0110 | 0.0196 | 0.0213 | 0.0003 | **0.7505** | 0.9999 |
| item_knn | 0.0351 | 0.0567 | 0.0733 | 0.0920 | 0.2567 | 0.7118 | 0.8016 |
| als_joint | 0.0268 | 0.0424 | 0.0610 | 0.0740 | 0.0273 | 0.6935 | 0.9936 |
| content | 0.0086 | 0.0129 | 0.0192 | 0.0233 | **0.2648** | 0.3974 | 0.6829 |
| two_tower | 0.0227 | 0.0372 | 0.0572 | 0.0720 | 0.1006 | 0.6061 | 0.9582 |
| rrf_fusion | 0.0340 | 0.0547 | 0.0762 | 0.0980 | 0.0600 | 0.6697 | 0.9861 |
| crossverse_ranker | **0.0364** | **0.0585** | **0.0875** | **0.0983** | 0.1493 | 0.6721 | 0.9367 |
| crossverse_ranker+diversity | 0.0363 | 0.0580 | 0.0836 | 0.0977 | 0.1493 | 0.6756 | 0.9369 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0134** | **0.0219** | **0.0223** | 0.0003 | **0.7506** | 0.9999 |
| content | 0.0003 | 0.0002 | 0.0008 | 0.0017 | 0.0579 | 0.2826 | 0.3990 |
| rrf_fusion | 0.0040 | 0.0065 | 0.0097 | 0.0147 | **0.1093** | 0.5392 | 0.7574 |
| crossverse_ranker | 0.0045 | 0.0075 | 0.0114 | 0.0170 | 0.1001 | 0.5606 | 0.8709 |
| crossverse_ranker+diversity | 0.0045 | 0.0073 | 0.0113 | 0.0170 | 0.1012 | 0.5635 | 0.8696 |

### movie_to_game (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0307 | 0.0384 | 0.0574 | 0.1285 | 0.0002 | 0.6733 | 0.9998 |
| item_knn | 0.0051 | 0.0059 | 0.0091 | 0.0271 | 0.0515 | **0.8208** | 0.5859 |
| als_joint | 0.0206 | 0.0270 | 0.0437 | 0.0922 | 0.0093 | 0.7345 | 0.9953 |
| content | 0.0033 | 0.0050 | 0.0077 | 0.0144 | **0.0653** | 0.5223 | 0.5624 |
| copref | 0.0183 | 0.0236 | 0.0357 | 0.0862 | 0.0243 | 0.7740 | 0.7963 |
| two_tower | 0.0234 | 0.0318 | 0.0537 | 0.1150 | 0.0364 | 0.7086 | 0.9633 |
| rrf_fusion | 0.0128 | 0.0166 | 0.0342 | 0.0627 | 0.0086 | 0.8206 | 0.6395 |
| crossverse_ranker | **0.0352** | **0.0466** | **0.0726** | **0.1597** | 0.0165 | 0.6843 | 0.9937 |
| crossverse_ranker+diversity | 0.0334 | 0.0437 | 0.0696 | 0.1505 | 0.0169 | 0.7292 | 0.9934 |

### game_to_movie (n=2505)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0213 | 0.0255 | 0.0358 | 0.1182 | 0.0002 | 0.7519 | 0.9998 |
| item_knn | 0.0067 | 0.0079 | 0.0133 | 0.0383 | 0.0982 | **0.8002** | 0.7772 |
| als_joint | 0.0182 | 0.0212 | 0.0321 | 0.0930 | 0.0132 | 0.7318 | 0.9954 |
| content | 0.0033 | 0.0040 | 0.0058 | 0.0188 | **0.1059** | 0.5200 | 0.4935 |
| copref | 0.0159 | 0.0191 | 0.0278 | 0.0874 | 0.0333 | 0.7547 | 0.9335 |
| two_tower | 0.0220 | 0.0267 | 0.0372 | 0.1126 | 0.0439 | 0.7161 | 0.9579 |
| rrf_fusion | 0.0236 | 0.0257 | 0.0335 | 0.1126 | 0.0094 | 0.7640 | 0.9714 |
| crossverse_ranker | **0.0283** | **0.0317** | **0.0473** | **0.1369** | 0.0212 | 0.7141 | 0.9937 |
| crossverse_ranker+diversity | 0.0281 | 0.0306 | 0.0464 | 0.1345 | 0.0207 | 0.7359 | 0.9941 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0045 (n=3000) | 0.0062 (n=990) | 0.0037 (n=916) | 0.0036 (n=1094) | 0.0045 (n=2018) |
| game_to_movie | 0.0281 (n=2505) | 0.0408 (n=379) | 0.0260 (n=332) | 0.0258 (n=1794) | 0.0281 (n=2505) |
| mixed | 0.0363 (n=3000) | 0.0308 (n=1087) | 0.0335 (n=899) | 0.0446 (n=1014) | 0.0363 (n=3000) |
| movie_to_game | 0.0334 (n=2505) | 0.0427 (n=629) | 0.0296 (n=384) | 0.0304 (n=1492) | 0.0334 (n=2505) |
| within_game | 0.0567 (n=3000) | 0.0539 (n=1036) | 0.0480 (n=933) | 0.0674 (n=1031) | 0.0561 (n=2128) |
| within_movie | 0.0427 (n=3000) | 0.0340 (n=1218) | 0.0358 (n=866) | 0.0608 (n=916) | 0.0413 (n=2016) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 23,310 |
| item_knn_z | 18,475 |
| item_knn_score | 13,460 |
| item_knn_rank | 9,678 |
| log_rating_count | 9,134 |
| n_sources | 8,135 |
| two_tower_rank | 8,040 |
| popularity_z | 6,366 |
| two_tower_score | 5,673 |
| max_sim_pos | 5,641 |
| bayes_rating | 5,260 |
| als_score | 4,976 |
| als_z | 4,874 |
| popularity_rank | 4,421 |
| year_gap | 4,272 |