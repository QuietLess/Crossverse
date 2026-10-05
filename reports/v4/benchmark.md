# CrossVerse offline benchmark — v4

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
| within_movie | popularity | 3000 | 0.0368 | 0.0094 | +289.9% | [+0.0222, +0.0328] | yes |
| within_game | popularity | 3000 | 0.0665 | 0.0195 | +242.1% | [+0.0394, +0.0547] | yes |
| mixed | popularity | 3000 | 0.0392 | 0.0091 | +329.3% | [+0.0249, +0.0355] | yes |
| cold_start | popularity | 3000 | 0.0043 | 0.0084 | -48.1% | [-0.0065, -0.0015] | yes |
| movie_to_game | popularity | 2518 | 0.0339 | 0.0293 | +15.5% | [+0.0007, +0.0083] | yes |
| game_to_movie | popularity | 2518 | 0.0279 | 0.0203 | +37.4% | [+0.0046, +0.0108] | yes |
| within_movie | rrf_fusion | 3000 | 0.0368 | 0.0305 | +20.7% | [+0.0021, +0.0108] | yes |
| within_game | rrf_fusion | 3000 | 0.0665 | 0.0617 | +7.9% | [-0.0007, +0.0099] | no |
| mixed | rrf_fusion | 3000 | 0.0392 | 0.0352 | +11.2% | [+0.0003, +0.0077] | yes |
| cold_start | rrf_fusion | 3000 | 0.0043 | 0.0033 | +32.7% | [+0.0001, +0.0020] | yes |
| movie_to_game | rrf_fusion | 2518 | 0.0339 | 0.0180 | +88.3% | [+0.0122, +0.0198] | yes |
| game_to_movie | rrf_fusion | 2518 | 0.0279 | 0.0205 | +36.3% | [+0.0048, +0.0102] | yes |

## Results (all users)

### within_movie (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0094 | 0.0164 | 0.0273 | 0.0293 | 0.0004 | 0.6941 | 0.9998 |
| item_knn | 0.0345 | 0.0474 | 0.0626 | 0.0803 | **0.2642** | **0.7088** | 0.6455 |
| als_single_domain | 0.0260 | 0.0380 | 0.0569 | 0.0687 | 0.0279 | 0.6468 | 0.9901 |
| als_joint | 0.0258 | 0.0403 | 0.0606 | 0.0717 | 0.0227 | 0.6329 | 0.9914 |
| content | 0.0075 | 0.0125 | 0.0206 | 0.0250 | 0.2372 | 0.3611 | 0.6980 |
| two_tower | 0.0198 | 0.0308 | 0.0496 | 0.0577 | 0.0983 | 0.5544 | 0.9325 |
| rrf_fusion | 0.0305 | 0.0494 | 0.0747 | 0.0843 | 0.0480 | 0.6210 | 0.9822 |
| crossverse_ranker | 0.0365 | 0.0536 | 0.0819 | 0.0927 | 0.1378 | 0.6152 | 0.9240 |
| crossverse_ranker+diversity | **0.0368** | **0.0537** | **0.0820** | **0.0953** | 0.1315 | 0.6487 | 0.9306 |

### within_game (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0195 | 0.0338 | 0.0514 | 0.0510 | 0.0004 | 0.6065 | 0.9998 |
| item_knn | 0.0572 | 0.0876 | 0.1229 | 0.1197 | **0.1497** | **0.6522** | 0.7465 |
| als_single_domain | 0.0566 | 0.0936 | 0.1385 | 0.1277 | 0.0225 | 0.6382 | 0.9845 |
| als_joint | 0.0509 | 0.0839 | 0.1295 | 0.1153 | 0.0112 | 0.6264 | 0.9928 |
| content | 0.0161 | 0.0286 | 0.0453 | 0.0430 | 0.1493 | 0.3339 | 0.6592 |
| two_tower | 0.0459 | 0.0790 | 0.1196 | 0.1130 | 0.0613 | 0.5355 | 0.9423 |
| rrf_fusion | 0.0617 | 0.1075 | 0.1565 | 0.1483 | 0.0365 | 0.6054 | 0.9740 |
| crossverse_ranker | **0.0701** | **0.1154** | **0.1682** | **0.1560** | 0.0715 | 0.5948 | 0.9527 |
| crossverse_ranker+diversity | 0.0665 | 0.1057 | 0.1511 | 0.1440 | 0.0713 | 0.6497 | 0.9539 |

### mixed (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0091 | 0.0145 | 0.0273 | 0.0290 | 0.0003 | **0.7348** | 0.9999 |
| item_knn | 0.0355 | 0.0502 | 0.0652 | 0.0850 | **0.2883** | 0.7150 | 0.6917 |
| als_joint | 0.0271 | 0.0401 | 0.0588 | 0.0757 | 0.0283 | 0.6706 | 0.9936 |
| content | 0.0075 | 0.0123 | 0.0216 | 0.0263 | 0.2514 | 0.3743 | 0.6918 |
| two_tower | 0.0220 | 0.0348 | 0.0584 | 0.0653 | 0.1037 | 0.5860 | 0.9565 |
| rrf_fusion | 0.0352 | 0.0569 | 0.0836 | 0.0993 | 0.0668 | 0.6278 | 0.9827 |
| crossverse_ranker | **0.0392** | **0.0587** | **0.0876** | 0.1007 | 0.1534 | 0.6468 | 0.9336 |
| crossverse_ranker+diversity | 0.0392 | 0.0586 | 0.0854 | **0.1010** | 0.1533 | 0.6500 | 0.9339 |

### cold_start (n=3000)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | **0.0084** | **0.0151** | **0.0245** | **0.0263** | 0.0003 | **0.7353** | 0.9999 |
| content | 0.0002 | 0.0006 | 0.0009 | 0.0010 | 0.0413 | 0.2453 | 0.4022 |
| rrf_fusion | 0.0033 | 0.0054 | 0.0087 | 0.0113 | **0.0771** | 0.5443 | 0.8077 |
| crossverse_ranker | 0.0044 | 0.0068 | 0.0116 | 0.0140 | 0.0699 | 0.5474 | 0.9081 |
| crossverse_ranker+diversity | 0.0043 | 0.0068 | 0.0109 | 0.0137 | 0.0706 | 0.5490 | 0.9072 |

### movie_to_game (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0293 | 0.0390 | 0.0563 | 0.1322 | 0.0002 | 0.6002 | 0.9998 |
| item_knn | 0.0037 | 0.0040 | 0.0066 | 0.0131 | 0.1433 | 0.7980 | 0.3780 |
| als_joint | 0.0201 | 0.0264 | 0.0408 | 0.0902 | 0.0097 | 0.7221 | 0.9949 |
| content | 0.0037 | 0.0056 | 0.0095 | 0.0191 | 0.0613 | 0.5063 | 0.5614 |
| copref | 0.0036 | 0.0039 | 0.0055 | 0.0119 | **0.1616** | 0.7782 | 0.3786 |
| two_tower | 0.0255 | 0.0347 | 0.0556 | 0.1140 | 0.0360 | 0.7158 | 0.9611 |
| rrf_fusion | 0.0180 | 0.0232 | 0.0373 | 0.0814 | 0.0169 | **0.8094** | 0.7771 |
| crossverse_ranker | **0.0360** | **0.0466** | **0.0733** | **0.1553** | 0.0165 | 0.6878 | 0.9929 |
| crossverse_ranker+diversity | 0.0339 | 0.0427 | 0.0656 | 0.1446 | 0.0162 | 0.7338 | 0.9924 |

### game_to_movie (n=2518)

| model | ndcg@10 | recall@10 | recall@20 | hit@10 | coverage@10 | diversity@10 | pop_pct@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0203 | 0.0250 | 0.0377 | 0.1144 | 0.0002 | 0.6937 | 0.9998 |
| item_knn | 0.0015 | 0.0016 | 0.0027 | 0.0111 | 0.2286 | 0.8400 | 0.2820 |
| als_joint | 0.0161 | 0.0193 | 0.0290 | 0.0886 | 0.0128 | 0.7033 | 0.9953 |
| content | 0.0018 | 0.0027 | 0.0041 | 0.0131 | 0.1017 | 0.5010 | 0.5049 |
| copref | 0.0008 | 0.0008 | 0.0010 | 0.0064 | **0.2452** | **0.8435** | 0.3056 |
| two_tower | 0.0202 | 0.0259 | 0.0371 | 0.1041 | 0.0453 | 0.6857 | 0.9589 |
| rrf_fusion | 0.0205 | 0.0224 | 0.0315 | 0.1044 | 0.0232 | 0.7506 | 0.8946 |
| crossverse_ranker | **0.0280** | **0.0331** | **0.0485** | 0.1406 | 0.0203 | 0.6776 | 0.9939 |
| crossverse_ranker+diversity | 0.0279 | 0.0328 | 0.0460 | **0.1410** | 0.0205 | 0.7089 | 0.9938 |

## Full pipeline by user segment (NDCG@10)

| task | all | heavy | medium | sparse | bridge |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.0043 (n=3000) | 0.0045 (n=996) | 0.0058 (n=889) | 0.0031 (n=1115) | 0.0046 (n=2041) |
| game_to_movie | 0.0279 (n=2518) | 0.0430 (n=334) | 0.0267 (n=299) | 0.0254 (n=1885) | 0.0279 (n=2518) |
| mixed | 0.0392 (n=3000) | 0.0346 (n=1113) | 0.0423 (n=889) | 0.0414 (n=998) | 0.0392 (n=3000) |
| movie_to_game | 0.0339 (n=2518) | 0.0402 (n=672) | 0.0437 (n=330) | 0.0289 (n=1516) | 0.0339 (n=2518) |
| within_game | 0.0665 (n=3000) | 0.0481 (n=1103) | 0.0651 (n=881) | 0.0878 (n=1016) | 0.0622 (n=2112) |
| within_movie | 0.0368 (n=3000) | 0.0337 (n=1207) | 0.0376 (n=895) | 0.0402 (n=898) | 0.0375 (n=2064) |

## Ranker feature importance (gain, top 15)

| feature | gain |
|---|---:|
| two_tower_z | 23,541 |
| item_knn_z | 10,282 |
| item_knn_score | 9,257 |
| log_rating_count | 8,657 |
| n_sources | 8,260 |
| item_knn_rank | 6,294 |
| popularity_z | 5,615 |
| two_tower_rank | 5,489 |
| log_pop | 4,793 |
| popularity_rank | 4,752 |
| als_z | 4,242 |
| two_tower_score | 4,135 |
| max_sim_pos | 3,634 |
| popularity_score | 3,136 |
| content_score | 3,096 |