# Evaluation protocol

The question that matters is not "can it recommend another movie?" but "can one domain help predict
the other?". The protocol is built so a positive answer cannot come from leakage.

## Split (`crossverse.evaluation.protocol.make_split`)

1. **Cross-domain holdout users.** From bridge users with ≥2 positives in each domain, four disjoint
   random groups (6% each by default; 2,504 users per group on ds3) are drawn: `movie_to_game_{val,test}` and
   `game_to_movie_{val,test}`. For a movie→game user **every game interaction is removed from
   training**. The model never sees a single game this user touched, so the only route to their
   game taste is their movies.
2. **Regular users.** Each user's interactions are sorted by time. The last 20% are test, the
   preceding 10% validation, the rest train. There is no random row splitting.
3. `assert_no_leakage` checks that holdout users have no hidden-domain rows in train and that no
   target item appears in any input history. The test suite enforces both.

## Tasks

| task | input | target | candidates |
|---|---|---|---|
| within_movie / within_game | the user's same-domain train(+val) history | future same-domain positives | that domain |
| movie_to_game | holdout user's movies only | all their (hidden) game positives | games |
| game_to_movie | holdout user's games only | all their (hidden) movie positives | movies |
| mixed | the full history from both domains | future positives in either domain | both |
| cold_start | **no history**, only the top-3 themes of the user's past likes | future positives | both |

Validation tasks (built the same way) train the LambdaRank ranker. Test tasks are used only by
`pipelines/evaluate.py`.

## Promotion gate

`pipelines/publish.py` promotes a candidate only if (1) `cross_domain_ndcg@10` drops by no more than
2% (relative) versus production, and (2) no other task's NDCG@10 drops by more than 10% (guardrails).
Every lift in `reports/<version>/benchmark.md` comes with a paired bootstrap CI on the same users.

## Metrics

Recall@K, NDCG@K, HitRate@K (K = 10, 20, 50) and MAP@10, plus catalog coverage@10, intra-list
diversity@10 (1 − mean pairwise cosine), novelty@10 (self-information) and pop_pct@10 (mean
popularity percentile, a popularity-bias diagnostic). Results are reported for all users and by
segment: heavy (top-quartile history length), medium, sparse (≤5 interactions) and bridge.

## Baselines

popularity, item-kNN, single-domain ALS (within-domain tasks only), joint ALS, content-only, the
co-preference transfer model (cross-domain tasks only), the two-tower model, reciprocal-rank fusion
of all retrievers without the ranker, the full ranker, and ranker + post-ranking rules.

## Hyper-parameter choices (validation tasks only, 800 cases each, dataset 1)

| knob | grid | movie→game NDCG@10 | game→movie NDCG@10 | standalone optimum |
|---|---|---|---|---|
| co-preference target-popularity exponent α | 0 / 0.25 / 0.5 / 0.75 | **0.0245** / 0.0112 / 0.0023 / 0.0013 | **0.0187** / 0.0067 / 0.0003 / 0.0001 | 0.0 |
| item-kNN shrinkage | 10 / 100 / 500 | 0.0028 / 0.0130 / **0.0185** | 0.0011 / 0.0108 / **0.0172** | 500 |
| cold-start theme boost on popularity | 0 / 0.5 / 1 / 2 / 4 | cold start: **0.0112** / 0.0106 / 0.0095 / 0.0072 / 0.0066 | | 0 (0.5 used, see below) |

**Standalone optima are not pipeline optima.** α = 0 and shrinkage 500 make each retriever 4–7×
better *on its own*, yet the full pipeline got worse every time they were used (v3 on dataset 1,
v6 on dataset 2). Without popularity normalisation, those retrievers nominate the same popular
items the others already find, so the ranker loses the complementary niche candidates it was
using. The production defaults therefore stay at α = 0.5 and shrinkage 10.

Cold start: on held-out users, the themes of their past likes do **not** predict their next
purchases better than popularity. The product still applies a mild boost and explicit-intent tiers
(cold-start users who ask for "cyberpunk" get cyberpunk titles first). On dataset 3 that is a
statistical tie with popularity (−1%, n.s.).

## Model versions

| version | dataset | change | outcome |
|---|---|---|---|
| v1 | ds1 | first full pipeline | ranker early-stopped at 18 trees: validation queries were ordered by task, so the early-stopping set was all game→movie |
| v2 | ds1 | shuffle LambdaRank validation queries across tasks; two-tower 8 → 15 epochs | 122 trees; small gains on most tasks; production |
| v3 | ds1 | validation-tuned co-preference α and kNN shrinkage | cross-domain tie (0.0310 vs 0.0308), mixed −12.8%, cold start −13.9% → **blocked by guardrail** |
| v4 | ds2 | v2 recipe on 3.4× more bridge users | movies→games becomes **significant** (+16% vs popularity); promoted as a dataset migration |
| v5 | ds2 | + routed cross-domain specialist ranker, 2× ranker queries | cross-domain 0.0287 vs 0.0309: the specialist early-stops after ~11 trees → **blocked by gate** |
| v6 | ds2 | + tuned retriever knobs (as in v3) | cross-domain 0.0264 vs 0.0309 → **blocked by gate** |
| **v7** | ds3 | v4 recipe after removing misfiled DVDs from the game catalog | all behavioural tasks significantly above popularity; cold start ties; **production** |
| v9 | ds4 | v7 recipe on ds4: edition grouping fixes, theme-tagging fixes, cover images | within-domain/mixed −7–10% and games→movies +7% in aggregate, but only ~7% of test users overlap with v7; on shared users no task differs significantly. Movies→games no longer significant (+9%); cold start −42% vs popularity (significant). **Not promoted** (candidate) |
| v9a | ds4′ | ablation: v9 with the old theme rules, identical test users | within noise of v9 on every task (cold start −49%); the theme rules are neutral, the cold-start drop comes with ds4 |

Each candidate (v4–v6) was fixed before its test results were seen, so the test set was not used
for tuning. v2–v7 were all evaluated with the final serving code (eligibility filter, cold-start
tiers). v1's report predates those rules and is kept for history only. Models trained on different
datasets are never compared by the gate. A dataset migration is promoted explicitly
(`publish.py --force`) after the new baseline's own benchmark has been reviewed.

## Qualitative analysis

`reports/error_analysis.md` lists 50 random movie→game and 50 game→movie holdout users with their
liked inputs, the top 10 recommendations (✅ = liked later), and what they actually liked.

## Known limitations

* Amazon ratings are purchases plus reviews, not consumption logs. 87% of ratings are ≥4, so
  "positive" mostly means "bought and reviewed". Explicit dislikes are rare.
* Popular titles dominate held-out positives, so popularity is a strong baseline on Amazon data.
  Lifts are reported against it rather than in isolation.
* Canonicalisation merges formats and seasons but not every edition or subtitle variant (e.g.
  "Witcher 3: Wild Hunt" vs. "The Witcher 3 GOTY"). See `docs/data.md`.
* Changing the catalog (ds3 → ds4) reshuffles which users are sampled for the test tasks, so
  aggregate numbers across datasets are different exams. Compare on shared users
  (`reports/<v>/benchmark_cases.csv.gz`) or on a controlled ablation instead.
* The cold-start task derives each user's stated preferences from the first three themes of their
  liked items, so changing the theme rules also changes the test questions.
