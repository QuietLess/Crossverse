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
| v10 | ds4 | v9 + minimum support for co-like pairs (kNN ≥ 2 users, co-preference ≥ 3 bridge users) | identical test users as v9: no task differs significantly (all within ±0.0022). The retrievers themselves improve a lot: co-preference alone 10–20× on movies→games / games→movies (0.0018 → 0.0183, 0.0008 → 0.0159), kNN 2–4× cross-domain and +6–7% within-domain; the ranker had already learned to discount the noisy pairs. Candidate |
| v11 | ds4 | v10 + semantic retriever (sentence embeddings, bge-small) | identical test users as v10: no task differs significantly (−0.0024 … +0.0016). On the judged set it is the first version where high taste helps: 50% good / 27.5% bad in the top 5 at taste 0.8 (v10 at 0.5: 35% / 36%). Candidate |
| v12 | ds5 | v11 recipe on ds5: TMDB/IGDB overviews, genres, keywords and release years for 67% of movie and 72% of game items (~80% of interactions); 369 editions merged by external id | different test users (catalog changed); on shared users no task differs significantly. Lift over popularity: within-movie +504% (v11 +367%), mixed +688% (+438%), cold start −34% (−47%), games→movies +25% (+37%). Judged set at taste 0.8: 59% good / 15% bad (v11: 50% / 27.5%). Candidate |
| v13 | ds6 | v12 recipe on ds6: TMDB/IGDB matches can no longer be works released after the data ends (2023). 63 items moved from a remake to the original (Road House 2024 → 1989, Nosferatu 2024 → 1922), 31 lost their match | catalog nearly unchanged: cross-domain NDCG@10 0.0307 vs 0.0295, movie→game +7%, others within noise. Judged set at taste 0.8 (7 new suggestions rated by Claude, blind): 58.8% good / 12.5% bad (v12 57.5% / 12.5%; v12's numbers shifted slightly since the add-on filter). Candidate |

Each candidate (v4–v6) was fixed before its test results were seen, so the test set was not used
for tuning. v2–v7 were all evaluated with the final serving code (eligibility filter, cold-start
tiers). v1's report predates those rules and is kept for history only. Models trained on different
datasets are never compared by the gate. A dataset migration is promoted explicitly
(`publish.py --force`) after the new baseline's own benchmark has been reviewed.

## Serving settings: taste and compilations

Measured on v10 with `scripts/serving_settings_benchmark.py` (same 17,010 test users for every
setting; per-user rows in `reports/v10/serving_settings_cases.csv.gz`). "Shares a theme" is the
share of recommendations carrying a specific (non-generic) theme of the user's history: it
describes how on-topic the list is, not whether it is good.

| task | NDCG@10 before | no compilations | taste 0.3 | taste 0.5 | taste 1.0 | shares a theme: before → taste 0.5 |
|---|---:|---:|---:|---:|---:|---|
| games → movies | 0.0279 | 0.0270 | 0.0214 | 0.0183 | 0.0066 | 46% → 74% |
| movies → games | 0.0344 | 0.0344 | 0.0234 | 0.0181 | 0.0055 | 54% → 78% |
| within movies | 0.0431 | 0.0418 | 0.0255 | 0.0202 | 0.0084 | 70% → 79% |
| within games | 0.0579 | 0.0579 | 0.0406 | 0.0353 | 0.0175 | 71% → 88% |
| mixed | 0.0344 | 0.0336 | 0.0238 | 0.0201 | 0.0086 | 73% → 85% |

* **Dropping movie compilations is nearly free**: −0.0013 NDCG at most, significant only on cold
  start (−0.0005). It is on by default.
* **Taste trades purchase prediction for topicality.** NDCG measures "will this user buy it on
  Amazon", which rewards bestsellers; taste 0.5 costs 35–55% of it and makes 74–88% of the list
  share a specific theme. Popularity of the picks barely moves (the ≥ 20-fans quality floor), and
  at 1.0 keyword matches take over ("River Monsters" for The Witcher 3). The API default stays 0,
  so every benchmark number in this document is the plain ranker; the demo UI defaults to 0.5.

## Judged evaluation

The offline benchmark asks "will this Amazon user buy it". The judged set asks "does it fit someone
who liked the seed": 16 seeds (8 games → movies, 8 movies → games), top 5 per setting, rated 2 / 1 / 0
(see `docs/judgments/`). Ratings come from the owner (`you`, 21 items) and an LLM judge (`claude`,
all pooled items, blind, rules fixed in `RUBRIC.md` before rating). Agreement on shared items: 62%
identical, 90% within one step; the owner is stricter on same-genre-different-setting picks. Numbers
below use the owner's rating wherever given, otherwise Claude's (`scripts/judged_eval.py`).

| version | taste | good in top 5 | bad in top 5 |
|---|---:|---:|---:|
| v10 | 0 | 12.5% | 61% |
| v10 | 0.5 (old UI default) | 35% | 36% |
| v10 | 1.0 | 26% | 48% |
| v11 | 0 | 16% | 57.5% |
| v11 | 0.5 | 41% | 36% |
| v11 | 0.8 | 50% | 27.5% |
| v11 | 1.0 | 49% | 31% |
| v12 | 0 | 16% | 62.5% |
| v12 | 0.5 | 44% | 21% |
| **v12** | **0.8 (UI default)** | **59%** | **15%** |
| v12 | 1.0 | 55% | 16% |

With TF-IDF similarity (v10) high taste drifts into keyword matches ("River Monsters" for The
Witcher 3); sentence embeddings (v11) keep improving up to 0.8–0.9. External metadata (v12) mostly fixes seeds the
sentence model misread from Amazon product copy (Mass Effect 2 → Battlestar Galactica instead of Iron Man 2).
Caveats: 16 seeds, so ±1 item
moves a cell by ~1 point; most ratings are the LLM's; and v11 trades variety for precision on
franchises (Transformers → five Transformers games) and misreads some seeds (BioShock → Resident
Evil, Pacific Rim instead of v10's V for Vendetta, Children of Men).

With v12 the weakest seed is Blade Runner 2049 → games: sentence-embedding similarities between
movie and game descriptions are nearly flat (std ≈ 0.04), and the best-fitting games are missing
from the Amazon catalog (Observer, Detroit: Become Human) or share only "sci-fi" in their tags
(Deus Ex, Cyberpunk 2077). Subtracting each domain's mean embedding doubles the spread but helped
some seeds (Red Dead Redemption 2 → Unforgiven, Bone Tomahawk) and hurt others (BioShock → Saw,
Scream), so it is not used. Blocking add-ons (expansions, figures, vouchers) in v12 moved taste 0.8
from 59% good / 15% bad to 56% / 14%: some blocked add-ons had been rated as fits.

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
