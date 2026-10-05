# Data

## Sources

| source | role | notes |
|---|---|---|
| UCSD Amazon Reviews 2023: `Movies_and_TV`, `Video_Games` (rating_only 0-core CSVs + item metadata) | primary behavioural data | The same `user_id` appears in both categories, which provides real shared-user cross-domain labels. |
| MovieLens 32M | not used for training | Its user ids do not align with Amazon's, so it cannot provide cross-domain labels. A movie-only benchmark is possible as future work. |
| IGDB API | optional enrichment (not required) | Canonical game genres and themes; needs Twitch credentials. Used only for metadata, never as labels. |

`pipelines/build_dataset.py --download` fetches only the two categories, about 830 MB in total:
ratings 349 + 115 MB and metadata 271 + 103 MB. Raw and processed data are git-ignored. Re-check
each dataset's license before redistributing derived data.

## Dataset versions

| dataset | bridge-user rule | extra filters | interactions | bridge users | movies / games | used by |
|---|---|---|---:|---:|---:|---|
| ds1 `470c4aa5a9ab` | ≥3 liked movies **and** ≥3 liked games | accessories | 739,070 | 36,124 | 18,965 / 7,680 | v1–v3 |
| ds2 `da6e22f7ac88` | ≥1 liked movie and ≥1 liked game | + audit fixes (edition words, ordinal seasons, year-split games, hardware) | 1,624,150 | 120,956 | 31,870 / 10,856 | v4–v6 |
| **ds3 `531c1785037f`** | ≥1 / ≥1 | + **DVDs/CDs misfiled under Video_Games** removed | **1,620,533** | **119,016** | **31,847 / 10,741** | **v7 (production)** |

The IDs are content fingerprints of the processed tables, stored in every model manifest. The
promotion gate refuses to compare models across datasets. Archived copies live in
`data/processed_ds1/` and `data/processed_ds2/`.

## Funnel (ds3)

| step | count |
|---|---:|
| raw ratings (both categories) | 21,714,019 |
| raw users: movies / games | 6,503,429 / 2,766,656 |
| users with any rating in both domains | 674,122 |
| **bridge users**: ≥1 positive (rating ≥4) in *each* domain | 500,389 |
| + sampled single-domain users (20k per domain, ≥5 positives) | 40,000 |
| products with usable metadata | 259,808 |
| canonical items after entity resolution | 169,214 (90,582 product rows merged away) |
| after pruning (items ≥5 positives, users ≥3 positives) | **1,620,533 interactions · 172,184 users · 119,016 bridge users · 31,847 movies · 10,741 games** |

87% of ratings are ≥4, and 95% of items carry at least one theme. User ids are stable pseudonyms
(`u` + 12 hex characters of SHA-1 of the Amazon id), identical across rebuilds.

## Cleaning decisions

* **Positives**: rating ≥4 is a positive implicit event. Ratings ≤2 are explicit dislikes and enter
  user histories with negative weight. The rating itself is kept as an auxiliary feature.
* **Duplicates**: when a user reviews the same product twice, the latest review is kept. When several
  products collapse into one canonical item, the max rating and the earliest timestamp are kept.
* **Missing metadata**: 37% of the wanted Movies_and_TV products are Prime Video stubs with a null
  title and no other fields. They can't be resolved or explained, so they are dropped.
* **Not recommendable**: memberships, gift/points cards and DLC/season passes stay in the catalog (users
  did buy them) but a serving-time eligibility rule never recommends them.
* **Misfiled media**: some sellers list DVDs and CDs under Video_Games, e.g. *Stargate Atlantis: The
  Complete Series* with categories `['Video Games', 'PC', 'Games']`. On ds2 these created most of
  the top "movie ↔ game" affinities. ds3 drops game-domain products whose metadata says media:
  main category Movies & TV / Digital Music / Prime Video, store "Format: DVD/Blu-ray/Audio CD",
  actor credits, `Type of item: Blu-ray/DVD`, or an MPAA rating. `main_category` alone is not
  trusted (*Crash Team Racing* is filed under Books). A few misfiled items with no media
  metadata remain.
* **Non-games**: Video_Games also sells controllers, headsets, consoles and gift cards. Items in an
  accessories/consoles/VR category are dropped. Items with a "Games" category are kept unless the
  title contains an unmistakable hardware word. Items with neither fall back to a title keyword
  heuristic. A handful of edge cases remain (e.g. console bundles that include a game).

## Canonical item resolution (`crossverse.data.canonical`)

Amazon items are products, not entertainment entities: one film appears as DVD, Blu-ray, 4K, box
set and collector's edition, and one game as per-platform editions, GOTY and bundles.

1. Normalise titles: accents, `&`→and, drop bracketed segments, domain-specific
   format/platform/edition/season tokens (platform words such as "Switch" are stripped only from
   games), Roman sequel numerals (II→2), punctuation and leading articles.
2. Group by (domain, normalised title).
3. Groups whose titles carry conflicting explicit years ("Dune (1984)" vs "Dune (2021)") are split
   into year clusters (±1 year tolerance). Items without a year join the largest cluster.
4. `asin_map.parquet` keeps every `source_asin → canonical_item_id` mapping.

TV seasons are deliberately merged into their series, since taste is expressed for the show.

**Audit.** Each build writes a stratified sample (25 groups per domain × merged/singleton) to
`canonical_audit_sample.csv`. Two samples were labelled by Claude from product titles, so treat
them as a careful first pass that is worth a human spot-check. Both are in `docs/audit/`.

| | merged groups correct | singletons correct (strict*) | singletons recommendable |
|---|---:|---:|---:|
| ds2 before fixes: games | 84% | – | 84% |
| ds2 before fixes: movies | 80% | 88% | – |
| **final normaliser: games** | **92%** | 72% | 92% |
| **final normaliser: movies** | **80%** | **96%** | 100% |

\* strict = also no missed merge with another catalog item. The first audit found two bugs, both
fixed: real title words such as "special" and "limited" were stripped as edition noise
("The Midnight Special" became "Midnight"), and ordinal seasons ("The Complete Fifth Season") never
merged into their show. It also found same-titled remakes merged together (Ratchet & Clank 2002
vs 2016); same-titled games released more than 3 years apart are now split by release year.

**Known misses.** Subtitle variants and named editions are not merged ("Oblivion" vs "The Elder
Scrolls IV: Oblivion", "Psychonauts 2: Motherlobe Edition"). Movies that share a title without an
explicit year can over-merge (*Starman* film and TV series). Movie metadata years are DVD release
dates, so they can't be used to split; that needs an external ID source (TMDB/IMDb).

## Themes (`crossverse.features.themes`)

Amazon has no shared genre taxonomy: game categories are mostly platforms. A curated lexicon of
about 50 themes (cyberpunk, fantasy, identity, choices-matter, …) maps text from both domains into
one vocabulary. Curated movie genre labels map with full trust. Description keywords need ≥2
mentions, and single mentions only fill up to three themes. Themes drive cold start, ranker
features and explanations.
