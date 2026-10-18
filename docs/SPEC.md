# Product spec

Written after the fact: it describes what CrossVerse does today. Details of the data, model and
evaluation are in [data.md](data.md), [architecture.md](architecture.md) and [evaluation.md](evaluation.md).

## Goal

Answer "if you liked this movie, what game should you play next, and vice versa?" from real
shared-user behaviour, with reasons that come from evidence (co-likes, themes, story similarity)
rather than generated text.

## Users

* **Visitors of the public app** (https://crossverse-app.web.app): people who like films, series or
  games and want a pick in the other medium. No account, nothing stored per person.
* **The owner** as researcher and portfolio author: benchmarks, judged ratings, model promotion.

## Main journeys

1. **Cross-medium pick.** Search and add a few liked titles (and optionally dislikes), choose Games,
   Films, TV or Everything, and get a ranked list with a one-line reason per item.
2. **Adjust the feel.** Move the slider between Balanced (fan favourites that share the feel) and Same
   DNA (closest story, setting and themes).
3. **Start from a mood.** With no titles, pick moods or type free text ("dark sci-fi about AI").
4. **Explore around a pick.** Open an item: similar titles in both media, "Same universe" adaptations
   and tie-ins, and newer titles ("New releases") that the review data doesn't contain.
5. **Import taste.** Load a Letterboxd export; 4★+ films become favourites, ≤1.5★ dislikes. The file
   is read in the browser and only names, years and stars are sent.
6. **Share.** The picks live in the URL.
7. **API.** The same functions over HTTP (`/recommend`, `/similar`, `/items/search`, ...).

## Acceptance criteria (current behaviour, covered by `tests/`)

* Movie→game and game→movie requests return only items of the target medium; film/tv targets return
  only that kind.
* Unknown titles come back in `unresolved` with "did you mean" suggestions; a domain request where
  nothing resolves is a 422 with suggestions.
* A title that exists as both a movie and a game resolves to one and lists the other in `ambiguous`.
* Every recommended item carries an evidence object; the summary only states signals above threshold.
* No memberships, gift cards, DLC, hardware or compilations in recommendations.
* Letterboxd matching is strict: same title, year within one; near-misses are reported as unmatched.
* Requests are bounded: ≤100 liked and ≤100 disliked titles, titles ≤300 characters, feedback context
  ≤4 KB; larger requests get a 422.
* A model is only promoted if cross-domain NDCG@10 doesn't drop beyond tolerance and no per-task
  guardrail regresses, and never against a model trained on a different dataset.

## Out of scope (for now)

Accounts and saved profiles, commercial use of the data, watch/play logs (Amazon reviews are
purchases plus reviews), titles after September 2023 inside the trained model (they are shown as new
releases, matched by story and franchise only).
