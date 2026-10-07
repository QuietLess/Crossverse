# Rating rubric

Question for every pair: *would someone who liked the seed enjoy this recommendation?*
Judge the work itself (story, genre, tone, setting, franchise), not the catalog's tags or cover.

| rating | meaning | typical cases |
|---:|---|---|
| 2 | good fit | same franchise or adaptation; or shares the seed's core genre **and** its tone or setting (Red Dead Redemption 2 → a gritty western) |
| 1 | okay | shares the broad genre or one strong element but differs in tone or setting; or a broadly liked title in an adjacent genre that many fans of the seed would accept |
| 0 | bad | no meaningful link: different genre and tone, a kids' title for a mature seed, documentaries, workout or reality TV, sports games for a story-driven seed |
| −1 | don't know | the rater doesn't know the title well enough; stored but never scored |

Raters

* `you`: the project owner, in the UI's ⭐ Rate tab; only titles they know.
* `claude`: LLM-as-judge (Claude), following this rubric, with a one-line reason per rating. Rated
  blind: the suggestions of all settings were merged and shuffled before rating, without the
  setting that produced them. The rubric was committed before any rating was made.

Claude also built the systems being compared, so its ratings may lean towards them. The check is
`agreement()` in `src/crossverse/evaluation/judgments.py`: on items both raters know, how often
`you` and `claude` give the same rating. `scripts/judged_eval.py` prints it; reports state which
rater's ratings they use (`--rater you | claude | combined`).
