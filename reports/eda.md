# Data audit (M1)

## Funnel

| metric | value |
|---|---:|
| raw_interactions | 21,714,019 |
| users_in_both_domains | 674,122 |
| bridge_users_selected | 500,389 |
| single_domain_users_selected | 40,000 |
| products_with_metadata | 259,808 |
| canonical_items | 169,214 |
| products_merged_away | 90,582 |
| final_interactions | 1,620,533 |
| final_users | 172,184 |
| final_bridge_users | 119,016 |

## Shape

* interactions per user: median 5, p90 17, max 1672
* rating distribution: 1★ 3.3%, 2★ 2.9%, 3★ 6.5%, 4★ 16.4%, 5★ 70.9%
* matrix density: 0.02210%
* movies: 31,847 items with positives, Gini 0.626, top 1% of items = 16.3% of positives
* games: 10,741 items with positives, Gini 0.670, top 1% of items = 17.1% of positives
* bridge users (≥1 positive each domain): 119,016; median positives movie 3 / game 2
* items with ≥1 theme: 94.5%; canonical items merging >1 product: 54.3%

## Strongest movie ↔ game affinities (bridge users, ≥15 co-likes, ranked by lift)

| movie / series | game | co-likes | lift |
|---|---|---:|---:|
| How to Fall in Love | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 20 | 229.8 |
| The Magic of Ordinary Days - Hallmark | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 29 | 228.3 |
| The Lost Valentine | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 16 | 206.1 |
| The Love Letter | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 16 | 183.8 |
| All of My Heart Collection (All of My Heart / Inn Love / The | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 15 | 141.7 |
| Soldier Love Story | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 17 | 136.3 |
| Outlander | Outlander - Jamie Fraser | 18 | 134.1 |
| Crown for Christmas | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 17 | 103.2 |
| A Very Merry Mix-Up | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 18 | 98.1 |
| The Christmas Card (Hallmark) | Loving Leah: Hallmark Hall of Fame Gold Crown Collector's Ed | 20 | 69.7 |
| Wonder | Wonder | 15 | 51.2 |
| Kingsglaive - Final Fantasy XV | Final Fantasy XV: Day One Edition | 26 | 49.8 |
| Final Fantasy VII - Advent Children | Kingdom Hearts II | 22 | 38.1 |
| Sin City | Resident Evil 4 | 17 | 31.8 |
| Final Fantasy VII - Advent Children | Final Fantasy VII | 25 | 31.8 |
| The Terminator [UMD for PSP] | Grand Theft Auto Vice City | 16 | 27.2 |
| Kill Bill: Volume 1 | Resident Evil 4 | 19 | 25.0 |
| Final Fantasy VII - Advent Children | Final Fantasy XIII | 16 | 21.6 |
| Final Fantasy VII - Advent Children | Final Fantasy X | 17 | 19.8 |
| The Lego Movie | The LEGO Movie Videogame | 17 | 19.6 |
| The Lord of the Rings: The Fellowship of the Ring | Grand Theft Auto Vice City | 21 | 19.3 |
| Final Fantasy VII - Advent Children | Kingdom Hearts | 17 | 18.5 |
| The Dark Knight | Resident Evil 5 - Standard Edition | 18 | 16.1 |
| Lego Movie, The | The LEGO Movie Videogame | 16 | 15.7 |
| The Lord of the Rings: The Fellowship of the Ring | Final Fantasy VII | 16 | 14.7 |

Lift = P(both) / (P(movie)·P(game)): how much more often the pair co-occurs than chance. These pairs are the raw cross-domain signal that the co-preference and two-tower models learn from.