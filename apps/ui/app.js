// CrossVerse web UI: talks to the FastAPI service it is served from (/ui -> same origin).
const API = new URL("..", location.href).pathname.replace(/\/$/, "");
const K = 24;
// Starting points for the empty state and "Surprise me" (searched by title, so they survive retraining).
const FAVOURITES = [
  { q: "Red Dead Redemption 2", domain: "game" }, { q: "Blade Runner 2049", domain: "movie" },
  { q: "Witcher 3: Wild Hunt", domain: "game" }, { q: "(500) Days of Summer", domain: "movie" },
  { q: "Mass Effect 2", domain: "game" }, { q: "Interstellar", domain: "movie" },
  { q: "The Last of Us", domain: "game" }, { q: "John Wick", domain: "movie" },
  { q: "BioShock", domain: "game" }, { q: "Spirited Away", domain: "movie" },
  { q: "Alien", domain: "movie" }, { q: "Final Fantasy VII", domain: "game" },
];
// The slider runs from Balanced (0.5) to Same DNA (1). Below 0.5 the trained ranker alone mostly returns
// bestsellers (judged set: 16% good / 62.5% bad picks at 0), so the UI doesn't offer it; the API still does.
const TASTE_MIN = 0.5;
const VIBES = [
  { max: 0.7, label: "⚖️ Balanced", hint: "fan favourites that also share the feel" },
  { max: 1.01, label: "🧬 Same DNA", hint: "closest story, setting and themes" },
];
// The API accepts up to 100 liked and 100 disliked titles; a single cap on all picks keeps both under it.
const MAX_PICKS = 100;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { picks: [], to: "game", toTouched: false, taste: 0.8, moods: new Set(), text: "" };
const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : String(c));
  return el;
}

const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function api(path, { body, signal } = {}) {
  const r = await fetch(API + path, {
    method: body ? "POST" : "GET", signal,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

const hue = (s) => [...s].reduce((x, ch) => (x * 31 + ch.charCodeAt(0)) % 360, 7);
const initials = (t) => t.replace(/[^\p{L}\p{N} ]/gu, " ").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();

/** Cover image, or a coloured placeholder with initials when there is none (or it fails to load). */
function cover(item, cls = "thumb") {
  const fallback = () => h("div", { class: `${cls} fallback`, style: `--h:${hue(item.title)}`, "aria-hidden": "true" }, initials(item.title));
  if (!item.image) return fallback();
  const img = h("img", { class: cls, src: item.image, alt: "", loading: "lazy", decoding: "async", referrerpolicy: "no-referrer" });
  img.addEventListener("error", () => img.replaceWith(fallback()), { once: true });
  return img;
}

const year = (item) => (item.year && !String(item.title).includes(String(item.year)) ? item.year : "");
// kind: "movie" | "tv" | "game" (older API responses only have domain)
const kindOf = (it) => it.kind || (it.domain === "game" ? "game" : "movie");
const KIND_LABEL = { movie: "Movie", tv: "TV series", game: "Game" };
const domainLabel = (it) => KIND_LABEL[kindOf(it)];
const tag = (it) => h("span", { class: `tag ${kindOf(it)}` }, { movie: "Film", tv: "TV", game: "Game" }[kindOf(it)]);
// Anime is a tag on films and series, not a medium of its own: it shows under Movies and TV series.
const animeBadge = (it) => (it.domain !== "game" && it.themes?.includes("anime") ? h("span", { class: "tag anime" }, "Anime") : null);
// titles newer than the review data (served by story match only)
/** Public score: IGDB critics (games, /100) first, else the audience score (TMDB /10, IGDB players /100). */
function scoreText(it) {
  const r = it.rating;
  if (!r) return null;
  if (r.critics != null) return { short: `🏆 ${r.critics}`, long: `Critics ${r.critics}/100 (${r.critics_count} reviews, IGDB)` };
  if (r.audience == null) return null;
  return r.source === "tmdb"
    ? { short: `★ ${r.audience.toFixed(1)}`, long: `Audience ${r.audience.toFixed(1)}/10 (${r.votes.toLocaleString()} votes, TMDB)` }
    : { short: `★ ${Math.round(r.audience)}`, long: `Players ${Math.round(r.audience)}/100 (${r.votes.toLocaleString()} ratings, IGDB)` };
}
const scoreBadge = (it) => {
  const s = scoreText(it);
  return s ? h("span", { class: "score", title: s.long }, s.short) : null;
};
const newBadge = (it) => (it.new ? h("span", { class: "tag new", title: "Newer than our review data: matched by story only" }, "New") : null);

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  t.style.animation = "none"; void t.offsetWidth; t.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 2200);
}

// ---------------------------------------------------------------------------------------------
// URL state: picks, direction, vibe and moods survive reloads and can be shared
// ---------------------------------------------------------------------------------------------
function saveUrl() {
  const p = new URLSearchParams();
  const liked = state.picks.filter((x) => x.liked).map((x) => x.item_id);
  const disliked = state.picks.filter((x) => !x.liked).map((x) => x.item_id);
  if (liked.length) p.set("l", liked.join(","));
  if (disliked.length) p.set("d", disliked.join(","));
  p.set("to", state.to);
  p.set("t", String(state.taste));
  if (state.moods.size) p.set("m", [...state.moods].join(","));
  if (state.text) p.set("q", state.text);
  history.replaceState(null, "", `${location.pathname}?${p}`);
}

async function loadUrl() {
  const p = new URLSearchParams(location.search);
  const to = p.get("to") === "movie" ? "film" : p.get("to");  // links from before TV got its own tab
  state.to = ["game", "film", "tv", "both"].includes(to) ? to : "game";
  state.toTouched = p.has("to");
  const t = Number(p.get("t"));
  state.taste = p.has("t") && t >= 0 && t <= 1 ? Math.max(t, TASTE_MIN) : 0.8;
  state.moods = new Set((p.get("m") || "").split(",").filter(Boolean));
  state.text = p.get("q") || "";
  const ids = [...(p.get("l") || "").split(",").map((id) => [id, true]), ...(p.get("d") || "").split(",").map((id) => [id, false])]
    .filter(([id]) => id).slice(0, MAX_PICKS);
  const items = await Promise.all(ids.map(([id]) => api(`/items/${encodeURIComponent(id)}`).catch(() => null)));
  state.picks = items.map((it, n) => it && { ...it, liked: ids[n][1] }).filter(Boolean);
}

// ---------------------------------------------------------------------------------------------
// search with type-ahead
// ---------------------------------------------------------------------------------------------
const search = $("#search");
const dropdown = $("#search-results");
let searchCtl, hits = [], active = -1;

function closeDropdown() {
  dropdown.hidden = true;
  search.parentElement.setAttribute("aria-expanded", "false");
  active = -1;
}

function renderDropdown() {
  dropdown.replaceChildren();
  if (!hits.length) dropdown.append(h("li", { class: "empty" }, "No match. Try another spelling or a shorter title."));
  hits.forEach((it, n) => {
    dropdown.append(h("li", {
      role: "option", id: `opt-${n}`, "aria-selected": n === active ? "true" : "false",
      onmousedown: (e) => { e.preventDefault(); addPick(it); },
    }, cover(it), h("div", {},
      h("div", { class: "t" }, it.title, " ", h("span", { class: "year" }, year(it))),
      h("div", { class: "s" }, tag(it), animeBadge(it), newBadge(it), it.themes?.length ? `  ${it.themes.slice(0, 3).join(" · ")}` : ""))));
  });
  dropdown.hidden = false;
  search.parentElement.setAttribute("aria-expanded", "true");
}

const runSearch = debounce(async (q) => {
  searchCtl?.abort();
  if (q.trim().length < 2) return closeDropdown();
  searchCtl = new AbortController();
  try {
    const picked = new Set(state.picks.map((x) => x.item_id));
    hits = (await api(`/items/search?q=${encodeURIComponent(q)}&limit=8`, { signal: searchCtl.signal })).filter((x) => !picked.has(x.item_id));
    active = hits.length ? 0 : -1;
    renderDropdown();
  } catch (e) {
    if (e.name !== "AbortError") toast(`Search failed: ${e.message}`);
  }
}, 110);

search.addEventListener("input", () => runSearch(search.value));
search.addEventListener("keydown", (e) => {
  if (dropdown.hidden) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + hits.length) % hits.length;
    renderDropdown();
    search.setAttribute("aria-activedescendant", `opt-${active}`);
  } else if (e.key === "Enter" && hits[active]) {
    e.preventDefault();
    addPick(hits[active]);
  } else if (e.key === "Escape") closeDropdown();
});
search.addEventListener("blur", () => setTimeout(closeDropdown, 120));
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !$("#detail").open && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
    e.preventDefault();
    search.focus();
  }
});

// ---------------------------------------------------------------------------------------------
// picks, moods, direction and vibe
// ---------------------------------------------------------------------------------------------
function addPick(item, liked = true) {
  if (state.picks.some((x) => x.item_id === item.item_id)) return;
  if (state.picks.length >= MAX_PICKS) return toast(`That's ${MAX_PICKS} picks already: remove one to add another.`);
  state.picks.push({ ...item, liked });
  if (!state.toTouched && state.picks.length === 1) state.to = item.domain === "game" ? "film" : "game";
  search.value = "";
  closeDropdown();
  changed();
}

function renderPicks() {
  $("#picks").replaceChildren(...state.picks.map((p) => h("li", { class: `pick ${kindOf(p)}${p.liked ? "" : " disliked"}` },
    cover(p),
    h("span", { class: "t", title: p.title }, p.title),
    h("button", {
      type: "button", title: p.liked ? "You love this. Click to mark it as not your thing" : "Not your thing. Click to switch back",
      "aria-label": p.liked ? "Liked" : "Disliked", onclick: () => { p.liked = !p.liked; changed(); },
    }, p.liked ? "♥" : "✕"),
    h("button", { type: "button", "aria-label": `Remove ${p.title}`, onclick: () => { state.picks = state.picks.filter((x) => x !== p); changed(); } }, "×"))));
  $("#moods").hidden = state.picks.length > 0;
}

async function renderMoods() {
  const themes = await api("/themes").catch(() => []);
  const wrap = $("#mood-chips");
  let all = false;
  const SHOWN = 14;  // 52 chips fill a phone screen; the rest are one tap away
  const draw = () => {
    const visible = all ? themes : themes.filter((t, n) => n < SHOWN || state.moods.has(t));
    wrap.replaceChildren(...visible.map((t) => h("button", {
      type: "button", class: "mood", "aria-pressed": state.moods.has(t) ? "true" : "false",
      onclick: () => { state.moods.has(t) ? state.moods.delete(t) : state.moods.add(t); draw(); changed(); },
    }, t)), themes.length > SHOWN ? h("button", {
      type: "button", class: "mood more", onclick: () => { all = !all; draw(); },
    }, all ? "− fewer" : `+ ${themes.length - visible.length} more`) : null);
  };
  draw();
  const free = $("#free-text");
  free.value = state.text;
  free.addEventListener("input", debounce(() => { state.text = free.value.trim(); changed(); }, 350));
}

function renderControls() {
  for (const b of document.querySelectorAll(".direction [role=radio]")) b.setAttribute("aria-checked", b.dataset.to === state.to ? "true" : "false");
  $("#taste").value = String(state.taste);
  const v = VIBES.find((x) => state.taste < x.max);
  $("#vibe-label").textContent = v.label;
  $("#vibe-hint").textContent = v.hint;
}

for (const b of document.querySelectorAll(".direction [role=radio]")) {
  b.addEventListener("click", () => { state.to = b.dataset.to; state.toTouched = true; changed(); });
}
let lastScreen = "film";  // the swap button returns to whichever of Movies / TV was used last
$("#swap").addEventListener("click", (e) => {
  if (state.to === "film" || state.to === "tv") lastScreen = state.to;
  state.to = state.to === "game" ? lastScreen : "game";
  state.toTouched = true;
  e.currentTarget.classList.toggle("spin");
  changed();
});
$("#taste").addEventListener("input", (e) => { state.taste = Number(e.target.value); changed(); });

$("#share").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(location.href); toast("Link copied. Send it to a friend!"); }
  catch { prompt("Copy this link:", location.href); }
});
$("#surprise").addEventListener("click", async () => {
  const ex = FAVOURITES[Math.floor(Math.random() * FAVOURITES.length)];
  const [hit] = await api(`/items/search?q=${encodeURIComponent(ex.q)}&domain=${ex.domain}&limit=1`).catch(() => []);
  if (!hit) return toast("No luck this time. Try again!");
  state.picks = [];
  state.toTouched = false;
  addPick(hit);
  toast(`🎲 Starting from ${hit.title}`);
  window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
});

// ---------------------------------------------------------------------------------------------
// recommendations
// ---------------------------------------------------------------------------------------------
let recCtl, lastBody = "";

function skeletons(n = 12) {
  $("#featured").replaceChildren();
  renderUniverse([]);
  renderFresh([]);
  $("#grid").replaceChildren(...Array.from({ length: n }, () =>
    h("div", { class: "skeleton", "aria-hidden": "true" }, h("div", { class: "poster" }), h("div", { class: "line" }), h("div", { class: "line", style: "width:60%" }))));
}

async function emptyState() {
  $("#results-title").textContent = "Pick a starting point";
  $("#results-meta").textContent = "";
  $("#notice").hidden = true;
  $("#featured").replaceChildren();
  renderUniverse([]);
  renderFresh([]);
  const starters = h("div", { class: "starters" });
  $("#grid").replaceChildren(h("div", { class: "empty-state" },
    h("strong", {}, "Start with something you love"),
    "Search above, pick a vibe, or tap one of these:", starters));
  const found = await Promise.all(FAVOURITES.slice(0, 8).map((ex) =>
    api(`/items/search?q=${encodeURIComponent(ex.q)}&domain=${ex.domain}&limit=1`).then((r) => r[0]).catch(() => null)));
  starters.replaceChildren(...found.filter(Boolean).map((it) => h("button", { type: "button", class: "starter", onclick: () => addPick(it) },
    h("div", { class: "poster" }, cover(it), tag(it)), h("span", {}, it.title))));
}

const recommend = debounce(async () => {
  const liked = state.picks.filter((x) => x.liked);
  if (!liked.length && !state.moods.size && !state.text) { lastBody = ""; return emptyState(); }
  const body = {
    liked: liked.map((x) => ({ item: x.item_id })), disliked: state.picks.filter((x) => !x.liked).map((x) => x.item_id),
    target_domain: state.to === "both" ? null : state.to, k: K, taste: state.taste,
    preferences: [...state.moods], free_text: state.text, explain: true,
  };
  const key = JSON.stringify(body);
  if (key === lastBody) return;
  lastBody = key;
  recCtl?.abort();
  recCtl = new AbortController();
  if (!$("#grid").querySelector(".card")) skeletons();
  else $("#grid").style.opacity = "0.5";
  const t0 = performance.now();
  try {
    renderResults(await api("/recommend", { body, signal: recCtl.signal }), Math.round(performance.now() - t0));
  } catch (e) {
    if (e.name === "AbortError") return;
    lastBody = "";
    $("#featured").replaceChildren();
    $("#grid").replaceChildren(h("div", { class: "empty-state" }, h("strong", {}, "Something went wrong"), e.message));
  } finally {
    $("#grid").style.opacity = "";
  }
}, 140);

function because(it) {
  const a = it.evidence?.anchors?.[0];
  if (!a || state.picks.length < 2) return null;  // with one pick it is obvious
  const src = state.picks.find((p) => p.item_id === a.item_id) || a;
  return h("div", { class: "because", title: `Because you liked ${a.title}` }, "⤷", cover(src), h("b", {}, a.title));
}

function renderResults(res, ms) {
  const label = { game: "Games", film: "Movies", tv: "TV series", both: "Movies, TV & games" }[state.to];
  $("#results-title").textContent = `${label} for you`;
  $("#results-meta").textContent = `${res.items.length} picks · ${res.candidate_count} candidates · ${ms} ms`;
  $("#model-badge").textContent = `model ${res.model_version} · ${ms} ms`;
  const notes = [];
  if (res.unresolved?.length) notes.push(`Not found: ${res.unresolved.join(", ")}`);
  if (res.cold_start) notes.push("No titles yet: picking by vibe and what people love most.");
  for (const [id, subs] of Object.entries(res.stand_ins || {})) {
    const name = state.picks.find((p) => p.item_id === id)?.title || "A new title";
    notes.push(subs.length
      ? `${name} is newer than our review data, so the list below uses ${subs.map((x) => x.title).join(" and ")} in its place.`
      : `${name} is newer than our review data: see the new releases above.`);
  }
  $("#notice").hidden = !notes.length;
  $("#notice").textContent = notes.join(" · ");
  renderUniverse(res.same_universe || []);
  const inUniverse = new Set((res.same_universe || []).map((x) => x.item_id));
  renderFresh((res.new_releases || []).filter((x) => !inUniverse.has(x.item_id)));
  if (!res.items.length && !res.same_universe?.length) {
    $("#featured").replaceChildren();
    $("#grid").replaceChildren(h("div", { class: "empty-state" }, h("strong", {}, "Nothing to show"), "Try adding another title."));
    return;
  }
  // an adaptation already shown in the same-universe row is not repeated in the ranked list
  const shown = new Set((res.same_universe || []).map((x) => x.item_id));
  const [top, ...rest] = res.items.filter((x) => !shown.has(x.item_id));
  $("#featured").replaceChildren(...(top ? [feature(top)] : []));
  $("#grid").replaceChildren(...rest.map((it, n) => card(it, n + 2)));
}

/** Adaptations and tie-ins of the picked titles (The Last of Us game -> the HBO series). Not ranked by taste. */
function renderUniverse(items) {
  const vias = [...new Set(items.map((x) => x.via))];
  shelf($("#universe"), "universe-title", "🔗 Same universe", items,
    items.length && `${describe(items)} from the world of ${vias.length === 1 ? vias[0] : "your picks"}`, vias.length > 1 ? "from" : null);
}

/** Titles newer than the review data, closest in story to the picks. */
function renderFresh(items) {
  const vias = new Set(items.map((x) => x.via));
  shelf($("#fresh"), "fresh-title", "✨ New releases", items,
    items.length && `${describe(items)} from after our review data (2023), matched by story`, vias.size > 1 ? "like" : null);
}

/** A titled row of compact cards; `viaLabel` adds "from X" / "like X" under each card. */
function shelf(box, id, title, items, caption, viaLabel) {
  box.hidden = !items.length;
  if (!items.length) return box.replaceChildren();
  box.replaceChildren(
    h("div", { class: "universe-head" }, h("h3", { id }, title), h("span", { class: "meta" }, caption)),
    h("div", { class: "row" }, items.map((it) => {
      const el = card(it, null, true);
      if (viaLabel) el.querySelector(".card-body").append(h("div", { class: "via" }, `${viaLabel} ${it.via}`));
      return el;
    })));
}

function describe(items) {
  const kinds = new Set(items.map(kindOf));
  const words = [kinds.has("movie") && "films", kinds.has("tv") && "series", kinds.has("game") && "games"].filter(Boolean);
  const text = words.length > 1 ? `${words.slice(0, -1).join(", ")} and ${words.at(-1)}` : words[0];
  return text[0].toUpperCase() + text.slice(1);
}

function feedback(item, event, btn) {
  api("/feedback", { body: { item_id: item.item_id, event, recommendation_id: item.recommendation_id } })
    .then(() => { btn?.classList.add(event === "like" ? "on-good" : "on-bad"); toast(event === "like" ? "Nice! We'll remember that." : "Got it. Less like this."); })
    .catch((e) => toast(`Feedback failed: ${e.message}`));
}

function quickActions(it) {
  const stop = (fn) => (e) => { e.stopPropagation(); fn(e); };
  return h("div", { class: "quick" },
    h("button", { type: "button", title: "Good pick", "aria-label": "Good pick", onclick: stop((e) => feedback(it, "like", e.currentTarget)) }, "👍"),
    h("button", { type: "button", title: "Not for me", "aria-label": "Not for me", onclick: stop((e) => feedback(it, "dislike", e.currentTarget)) }, "👎"),
    h("button", { type: "button", title: "Add to my taste", "aria-label": "Add to my taste", onclick: stop(() => { addPick(it); toast(`♥ Added ${it.title}`); }) }, "♥"));
}

function tilt(el) {
  if (reduceMotion || !matchMedia("(hover: hover)").matches) return;
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    el.style.transform = `rotateX(${(0.5 - y) * 8}deg) rotateY(${(x - 0.5) * 10}deg) translateY(-4px)`;
    el.style.setProperty("--mx", `${x * 100}%`);
    el.style.setProperty("--my", `${y * 100}%`);
  });
  el.addEventListener("pointerleave", () => { el.style.transform = ""; });
}

function card(it, rank, compact = false) {
  const el = h("article", { class: `card ${kindOf(it)}`, tabindex: "0", "aria-label": `${it.title}, ${domainLabel(it)}`, style: `animation-delay:${Math.min(rank || 0, 14) * 25}ms` },
    h("span", { class: "shimmer" }),
    h("div", { class: "poster" }, cover(it), rank ? h("span", { class: "rank" }, rank) : null, tag(it), animeBadge(it), newBadge(it),
      compact ? null : quickActions(it)),
    h("div", { class: "card-body" },
      h("div", { class: "card-title" }, it.title, " ", h("span", { class: "year" }, year(it)), " ", scoreBadge(it)),
      compact ? null : h("div", { class: "pills" }, (it.themes || []).slice(0, 3).map((t) => h("span", { class: "pill" }, t))),
      compact ? null : because(it),
      compact || !it.evidence?.summary ? null : h("div", { class: "reason" }, it.evidence.summary.replace(/^Recommended because /, "").replace(/^./, (c) => c.toUpperCase()))));
  const open = () => openDetail(it);
  el.addEventListener("click", open);
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  tilt(el);
  return el;
}

function feature(it) {
  const reason = it.evidence?.summary?.replace(/^Recommended because /, "").replace(/^./, (c) => c.toUpperCase());
  const el = h("article", { class: "feature", tabindex: "0", "aria-label": `Top match: ${it.title}` },
    it.image ? h("div", { class: "backdrop", style: `background-image:url("${it.image.replace(/"/g, "%22")}")` }) : null,
    h("div", { class: "poster" }, cover(it), tag(it), animeBadge(it)),
    h("div", {},
      h("div", { class: "eyebrow" }, "★ Top match"),
      h("h3", {}, it.title, " ", h("span", { class: "year" }, year(it))),
      h("div", { class: "pills" }, scoreBadge(it), (it.themes || []).slice(0, 5).map((t) => h("span", { class: "pill" }, t))),
      reason ? h("p", { class: "reason" }, reason) : null,
      h("div", { class: "actions" },
        h("button", { type: "button", class: "btn primary", onclick: (e) => { e.stopPropagation(); addPick(it); toast(`♥ Added ${it.title}`); } }, "♥ Add to my taste"),
        h("button", { type: "button", class: "btn", onclick: (e) => { e.stopPropagation(); openDetail(it); } }, "Why this? →"))));
  el.addEventListener("click", () => openDetail(it));
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") openDetail(it); });
  return el;
}

// ---------------------------------------------------------------------------------------------
// detail dialog: why this pick, and more like it
// ---------------------------------------------------------------------------------------------
const dialog = $("#detail");
dialog.addEventListener("click", (e) => { if (e.target === dialog || e.target.closest("[data-close]")) dialog.close(); });

async function openDetail(it) {
  const ev = it.evidence || {};
  const facts = [];
  const score = scoreText(it);
  if (score) facts.push(["Score", score.long]);
  if (it.rating?.critics != null && it.rating.audience != null) facts.push(["Players", `${Math.round(it.rating.audience)}/100 (${it.rating.votes.toLocaleString()} ratings, IGDB)`]);
  if (ev.shared_themes?.length) facts.push(["Shared themes", ev.shared_themes.join(", ")]);
  if (ev.users_who_liked_both) facts.push(["Fans in common", `${ev.users_who_liked_both} people liked both`]);
  if (ev.anchors?.length) facts.push(["Closest to", ev.anchors.map((a) => a.title).join(", ")]);
  if (ev.reason_confidence) facts.push(["Confidence", ev.reason_confidence]);
  $("#detail-body").replaceChildren(
    h("div", { class: "detail-top" },
      it.image ? h("div", { class: "backdrop", style: `background-image:url("${it.image.replace(/"/g, "%22")}")` }) : null,
      h("div", { class: "poster" }, cover(it), tag(it), animeBadge(it)),
      h("div", {},
        h("h3", { id: "detail-title" }, it.title),
        h("div", { class: "meta" }, [domainLabel(it), year(it), it.new && "new release"].filter(Boolean).join(" · ")),
        it.new ? h("p", { class: "meta" }, "Newer than our review data (2023), so no fans-in-common yet: it is matched by story, setting and franchise.") : null,
        h("div", { class: "pills", style: "margin-top:10px" }, (it.themes || []).map((t) => h("span", { class: "pill" }, t))),
        ev.summary ? h("p", {}, ev.summary) : null,
        facts.length ? h("dl", { class: "facts" }, facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null,
        h("div", { class: "detail-actions" },
          h("button", { type: "button", class: "btn primary", onclick: () => { addPick(it); dialog.close(); toast(`♥ Added ${it.title}`); } }, "♥ Add to my taste"),
          h("button", { type: "button", class: "btn", onclick: () => { addPick(it, false); dialog.close(); } }, "✕ Not my thing"),
          h("a", { class: "btn", href: `https://www.google.com/search?q=${encodeURIComponent(`${it.title} ${{ game: "video game", tv: "TV series", movie: "film" }[kindOf(it)]}`)}`, target: "_blank", rel: "noopener" }, "Look it up ↗")))),
    h("p", { class: "row-title" }, "More like this"),
    h("div", { class: "row", id: "similar-row" }, Array.from({ length: 6 }, () => h("div", { class: "skeleton" }, h("div", { class: "poster" })))));
  if (!dialog.open) dialog.showModal();
  if (it.new) {  // not in the trained model: no neighbours to show
    $("#similar-row").previousElementSibling.remove();
    $("#similar-row").remove();
    return;
  }
  try {
    const sim = await api(`/similar/${it.domain}/${encodeURIComponent(it.item_id)}?k=10`);
    $("#similar-row").replaceChildren(...[...sim.cross_domain.slice(0, 6), ...sim.same_domain.slice(0, 6)].map((x) => card(x, null, true)));
  } catch {
    $("#similar-row").replaceChildren(h("span", { class: "meta" }, "Couldn't load similar titles."));
  }
}

// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// Letterboxd import: the member's own data export (.zip or ratings.csv), read in the browser
// ---------------------------------------------------------------------------------------------
const LOVED_STARS = 4, DISLIKED_STARS = 1.5, MAX_LOVED = 10, MAX_DISLIKED = 3;

/** RFC 4180 CSV: quoted fields may hold commas, quotes ("") and line breaks. */
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') quoted = false; else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((x) => x.trim()));
}

/** One file out of a .zip, via the central directory (sizes there are always filled in). */
async function unzipFile(blob, wanted) {
  const buf = new DataView(await blob.arrayBuffer());
  let end = buf.byteLength - 22;
  while (end >= 0 && buf.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0) throw new Error("not a zip file");
  let at = buf.getUint32(end + 16, true);
  const entries = {};
  for (let n = buf.getUint16(end + 10, true); n > 0; n--) {
    const nameLen = buf.getUint16(at + 28, true);
    const name = new TextDecoder().decode(new Uint8Array(buf.buffer, at + 46, nameLen));
    entries[name.split("/").pop()] = { method: buf.getUint16(at + 10, true), size: buf.getUint32(at + 20, true), local: buf.getUint32(at + 42, true) };
    at += 46 + nameLen + buf.getUint16(at + 30, true) + buf.getUint16(at + 32, true);
  }
  const e = wanted.map((w) => entries[w]).find(Boolean);
  if (!e) return null;
  const start = e.local + 30 + buf.getUint16(e.local + 26, true) + buf.getUint16(e.local + 28, true);
  const data = new Blob([new Uint8Array(buf.buffer, start, e.size)]);
  if (e.method === 0) return data.text();
  return new Response(data.stream().pipeThrough(new DecompressionStream("deflate-raw"))).text();
}

async function importLetterboxd(file) {
  try {
    const text = /\.zip$/i.test(file.name) ? await unzipFile(file, ["ratings.csv", "watched.csv"]) : await file.text();
    if (!text) throw new Error("no ratings.csv in that zip");
    const [head, ...rows] = parseCsv(text.replace(/^\uFEFF/, ""));
    const col = (name) => head.findIndex((h) => h.trim().toLowerCase() === name);
    const [ni, yi, ri] = [col("name"), col("year"), col("rating")];
    if (ni < 0) throw new Error("expected Letterboxd's ratings.csv (a Name column)");
    const films = rows.filter((r) => r[ni]).slice(0, 5000)
      .map((r) => ({ name: r[ni], year: Number(r[yi]) || null, rating: ri >= 0 && r[ri] ? Number(r[ri]) : null }));
    toast(`Matching ${films.length} films…`);
    const res = await api("/import/letterboxd", { body: { films } });
    const rated = res.matched.filter((m) => m.member_rating != null);
    const loved = (rated.length ? rated.filter((m) => m.member_rating >= LOVED_STARS) : res.matched).slice(0, MAX_LOVED);
    const disliked = rated.filter((m) => m.member_rating <= DISLIKED_STARS).slice(-MAX_DISLIKED);
    const had = new Set(state.picks.map((p) => p.item_id));
    const fresh = [...loved.map((m) => [m, true]), ...disliked.map((m) => [m, false])].filter(([m]) => !had.has(m.item_id))
      .slice(0, Math.max(MAX_PICKS - state.picks.length, 0));
    if (!fresh.length && state.picks.length >= MAX_PICKS) return toast(`That's ${MAX_PICKS} picks already: remove some to import more.`);
    if (!fresh.length) return toast(`Found ${res.matched.length} of ${res.total} films, but none rated ${LOVED_STARS}★ or more.`);
    fresh.forEach(([m, liked]) => state.picks.push({ ...m, liked }));
    if (!state.toTouched) state.to = "game";
    changed();
    const nLoved = fresh.filter(([, liked]) => liked).length, nBad = fresh.length - nLoved;
    toast(`Imported ${nLoved} favourite${nLoved === 1 ? "" : "s"}${nBad ? ` and ${nBad} dislike${nBad === 1 ? "" : "s"}` : ""} (${res.matched.length} of ${res.total} films found)`);
  } catch (e) {
    toast(`Letterboxd import failed: ${e.message}`);
  }
}
$("#letterboxd-file").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";  // the same file can be picked again
  if (file) importLetterboxd(file);
});

function changed() {
  renderPicks();
  renderControls();
  saveUrl();
  recommend();
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("cv-theme"); } catch { /* storage blocked */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const next = getComputedStyle(document.documentElement).colorScheme.includes("dark") ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("cv-theme", next); } catch { /* storage blocked */ }
  });
}

async function init() {
  initTheme();
  api("/health").then((hl) => ($("#model-badge").textContent = `model ${hl.model_version}`)).catch(() => ($("#model-badge").textContent = "API offline"));
  await loadUrl();
  renderMoods();
  changed();
}

init();
