// CrossVerse web UI: talks to the FastAPI service it is served from (/ui -> same origin).
const API = new URL("..", location.href).pathname.replace(/\/$/, "");
const K = 24;
const EXAMPLES = [
  { q: "Red Dead Redemption 2", domain: "game" },
  { q: "Blade Runner 2049", domain: "movie" },
  { q: "Witcher 3: Wild Hunt", domain: "game" },
  { q: "(500) Days of Summer", domain: "movie" },
];

const state = { picks: [], to: "game", toTouched: false, taste: 0.8, moods: new Set(), text: "" };
const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------------------------
// small helpers
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

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function api(path, { body, signal } = {}) {
  const r = await fetch(API + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

function hue(s) {
  let x = 0;
  for (const ch of s) x = (x * 31 + ch.charCodeAt(0)) % 360;
  return x;
}

function initials(title) {
  return title.replace(/[^\p{L}\p{N} ]/gu, " ").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
}

/** Cover image, or a coloured placeholder with initials when there is none (or it fails to load). */
function cover(item, cls = "thumb") {
  const fallback = () => h("div", { class: `${cls} fallback`, style: `--h:${hue(item.title)}`, "aria-hidden": "true" }, initials(item.title));
  if (!item.image) return fallback();
  const img = h("img", { class: cls, src: item.image, alt: "", loading: "lazy", decoding: "async", referrerpolicy: "no-referrer" });
  img.addEventListener("error", () => img.replaceWith(fallback()), { once: true });
  return img;
}

const year = (item) => (item.year && !String(item.title).includes(String(item.year)) ? item.year : "");
const domainLabel = (d) => (d === "game" ? "Game" : "Movie / TV");

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 2200);
}

// ---------------------------------------------------------------------------------------------
// URL state: picks, target, style and moods survive reloads and can be shared
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
  state.to = ["game", "movie", "both"].includes(p.get("to")) ? p.get("to") : "game";
  state.toTouched = p.has("to");
  const t = Number(p.get("t"));
  state.taste = p.has("t") && t >= 0 && t <= 1 ? t : 0.8;
  state.moods = new Set((p.get("m") || "").split(",").filter(Boolean));
  state.text = p.get("q") || "";
  const ids = [...(p.get("l") || "").split(",").map((id) => [id, true]), ...(p.get("d") || "").split(",").map((id) => [id, false])]
    .filter(([id]) => id);
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
  if (!hits.length) {
    dropdown.append(h("li", { class: "empty" }, "No match. Try another spelling, or a shorter title."));
  }
  hits.forEach((it, n) => {
    dropdown.append(h("li", {
      role: "option", id: `opt-${n}`, "aria-selected": n === active ? "true" : "false",
      onmousedown: (e) => { e.preventDefault(); addPick(it); },
    }, cover(it), h("div", {}, h("div", { class: "t" }, it.title, " ", h("span", { class: "card-year" }, year(it))),
      h("div", { class: "s" }, domainLabel(it.domain), it.themes?.length ? ` · ${it.themes.slice(0, 3).join(", ")}` : ""))));
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
    hits = (await api(`/items/search?q=${encodeURIComponent(q)}&limit=8`, { signal: searchCtl.signal }))
      .filter((x) => !picked.has(x.item_id));
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
  } else if (e.key === "Escape") {
    closeDropdown();
  }
});
search.addEventListener("blur", () => setTimeout(closeDropdown, 120));
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== search && !$("#detail").open && document.activeElement?.tagName !== "INPUT") {
    e.preventDefault();
    search.focus();
  }
});

// ---------------------------------------------------------------------------------------------
// picks, moods and controls
// ---------------------------------------------------------------------------------------------
function addPick(item, liked = true) {
  if (state.picks.some((x) => x.item_id === item.item_id)) return;
  state.picks.push({ ...item, liked });
  if (!state.toTouched && state.picks.length === 1) state.to = item.domain === "game" ? "movie" : "game";
  search.value = "";
  closeDropdown();
  changed();
}

function renderPicks() {
  const ul = $("#picks");
  ul.replaceChildren(...state.picks.map((p) => h("li", { class: `pick${p.liked ? "" : " disliked"}` },
    cover(p),
    h("span", { class: "t", title: p.title }, p.title),
    h("button", {
      type: "button", title: p.liked ? "Liked: click to mark as disliked" : "Disliked: click to mark as liked",
      "aria-label": p.liked ? "Liked" : "Disliked",
      onclick: () => { p.liked = !p.liked; changed(); },
    }, p.liked ? "👍" : "👎"),
    h("button", { type: "button", "aria-label": `Remove ${p.title}`, onclick: () => {
      state.picks = state.picks.filter((x) => x !== p); changed();
    } }, "✕"),
  )));
  $("#moods").hidden = state.picks.length > 0;
}

async function renderMoods() {
  const themes = await api("/themes").catch(() => []);
  const wrap = $("#mood-chips");
  const draw = () => wrap.replaceChildren(...themes.map((t) => h("button", {
    type: "button", class: "mood", "aria-pressed": state.moods.has(t) ? "true" : "false",
    onclick: () => { state.moods.has(t) ? state.moods.delete(t) : state.moods.add(t); draw(); changed(); },
  }, t)));
  draw();
  const free = $("#free-text");
  free.value = state.text;
  free.addEventListener("input", debounce(() => { state.text = free.value.trim(); changed(); }, 350));
}

function renderControls() {
  for (const b of document.querySelectorAll(".segmented button")) {
    b.setAttribute("aria-checked", b.dataset.to === state.to ? "true" : "false");
  }
  $("#taste").value = String(state.taste);
}

for (const b of document.querySelectorAll(".segmented button")) {
  b.addEventListener("click", () => { state.to = b.dataset.to; state.toTouched = true; changed(); });
}
$("#taste").addEventListener("input", (e) => { state.taste = Number(e.target.value); changed(); });

// ---------------------------------------------------------------------------------------------
// recommendations
// ---------------------------------------------------------------------------------------------
let recCtl, lastBody = "";

function skeletons(n = 12) {
  $("#grid").replaceChildren(...Array.from({ length: n }, () =>
    h("div", { class: "skeleton", "aria-hidden": "true" }, h("div", { class: "poster" }), h("div", { class: "line" }), h("div", { class: "line", style: "width:60%" }))));
}

function emptyState() {
  $("#results-title").textContent = "Recommendations";
  $("#results-meta").textContent = "";
  $("#notice").hidden = true;
  $("#grid").replaceChildren(h("div", { class: "empty-state" },
    h("strong", {}, "Start with something you love"),
    "Search above, pick a mood, or try one of these: ",
    h("div", { class: "mood-chips", style: "justify-content:center;margin-top:12px" }, EXAMPLES.map((ex) =>
      h("button", { type: "button", class: "mood", onclick: async () => {
        const [hit] = await api(`/items/search?q=${encodeURIComponent(ex.q)}&domain=${ex.domain}&limit=1`);
        if (hit) addPick(hit);
      } }, ex.q)))));
}

const recommend = debounce(async () => {
  const liked = state.picks.filter((x) => x.liked);
  if (!liked.length && !state.moods.size && !state.text) return emptyState();
  const body = {
    liked: liked.map((x) => ({ item: x.item_id })),
    disliked: state.picks.filter((x) => !x.liked).map((x) => x.item_id),
    target_domain: state.to === "both" ? null : state.to,
    k: K, taste: state.taste, preferences: [...state.moods], free_text: state.text, explain: true,
  };
  const key = JSON.stringify(body);
  if (key === lastBody) return;
  lastBody = key;
  recCtl?.abort();
  recCtl = new AbortController();
  if (!$("#grid").querySelector(".card")) skeletons();
  else $("#grid").style.opacity = "0.55";
  const t0 = performance.now();
  try {
    const res = await api("/recommend", { body, signal: recCtl.signal });
    renderResults(res, Math.round(performance.now() - t0));
  } catch (e) {
    if (e.name === "AbortError") return;
    lastBody = "";
    $("#grid").replaceChildren(h("div", { class: "empty-state" }, h("strong", {}, "Something went wrong"), e.message));
  } finally {
    $("#grid").style.opacity = "";
  }
}, 140);

function renderResults(res, ms) {
  const label = { game: "Games", movie: "Movies & TV", both: "Movies, TV & games" }[state.to];
  $("#results-title").textContent = `${label} for you`;
  $("#results-meta").textContent = `${res.items.length} picks · ${res.candidate_count} candidates · ${ms} ms`;
  $("#model-badge").textContent = `model ${res.model_version} · ${ms} ms`;
  const notes = [];
  if (res.unresolved?.length) notes.push(`Not found: ${res.unresolved.join(", ")}`);
  if (res.cold_start) notes.push("No titles yet: picking by mood and what people like most.");
  $("#notice").hidden = !notes.length;
  $("#notice").textContent = notes.join(" · ");
  if (!res.items.length) {
    $("#grid").replaceChildren(h("div", { class: "empty-state" }, h("strong", {}, "Nothing to show"), "Try adding another title."));
    return;
  }
  $("#grid").replaceChildren(...res.items.map((it, n) => card(it, n + 1)));
}

function feedback(item, event, btn) {
  api("/feedback", { body: { item_id: item.item_id, event, recommendation_id: item.recommendation_id } })
    .then(() => {
      btn?.classList.add(event === "like" ? "on-good" : "on-bad");
      toast(event === "like" ? "Thanks! Noted as a good pick." : "Noted. Less like this.");
    })
    .catch((e) => toast(`Feedback failed: ${e.message}`));
}

function card(it, rank, compact = false) {
  const reason = it.evidence?.summary;
  const el = h("article", { class: "card", tabindex: "0", "aria-label": `${it.title}, ${domainLabel(it.domain)}` },
    h("div", { class: "poster" }, cover(it), rank ? h("span", { class: "rank" }, rank) : null,
      h("span", { class: `domain ${it.domain}` }, it.domain === "game" ? "Game" : "Film/TV")),
    h("div", { class: "card-body" },
      h("div", { class: "card-title" }, it.title, " ", h("span", { class: "card-year" }, year(it))),
      compact ? null : h("div", { class: "pills" }, (it.themes || []).slice(0, 3).map((t) => h("span", { class: "pill" }, t))),
      compact || !reason ? null : h("div", { class: "reason" }, reason),
      compact ? null : h("div", { class: "card-actions" },
        h("button", { type: "button", title: "Good pick", "aria-label": "Good pick", onclick: (e) => { e.stopPropagation(); feedback(it, "like", e.currentTarget); } }, "👍"),
        h("button", { type: "button", title: "Not for me", "aria-label": "Not for me", onclick: (e) => { e.stopPropagation(); feedback(it, "dislike", e.currentTarget); } }, "👎"),
        h("button", { type: "button", title: "Add to my taste", "aria-label": "Add to my taste", onclick: (e) => { e.stopPropagation(); addPick(it); toast(`Added ${it.title}`); } }, "＋"))));
  const open = () => openDetail(it);
  el.addEventListener("click", open);
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
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
  if (ev.shared_themes?.length) facts.push(["Shared themes", ev.shared_themes.join(", ")]);
  if (ev.users_who_liked_both) facts.push(["Fans in common", `${ev.users_who_liked_both} people liked both`]);
  if (ev.anchors?.length) facts.push(["Closest to", ev.anchors.map((a) => a.title).join(", ")]);
  if (ev.reason_confidence) facts.push(["Confidence", ev.reason_confidence]);
  if (ev.candidate_sources?.length) facts.push(["Found by", ev.candidate_sources.join(", ")]);
  const body = $("#detail-body");
  body.replaceChildren(
    h("div", { class: "detail-top" },
      h("div", { class: "poster" }, cover(it)),
      h("div", {},
        h("h3", { id: "detail-title" }, it.title),
        h("div", { class: "meta" }, [domainLabel(it.domain), year(it)].filter(Boolean).join(" · ")),
        h("div", { class: "pills", style: "margin-top:8px" }, (it.themes || []).map((t) => h("span", { class: "pill" }, t))),
        ev.summary ? h("p", {}, ev.summary) : null,
        facts.length ? h("dl", { class: "facts" }, facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null,
        h("div", { class: "detail-actions" },
          h("button", { type: "button", class: "btn primary", onclick: () => { addPick(it); dialog.close(); } }, "＋ Add to my taste"),
          h("button", { type: "button", class: "btn", onclick: () => { addPick(it, false); dialog.close(); } }, "👎 Not my thing"),
          h("a", { class: "btn", href: `https://www.google.com/search?q=${encodeURIComponent(`${it.title} ${it.domain === "game" ? "game" : "film"}`)}`, target: "_blank", rel: "noopener" }, "Look it up ↗")))),
    h("p", { class: "row-title" }, "More like this"),
    h("div", { class: "row", id: "similar-row" }, Array.from({ length: 6 }, () => h("div", { class: "skeleton" }, h("div", { class: "poster" })))));
  if (!dialog.open) dialog.showModal();
  try {
    const sim = await api(`/similar/${it.domain}/${encodeURIComponent(it.item_id)}?k=10`);
    const items = [...sim.cross_domain.slice(0, 6), ...sim.same_domain.slice(0, 6)];
    $("#similar-row").replaceChildren(...items.map((x) => card(x, null, true)));
  } catch {
    $("#similar-row").replaceChildren(h("span", { class: "meta" }, "Couldn't load similar titles."));
  }
}

// ---------------------------------------------------------------------------------------------
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
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("cv-theme", next); } catch { /* storage blocked */ }
  });
}

async function init() {
  initTheme();
  api("/health").then((hl) => ($("#model-badge").textContent = `model ${hl.model_version}`))
    .catch(() => ($("#model-badge").textContent = "API offline"));
  await loadUrl();
  renderMoods();
  changed();
  if (!state.picks.length) search.focus();
}

init();
