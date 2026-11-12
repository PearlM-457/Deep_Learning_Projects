/* Movie Night Picker - runs fully in the browser.
   Data: data/movies.json (movie columns), data/emb.bin (int8 text embeddings), data/live.json (in-theaters snapshot). */
(() => {
'use strict';

const IMG = 'https://image.tmdb.org/t/p/w342';
const MAX_PICKS = 8;
const LS_WATCHED = 'mnp_watched_v1';

let M, EMB, N = 0, DIM = 0, INV = 1;
let GENRES = [], GN = 0, GH, GC, POP, TLC = [], ID2ROW = new Map();
let LIVE = null;
const kwCache = new Map();

const S = { likedA: [], likedB: [], watched: [], two: false, genres: new Set(),
            minYear: 1980, maxRt: 300, popW: 0.1, genreW: 0.2, hideFr: true, n: 12 };

const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const safeLS = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage can be blocked */ } }
};

/* ------------------------------------------------------------------ data */
function multihot(genreStrings) {
  const gi = new Map(GENRES.map((g, i) => [g, i]));
  const n = genreStrings.length, H = new Uint8Array(n * GN), C = new Uint8Array(n);
  for (let r = 0; r < n; r++) {
    for (const g of String(genreStrings[r] || '').split(',')) {
      const k = gi.get(g.trim());
      if (k !== undefined && !H[r * GN + k]) { H[r * GN + k] = 1; C[r]++; }
    }
  }
  return [H, C];
}

function b64ToInt8(b64) {
  const bin = atob(b64), out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
  return out;
}

async function loadData() {
  const [meta, buf, live] = await Promise.all([
    fetch('data/movies.json').then(r => r.json()),
    fetch('data/emb.bin').then(r => r.arrayBuffer()),
    fetch('data/live.json').then(r => r.ok ? r.json() : null).catch(() => null)
  ]);
  M = meta.cols; N = meta.n; DIM = meta.dim; INV = 1 / (meta.scale * meta.scale);
  GENRES = meta.genres; GN = GENRES.length;
  EMB = new Int8Array(buf);
  if (EMB.length !== N * DIM) throw new Error('embeddings and movie table are out of sync');
  [GH, GC] = multihot(M.genres);
  POP = new Float32Array(N);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N; i++) { POP[i] = Math.log1p(M.votes[i]); lo = Math.min(lo, POP[i]); hi = Math.max(hi, POP[i]); }
  for (let i = 0; i < N; i++) POP[i] = (POP[i] - lo) / (hi - lo + 1e-9);
  TLC = M.title.map(t => t.toLowerCase());
  M.id.forEach((id, i) => ID2ROW.set(id, i));
  if (live && live.movies && live.movies.length) {
    const [H, C] = multihot(live.movies.map(m => m.genres));
    LIVE = { ...live, emb: b64ToInt8(live.emb), H, C, n: live.movies.length };
    if (LIVE.emb.length !== LIVE.n * DIM) LIVE = null;
  }
}

const kws = i => {
  let s = kwCache.get(i);
  if (!s) { s = new Set(M.kw[i] ? M.kw[i].split('|') : []); kwCache.set(i, s); }
  return s;
};

/* ------------------------------------------------------------------ maths */
function dotCat(i, j) {
  let d = 0; const a = i * DIM, b = j * DIM;
  for (let t = 0; t < DIM; t++) d += EMB[a + t] * EMB[b + t];
  return d * INV;
}

// score every row of E (catalog or live) for ONE person whose picks are catalog rows `picks`
function personScores(E, H, C, rows, picks, genreW) {
  const m = picks.length, out = new Float32Array(rows);
  const w = new Float32Array(GN);
  for (const p of picks) for (let g = 0; g < GN; g++) w[g] += GH[p * GN + g] / m;
  for (let i = 0; i < rows; i++) {
    let sum = 0, mx = -Infinity; const bi = i * DIM;
    for (let k = 0; k < m; k++) {
      const pj = picks[k] * DIM; let d = 0;
      for (let t = 0; t < DIM; t++) d += E[bi + t] * EMB[pj + t];
      d *= INV; sum += d; if (d > mx) mx = d;
    }
    let s = 0.5 * (sum / m) + 0.5 * mx;
    if (genreW) {
      let gs = 0; const gb = i * GN;
      for (let g = 0; g < GN; g++) if (H[gb + g]) gs += w[g];
      s += genreW * gs / Math.max(C[i], 1);
    }
    out[i] = s;
  }
  return out;
}

function combine(list) {
  if (list.length === 1) return list[0];
  const a = list[0], b = list[1], out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = 0.5 * (a[i] + b[i]) + 0.5 * Math.min(a[i], b[i]);
  return out;
}

// greedy choice from the top `pool`: score - similarity to already chosen - overuse of one pick as the reason
function select(scores, picks, n, lam = 0.15, quotaPen = 0.04, pool = 100) {
  const order = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > -Infinity) order.push(i);
  order.sort((a, b) => scores[b] - scores[a]);
  const cand = order.slice(0, pool);
  const near = new Map();
  for (const i of cand) {
    let best = 0, bd = -Infinity;
    picks.forEach((p, k) => { const d = dotCat(i, p); if (d > bd) { bd = d; best = k; } });
    near.set(i, best);
  }
  const used = new Map(), chosen = [], remaining = cand.slice();
  while (remaining.length && chosen.length < n) {
    let bestI = -1, bestV = -Infinity;
    for (const i of remaining) {
      let v = scores[i] - quotaPen * (used.get(near.get(i)) || 0);
      if (chosen.length) {
        let mx = -Infinity;
        for (const c of chosen) { const d = dotCat(i, c); if (d > mx) mx = d; }
        v -= lam * mx;
      }
      if (v > bestV) { bestV = v; bestI = i; }
    }
    chosen.push(bestI);
    remaining.splice(remaining.indexOf(bestI), 1);
    used.set(near.get(bestI), (used.get(near.get(bestI)) || 0) + 1);
  }
  return chosen;
}

/* ------------------------------------------------------------------ explanations */
function nearestPick(vecFn, picks) {
  let best = picks[0], bd = -Infinity;
  for (const p of picks) { const d = vecFn(p); if (d > bd) { bd = d; best = p; } }
  return best;
}

function sharedGenres(i, j) {
  const out = [];
  for (let g = 0; g < GN; g++) if (GH[i * GN + g] && GH[j * GN + g]) out.push(GENRES[g]);
  return out.slice(0, 3);
}

function whyCatalog(i, groups) {
  if (groups.length === 1) {
    const g = groups[0], j = nearestPick(p => dotCat(i, p), g);
    const bits = [`Because you liked <b>${esc(M.title[j])}</b>`];
    const sg = sharedGenres(i, j);
    const liked = new Set(); g.forEach(p => kws(p).forEach(k => liked.add(k)));
    const sk = [...kws(i)].filter(k => liked.has(k)).sort().slice(0, 3);
    if (sg.length) bits.push('genres: ' + sg.join(', '));
    if (sk.length) bits.push('themes: ' + sk.join(', '));
    return bits.join(' · ');
  }
  const names = groups.map((g, k) => `${'AB'[k]}'s <b>${esc(M.title[nearestPick(p => dotCat(i, p), g)])}</b>`);
  return 'Bridges ' + names.join(' and ');
}

function dotLiveCat(li, p) {
  let d = 0; const a = li * DIM, b = p * DIM;
  for (let t = 0; t < DIM; t++) d += LIVE.emb[a + t] * EMB[b + t];
  return d * INV;
}

/* ------------------------------------------------------------------ recommend */
function franchiseBlockedSet(allPicks) { return new Set(allPicks.map(i => M.fk[i])); }

function recommend() {
  const groups = S.two ? [S.likedA, S.likedB] : [S.likedA];
  if (groups.some(g => g.length === 0)) return null;
  const all = [...new Set(groups.flat())];
  const blocked = S.hideFr ? franchiseBlockedSet(all) : new Set();
  const watched = new Set(S.watched);
  const picked = new Set(all);

  const scores = combine(groups.map(g => personScores(EMB, GH, GC, N, g, S.genreW)));
  const pop = S.popW;
  for (let i = 0; i < N; i++) {
    let ok = M.year[i] >= S.minYear && (S.maxRt >= 300 || M.runtime[i] <= S.maxRt || M.runtime[i] <= 0);
    if (ok && S.genres.size) {
      ok = false;
      for (const g of S.genres) { const k = GENRES.indexOf(g); if (k >= 0 && GH[i * GN + k]) { ok = true; break; } }
    }
    if (ok && (picked.has(i) || watched.has(i) || blocked.has(M.fk[i]))) ok = false;
    scores[i] = ok ? scores[i] + pop * POP[i] : -Infinity;
  }
  const items = select(scores, all, S.n).map(i => ({
    row: i, title: M.title[i], year: M.year[i], poster: M.poster[i], rating: M.rating[i],
    why: whyCatalog(i, groups), overview: M.overview[i], badge: ''
  }));

  let live = [];
  if (LIVE) {
    const ls = combine(groups.map(g => personScores(LIVE.emb, LIVE.H, LIVE.C, LIVE.n, g, S.genreW)));
    const seenIds = new Set([...all, ...S.watched].map(i => M.id[i]));
    const order = [];
    for (let i = 0; i < LIVE.n; i++) {
      const m = LIVE.movies[i];
      let ok = m.year >= S.minYear && !seenIds.has(m.id) && !blocked.has(m.fk);
      if (ok && S.genres.size) ok = m.genres.split(',').some(g => S.genres.has(g.trim()));
      if (ok) order.push(i);
    }
    order.sort((a, b) => ls[b] - ls[a]);
    live = order.slice(0, 6).map(i => {
      const m = LIVE.movies[i];
      const why = groups.length === 1
        ? `Close to <b>${esc(M.title[nearestPick(p => dotLiveCat(i, p), groups[0])])}</b>`
        : 'A fresh pick for the two of you';
      return { row: -1, title: m.title, year: m.year, poster: m.poster, rating: m.rating, why,
               overview: m.overview, badge: m.sources.join(' + ') };
    });
  }
  return { items, live };
}

/* ------------------------------------------------------------------ rendering */
function card(it, rank, canAct) {
  const ov = it.overview.length > 150 ? esc(it.overview.slice(0, 150)) + '…' : esc(it.overview);
  const img = it.poster ? `<img src="${IMG}${esc(it.poster)}" alt="" loading="lazy" onerror="this.style.display='none'">` : '';
  const acts = canAct
    ? `<div class="acts"><button data-act="seen" data-row="${it.row}">✓ Seen it</button><button data-act="like" data-row="${it.row}">＋ I love it</button></div>` : '';
  return `<article class="card"><div class="poster">${img}<span class="rank">No. ${rank}</span></div>
    <div class="body"><div class="t">${esc(it.title)} <span class="y">(${it.year || '?'})</span></div>
    <div class="meta">★ ${Number(it.rating).toFixed(1)}${it.badge ? `<span class="badge">${esc(it.badge)}</span>` : ''}</div>
    <div class="why">${it.why}</div><div class="ov">${ov}</div>${acts}</div></article>`;
}

function render() {
  const out = $('#results');
  const groups = S.two ? [S.likedA, S.likedB] : [S.likedA];
  if (groups.some(g => g.length === 0)) {
    out.innerHTML = `<div class="empty">🍿 ${S.two ? 'Both of you need to pick at least one movie.' : 'Pick a movie you love above to get started.'}</div>`;
    return;
  }
  const r = recommend();
  if (!r.items.length) { out.innerHTML = '<div class="empty">No matches. Try loosening the filters.</div>'; return; }
  let html = `<div class="section-title"><h2>🎯 Top picks</h2></div><div class="grid">${r.items.map((it, k) => card(it, k + 1, true)).join('')}</div>`;
  if (r.live.length) {
    html += `<div class="section-title"><h2>🎟️ In theaters &amp; trending</h2><span class="asof">snapshot from ${esc(LIVE.asof)}</span></div>
      <div class="grid">${r.live.map((it, k) => card(it, k + 1, false)).join('')}</div>`;
  }
  out.innerHTML = html;
}

/* ------------------------------------------------------------------ pickers (autocomplete + chips) */
function renderChips(key) {
  const root = $(`.picker[data-key="${key}"] .chips`);
  root.innerHTML = S[key].map(r => `<span class="chip">${esc(M.title[r])} <small>${M.year[r]}</small><button data-key="${key}" data-row="${r}" aria-label="Remove">×</button></span>`).join('');
}

function addPick(key, row) {
  const limit = key === 'watched' ? 400 : MAX_PICKS;
  if (S[key].includes(row) || S[key].length >= limit) return;
  S[key].push(row); renderChips(key); changed();
}

function initPicker(el) {
  const key = el.dataset.key, input = $('input', el), list = $('.sugg', el);
  let active = -1, shown = [];
  const hide = () => { list.hidden = true; active = -1; };
  const show = () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) return hide();
    shown = [];
    for (let i = 0; i < N && shown.length < 8; i++) if (TLC[i].includes(q) && !S[key].includes(i)) shown.push(i);
    list.innerHTML = shown.map((r, k) => `<li data-row="${r}" class="${k === active ? 'on' : ''}">${esc(M.title[r])}<small>${M.year[r]}</small></li>`).join('')
      || '<li style="cursor:default">No matches</li>';
    list.hidden = false;
  };
  input.addEventListener('input', () => { active = -1; show(); });
  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' && shown.length) { active = (active + 1) % shown.length; show(); e.preventDefault(); }
    else if (e.key === 'ArrowUp' && shown.length) { active = (active - 1 + shown.length) % shown.length; show(); e.preventDefault(); }
    else if (e.key === 'Enter' && shown.length) { addPick(key, shown[Math.max(active, 0)]); input.value = ''; hide(); e.preventDefault(); }
    else if (e.key === 'Escape') hide();
    else if (e.key === 'Backspace' && !input.value && S[key].length) { S[key].pop(); renderChips(key); changed(); }
  });
  list.addEventListener('mousedown', e => {
    const li = e.target.closest('li[data-row]');
    if (li) { addPick(key, +li.dataset.row); input.value = ''; hide(); e.preventDefault(); }
  });
  input.addEventListener('blur', () => setTimeout(hide, 120));
  el.addEventListener('click', e => {
    const b = e.target.closest('.chip button');
    if (b) { S[key] = S[key].filter(r => r !== +b.dataset.row); renderChips(key); changed(); }
  });
}

/* ------------------------------------------------------------------ state, url, storage */
let timer = null;
function changed() {
  persist();
  clearTimeout(timer);
  timer = setTimeout(render, 120);
}

function persist() {
  safeLS.set(LS_WATCHED, JSON.stringify(S.watched.map(r => M.id[r])));
  const ids = a => a.map(r => M.id[r]).join(',');
  const p = [];
  if (S.likedA.length) p.push('a=' + ids(S.likedA));
  if (S.two) { p.push('two=1'); if (S.likedB.length) p.push('b=' + ids(S.likedB)); }
  try { history.replaceState(null, '', p.length ? '#' + p.join('&') : location.pathname + location.search); } catch (e) { /* ignore */ }
}

function restore() {
  const toRows = s => (s || '').split(',').map(x => ID2ROW.get(+x)).filter(r => r !== undefined).slice(0, MAX_PICKS);
  try {
    const w = JSON.parse(safeLS.get(LS_WATCHED) || '[]');
    S.watched = w.map(id => ID2ROW.get(id)).filter(r => r !== undefined);
  } catch (e) { S.watched = []; }
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  S.likedA = toRows(h.get('a'));
  S.likedB = toRows(h.get('b'));
  S.two = h.get('two') === '1';
}

/* ------------------------------------------------------------------ controls */
function initControls() {
  const bindRange = (id, outId, key, fmt, parse = Number) => {
    const el = $('#' + id), out = $('#' + outId);
    const upd = () => { S[key] = parse(el.value); out.textContent = fmt(S[key]); };
    el.addEventListener('input', () => { upd(); changed(); });
    upd();
  };
  bindRange('minYear', 'oYear', 'minYear', v => v);
  bindRange('maxRt', 'oRt', 'maxRt', v => v >= 300 ? 'any' : v + ' min');
  bindRange('nPicks', 'oN', 'n', v => v, v => parseInt(v, 10));
  bindRange('popW', 'oPop', 'popW', v => v.toFixed(2));
  bindRange('genreW', 'oGen', 'genreW', v => v.toFixed(2));
  $('#hideFr').addEventListener('change', e => { S.hideFr = e.target.checked; changed(); });

  const gc = $('#genreChips');
  gc.innerHTML = GENRES.map(g => `<button type="button" class="g" data-g="${esc(g)}">${esc(g)}</button>`).join('');
  gc.addEventListener('click', e => {
    const b = e.target.closest('.g'); if (!b) return;
    const g = b.dataset.g;
    S.genres.has(g) ? S.genres.delete(g) : S.genres.add(g);
    b.classList.toggle('on', S.genres.has(g)); changed();
  });

  const two = $('#two');
  const applyTwo = () => {
    $('#colB').hidden = !S.two;
    $('#lblA').firstChild.textContent = S.two ? 'Person A loves ' : 'Movies you love ';
  };
  two.checked = S.two; applyTwo();
  two.addEventListener('change', () => { S.two = two.checked; applyTwo(); changed(); });

  $('#go').addEventListener('click', () => { clearTimeout(timer); render(); $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  $('#share').addEventListener('click', async () => {
    persist();
    try { await navigator.clipboard.writeText(location.href); $('#status').textContent = 'Link copied ✓'; }
    catch (e) { window.prompt('Copy this link:', location.href); }
    setTimeout(() => { $('#status').textContent = ''; }, 2500);
  });

  $('#results').addEventListener('click', e => {
    const b = e.target.closest('button[data-act]'); if (!b) return;
    const row = +b.dataset.row;
    if (b.dataset.act === 'seen') {
      if (!S.watched.includes(row)) S.watched.push(row);
      S.likedA = S.likedA.filter(r => r !== row); S.likedB = S.likedB.filter(r => r !== row);
      renderChips('watched'); renderChips('likedA'); renderChips('likedB'); changed();
    } else if (b.dataset.act === 'like') {
      addPick(S.two && S.likedA.length > 0 && S.likedB.length === 0 ? 'likedB' : 'likedA', row);
    }
  });
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  const results = $('#results');
  results.innerHTML = '<div class="loading">🎞️ Loading the film library…</div>';
  try {
    await loadData();
  } catch (e) {
    results.innerHTML = `<div class="empty">Couldn't load the movie data (${esc(e.message)}).<br>If you opened this file directly, serve the folder instead, e.g. <code>python -m http.server</code>.</div>`;
    return;
  }
  restore();
  initControls();
  document.querySelectorAll('.picker').forEach(initPicker);
  ['likedA', 'likedB', 'watched'].forEach(renderChips);
  render();
}

if (typeof document !== 'undefined') boot();
})();
