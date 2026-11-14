import { hydrateIcons, svg } from "./js/icons.js";
import { ENGINES, LANGS, nllb, mine, online } from "./js/engines.js";
import { annotate, GENDER_ARTICLE, phraseOfTheDay, EXAMPLES } from "./js/german.js";

// ------------------------------------------------------------------ storage
const store = {
  get(k, d) { try { const v = localStorage.getItem("translator." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("translator." + k, JSON.stringify(v)); } catch {} },
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  engine: store.get("engine", "online"),
  src: store.get("src", "en"),
  tgt: store.get("tgt", "uz"),
  words: store.get("words", []),
  history: store.get("history", []),
  last: null, // {src, tgt, srcLang, tgtLang, engine}
  savedLangs: null,
  reqId: 0,
};

// ------------------------------------------------------------------ UI refs
const el = {
  srcText: $("#srcText"), tgtText: $("#tgtText"), srcLang: $("#srcLang"), tgtLang: $("#tgtLang"),
  count: $("#charCount"), engineTag: $("#engineTag"), legend: $("#deLegend"), mineMeta: $("#mineMeta"),
  nllbPanel: $("#nllbPanel"), nllbStatus: $("#nllbStatus"), nllbProgress: $("#nllbProgress"),
  pop: $("#popover"), toast: $("#toast"),
};

function toast(msg, ms = 2600) {
  el.toast.textContent = msg;
  el.toast.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.toast.classList.remove("show"), ms);
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".view").forEach((v) => (v.hidden = v.dataset.view !== name));
  if (name === "words") renderWords();
  if (name === "history") renderHistory();
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------ theme
function applyThemeIcon() {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("#themeBtn i").dataset.icon = dark ? "sun" : "moon";
  hydrateIcons($("#themeBtn"));
}
$("#themeBtn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.set("theme", next); try { localStorage.setItem("translator.theme", next); } catch {}
  applyThemeIcon();
});

// ------------------------------------------------------------------ engines
function setEngine(id, { silent = false } = {}) {
  state.engine = id;
  store.set("engine", id);
  $$(".engine").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.engine === id)));
  el.nllbPanel.hidden = !(id === "nllb" && !nllb.ready());
  // my model only supports EN -> UZ
  if (id === "mine") {
    if (!(state.src === "en" && state.tgt === "uz")) {
      state.savedLangs = [state.src, state.tgt];
      setLangs("en", "uz");
      if (!silent) toast("My model translates English → O'zbekcha only");
    }
    el.srcLang.disabled = el.tgtLang.disabled = true;
    $("#swapBtn").disabled = true;
    mine.load().then(() => (el.mineStatus.textContent = "Ready · EN→UZ · " + scoreText())).catch((e) => toast(e.message));
  } else {
    el.srcLang.disabled = el.tgtLang.disabled = false;
    $("#swapBtn").disabled = false;
    if (state.savedLangs) { setLangs(...state.savedLangs); state.savedLangs = null; }
  }
  if (id === "nllb" && !nllb.ready()) {
    nllb.isCached().then((cached) => { if (cached && state.engine === "nllb") startNllb(); });
  }
  el.engineTag.textContent = ENGINES[id].label;
  if (!silent && el.srcText.value.trim()) doTranslate();
}
el.mineStatus = $("#mineStatus");
$$(".engine").forEach((b) => b.addEventListener("click", () => setEngine(b.dataset.engine)));

function scoreText() {
  const s = mine.vocab?.scores;
  return s ? `BLEU ${s.bleu.toFixed(1)}` : "";
}

// NLLB download / progress
const fileProgress = new Map();
nllb.onProgress = (p) => {
  if (p.status === "progress" && p.file) fileProgress.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
  if (p.status === "done" && p.file) { const f = fileProgress.get(p.file); if (f) f.loaded = f.total; }
  let loaded = 0, total = 0;
  fileProgress.forEach((f) => { loaded += f.loaded; total += f.total; });
  const pct = total ? Math.min(100, (loaded / total) * 100) : 0;
  el.nllbProgress.hidden = false;
  $(".bar", el.nllbProgress).style.width = pct.toFixed(1) + "%";
  $(".label", el.nllbProgress).textContent = total
    ? `Downloading ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB · ${pct.toFixed(0)}%`
    : "Preparing…";
};
nllb.onState = (s) => {
  const map = { idle: "NLLB-200 · private · offline", loading: "Loading model…", ready: "Ready · runs offline", error: "Failed — tap to retry" };
  el.nllbStatus.textContent = map[s];
  $("#nllbLoadBtn").disabled = s === "loading";
  if (s === "ready") {
    el.nllbPanel.hidden = true;
    toast("On-device AI is ready ✓");
    if (state.engine === "nllb" && el.srcText.value.trim()) doTranslate();
  }
  if (s === "loading") { el.nllbProgress.hidden = false; $(".label", el.nllbProgress).textContent = "Loading…"; }
};
function startNllb() {
  nllb.load().catch((e) => { toast("Could not load NLLB: " + e.message, 5000); el.nllbPanel.hidden = false; });
}
$("#nllbLoadBtn").addEventListener("click", startNllb);

// ------------------------------------------------------------------ languages
function setLangs(src, tgt) {
  state.src = src; state.tgt = tgt;
  el.srcLang.value = src; el.tgtLang.value = tgt;
  store.set("src", src); store.set("tgt", tgt);
  renderExamples();
  el.legend.hidden = tgt !== "de";
}
el.srcLang.addEventListener("change", () => {
  let tgt = state.tgt;
  if (el.srcLang.value === tgt) tgt = state.src;
  setLangs(el.srcLang.value, tgt); doTranslate();
});
el.tgtLang.addEventListener("change", () => {
  let src = state.src;
  if (el.tgtLang.value === src) src = state.tgt;
  setLangs(src, el.tgtLang.value); doTranslate();
});
function swap() {
  if (state.engine === "mine") return;
  const out = state.last?.tgt || "";
  setLangs(state.tgt, state.src);
  if (out) { el.srcText.value = out; updateCount(); }
  doTranslate();
}
$("#swapBtn").addEventListener("click", swap);

// ------------------------------------------------------------------ translate
let debounce;
function updateCount() { el.count.textContent = `${el.srcText.value.length} / 1000`; }
el.srcText.addEventListener("input", () => {
  updateCount();
  clearTimeout(debounce);
  const eng = ENGINES[state.engine];
  if (eng.ready() || state.engine === "mine") debounce = setTimeout(doTranslate, state.engine === "online" ? 800 : 500);
});
el.srcText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doTranslate(); }
});
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "s") { e.preventDefault(); swap(); }
  if (e.key === "Escape") closePop();
});
$("#translateBtn").addEventListener("click", doTranslate);
$("#clearBtn").addEventListener("click", () => {
  el.srcText.value = ""; updateCount(); state.last = null; el.mineMeta.hidden = true;
  el.tgtText.innerHTML = '<span class="placeholder">Translation appears here. Tap any word to look it up and save it.</span>';
  el.srcText.focus();
});

async function doTranslate() {
  const text = el.srcText.value.trim();
  if (!text) return;
  const id = ++state.reqId;
  const engine = ENGINES[state.engine];
  if (state.engine === "nllb" && !nllb.ready()) {
    el.nllbPanel.hidden = false;
    if (nllb.state !== "loading") { toast("Download the on-device model first (or pick Online)"); return; }
  }
  el.tgtText.classList.add("loading");
  el.tgtText.innerHTML = `<span class="placeholder dots">Translating</span>`;
  $("#translateBtn").disabled = true;
  try {
    const res = await engine.translate(text, state.src, state.tgt);
    if (id !== state.reqId) return;
    state.last = { src: text, tgt: res.text, srcLang: state.src, tgtLang: state.tgt, engine: state.engine };
    renderOutput(res);
    addHistory(state.last);
  } catch (e) {
    if (id !== state.reqId) return;
    el.tgtText.innerHTML = `<span class="placeholder">⚠️ ${esc(e.message || e)}</span>`;
  } finally {
    if (id === state.reqId) { el.tgtText.classList.remove("loading"); $("#translateBtn").disabled = false; }
  }
}

function renderOutput(res) {
  el.mineMeta.hidden = true;
  if (res.tokens) {
    // my model: color tokens by confidence
    el.tgtText.innerHTML = res.tokens.map((t, i) => {
      const punct = /^[.,?!;:]$/.test(t.word);
      const cls = t.p < 0.4 ? "conf-low" : t.p < 0.7 ? "conf-mid" : "";
      const span = punct ? esc(t.word) :
        `<span class="w ${cls}" data-w="${esc(t.word)}" title="confidence ${(t.p * 100).toFixed(0)}%">${esc(t.word)}</span>`;
      const next = res.tokens[i + 1];
      return span + (next && !/^[.,?!;:]$/.test(next.word) ? " " : "");
    }).join("");
    const avg = res.tokens.length ? res.tokens.reduce((a, t) => a + t.p, 0) / res.tokens.length : 0;
    const unk = [...new Set(res.unknown)];
    el.mineMeta.innerHTML = `<span>Avg. confidence <b>${(avg * 100).toFixed(0)}%</b></span>` +
      `<span>Beam search k=4</span>` +
      (unk.length ? `<span class="warn">Not in my vocabulary: <b>${unk.map(esc).join(", ")}</b></span>` : "") +
      `<span>wavy underline = unsure</span>`;
    el.mineMeta.hidden = false;
    return;
  }
  const toks = annotate(res.text, state.tgt);
  el.tgtText.innerHTML = toks.map((t) =>
    t.isWord ? `<span class="w ${t.cls}" data-w="${esc(t.t)}"${t.gender ? ` data-g="${t.gender}"` : ""}>${esc(t.t)}</span>` : esc(t.t)
  ).join("");
}

// ------------------------------------------------------------------ word popover
let popWord = null;
el.tgtText.addEventListener("click", (e) => {
  const w = e.target.closest(".w");
  if (!w || !state.last) return;
  openPop(w, w.dataset.w, state.last.tgtLang, w.dataset.g);
});

async function openPop(anchor, word, lang, gender) {
  const display = gender ? `${GENDER_ARTICLE[gender]} ${word}` : word;
  popWord = { word: display, lang, gender };
  $(".pop-word", el.pop).innerHTML = gender ? `<span class="${gender}">${esc(display)}</span>` : esc(display);
  const others = Object.keys(LANGS).filter((l) => l !== lang);
  const body = $(".pop-body", el.pop);
  body.innerHTML = others.map((l) => `<div class="pop-row"><span class="lang-pill">${l.toUpperCase()}</span><span data-l="${l}" class="dots"></span></div>`).join("");
  el.pop.hidden = false;
  placePop(anchor);
  const lookup = pickLookupEngine(lang);
  if (!lookup) {
    body.innerHTML += `<div class="pop-note">Word lookup needs the Online or On-device engine.</div>`;
    $$("[data-l]", body).forEach((s) => { s.classList.remove("dots"); s.textContent = "—"; });
    popWord.translations = {};
    return;
  }
  popWord.translations = {};
  await Promise.all(others.map(async (l) => {
    const span = $(`[data-l="${l}"]`, body);
    try {
      const r = await lookup.translate(display, lang, l);
      popWord.translations[l] = r.text;
      span.textContent = r.text;
    } catch { span.textContent = "—"; }
    span.classList.remove("dots");
  }));
}

function pickLookupEngine() {
  if (nllb.ready() && (state.engine === "nllb" || !navigator.onLine)) return nllb;
  if (navigator.onLine) return online;
  return nllb.ready() ? nllb : null;
}

function placePop(anchor) {
  const r = anchor.getBoundingClientRect();
  const pw = el.pop.offsetWidth, ph = el.pop.offsetHeight;
  let left = Math.min(Math.max(12, r.left + r.width / 2 - pw / 2), innerWidth - pw - 12);
  let top = r.bottom + 8;
  if (top + ph > innerHeight - 12) top = Math.max(12, r.top - ph - 8);
  el.pop.style.left = left + "px"; el.pop.style.top = top + "px";
}
function closePop() { el.pop.hidden = true; popWord = null; }
el.pop.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "close") closePop();
  if (act === "speak" && popWord) speak(popWord.word, popWord.lang);
  if (act === "save" && popWord) {
    const back = Object.entries(popWord.translations || {}).map(([l, t]) => `${l.toUpperCase()}: ${t}`).join(" · ");
    addWord(popWord.word, popWord.lang, back || "", "word");
    closePop();
  }
});
document.addEventListener("pointerdown", (e) => {
  if (!el.pop.hidden && !el.pop.contains(e.target) && !e.target.closest(".w")) closePop();
});
addEventListener("resize", closePop);

// ------------------------------------------------------------------ speech
function speak(text, lang) {
  if (!("speechSynthesis" in window)) return toast("Text-to-speech is not supported in this browser");
  const code = LANGS[lang].tts;
  const voices = speechSynthesis.getVoices();
  const voice = voices.find((v) => v.lang.replace("_", "-").toLowerCase().startsWith(code.slice(0, 2)));
  if (!voice && voices.length && lang === "uz") toast("No Uzbek voice on this device — trying anyway");
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = code; if (voice) u.voice = voice; u.rate = lang === "de" ? 0.9 : 1;
  speechSynthesis.speak(u);
}
if ("speechSynthesis" in window) speechSynthesis.getVoices();
$("#speakSrcBtn").addEventListener("click", () => el.srcText.value.trim() && speak(el.srcText.value, state.src));
$("#speakTgtBtn").addEventListener("click", () => state.last && speak(state.last.tgt, state.last.tgtLang));

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
$("#micBtn").addEventListener("click", () => {
  if (!SR) return toast("Voice input works in Chrome / Edge / Safari");
  if (rec) { rec.stop(); return; }
  rec = new SR();
  rec.lang = LANGS[state.src].asr; rec.interimResults = true; rec.continuous = false;
  const base = el.srcText.value ? el.srcText.value.trimEnd() + " " : "";
  rec.onresult = (e) => {
    el.srcText.value = base + [...e.results].map((r) => r[0].transcript).join("");
    updateCount();
  };
  rec.onend = () => { $("#micBtn").classList.remove("active"); rec = null; doTranslate(); };
  rec.onerror = (e) => toast("Mic: " + e.error);
  $("#micBtn").classList.add("active");
  rec.start();
});

// ------------------------------------------------------------------ copy / save
$("#copyBtn").addEventListener("click", async () => {
  if (!state.last) return;
  try { await navigator.clipboard.writeText(state.last.tgt); toast("Copied ✓"); } catch { toast("Copy failed"); }
});
$("#saveBtn").addEventListener("click", () => {
  if (!state.last) return;
  addWord(state.last.tgt, state.last.tgtLang, `${state.last.srcLang.toUpperCase()}: ${state.last.src}`, "sentence");
});

// ------------------------------------------------------------------ words + spaced repetition
const INTERVAL_DAYS = [0, 1, 2, 4, 8, 16];
function saveWords() { store.set("words", state.words); updateBadges(); }
function addWord(front, lang, back, kind) {
  if (state.words.some((w) => w.front.toLowerCase() === front.toLowerCase() && w.lang === lang)) return toast("Already in My words");
  state.words.unshift({ id: Date.now().toString(36), front, lang, back, kind, box: 0, due: Date.now(), added: Date.now() });
  saveWords();
  toast(`Saved “${front.length > 40 ? front.slice(0, 40) + "…" : front}” ★`);
}
const dueWords = () => state.words.filter((w) => w.due <= Date.now());
function updateBadges() {
  const n = state.words.length;
  $("#wordCount").hidden = !n; $("#wordCount").textContent = n;
  $("#dueCount").textContent = dueWords().length;
}

let boxFilter = "all";
$$("#boxFilter button").forEach((b) => b.addEventListener("click", () => {
  boxFilter = b.dataset.box; $$("#boxFilter button").forEach((x) => x.classList.toggle("on", x === b)); renderWords();
}));
$("#wordSearch").addEventListener("input", renderWords);

function renderWords() {
  const q = $("#wordSearch").value.toLowerCase();
  const list = state.words.filter((w) =>
    (boxFilter === "all" || (boxFilter === "new" ? w.box < 3 : w.box >= 3)) &&
    (!q || (w.front + " " + w.back).toLowerCase().includes(q)));
  $("#wordsEmpty").hidden = state.words.length > 0;
  $("#wordList").innerHTML = list.map((w) => {
    const ann = annotate(w.front, w.lang).map((t) => (t.cls ? `<span class="${t.cls}">${esc(t.t)}</span>` : esc(t.t))).join("");
    return `<li data-id="${w.id}">
      <span class="lang-pill">${w.lang.toUpperCase()}</span>
      <div class="wl-main"><div class="wl-front">${ann}</div><div class="wl-back">${esc(w.back)}</div></div>
      <div class="box-dots" title="Leitner box ${w.box}/5">${[1, 2, 3, 4, 5].map((i) => `<span class="${w.box >= i ? "on" : ""}"></span>`).join("")}</div>
      <button class="icon-btn sm" data-act="speak" title="Listen"><i data-icon="volume"></i></button>
      <button class="icon-btn sm" data-act="del" title="Delete"><i data-icon="trash"></i></button></li>`;
  }).join("");
  hydrateIcons($("#wordList"));
  updateBadges();
}
$("#wordList").addEventListener("click", (e) => {
  const li = e.target.closest("li"); const act = e.target.closest("[data-act]")?.dataset.act;
  if (!li || !act) return;
  const w = state.words.find((x) => x.id === li.dataset.id);
  if (act === "speak") speak(w.front, w.lang);
  if (act === "del") { state.words = state.words.filter((x) => x !== w); saveWords(); renderWords(); }
});

// review session
let queue = [], cur = null, reviewed = 0;
$("#reviewBtn").addEventListener("click", () => {
  queue = dueWords().sort(() => Math.random() - 0.5);
  if (!queue.length) return toast(state.words.length ? "Nothing due — come back later 🎉" : "Save some words first");
  reviewed = 0; $("#review").hidden = false; nextCard();
});
function nextCard() {
  cur = queue.shift();
  const fc = $("#flashcard");
  fc.classList.remove("flipped");
  if (!cur) { $("#review").hidden = true; toast(`Session done · ${reviewed} cards`); renderWords(); return; }
  $(".fc-front .fc-lang", fc).textContent = LANGS[cur.lang].name.toUpperCase();
  $(".fc-front .fc-text", fc).innerHTML = annotate(cur.front, cur.lang).map((t) => (t.cls ? `<span class="${t.cls}">${esc(t.t)}</span>` : esc(t.t))).join("");
  $(".fc-back .fc-lang", fc).textContent = "MEANING";
  $(".fc-back .fc-text", fc).textContent = cur.back || "—";
  $("#reviewProgress").textContent = `${queue.length + 1} left`;
}
$("#flashcard").addEventListener("click", () => $("#flashcard").classList.toggle("flipped"));
$("#flashcard").addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); $("#flashcard").classList.toggle("flipped"); } });
$("#fcSpeak").addEventListener("click", () => cur && speak(cur.front, cur.lang));
$("#fcGood").addEventListener("click", () => grade(true));
$("#fcAgain").addEventListener("click", () => grade(false));
$("#fcClose").addEventListener("click", () => { $("#review").hidden = true; renderWords(); });
function grade(ok) {
  if (!cur) return;
  reviewed++;
  if (ok) { cur.box = Math.min(5, cur.box + 1); cur.due = Date.now() + INTERVAL_DAYS[cur.box] * 864e5; }
  else { cur.box = 0; cur.due = Date.now(); queue.push(cur); }
  saveWords(); nextCard();
}

$("#exportBtn").addEventListener("click", () => {
  if (!state.words.length) return toast("No words to export yet");
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rows = state.words.map((w) => [q(w.front), q(w.back), q(`translator ${w.lang} ${w.kind}`)].join(","));
  const blob = new Blob(["﻿" + rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "translator-words.csv" });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast("Exported — import into Anki as CSV (Front, Back, Tags)", 4000);
});

// ------------------------------------------------------------------ history
function addHistory(h) {
  if (state.history[0]?.src === h.src && state.history[0]?.tgtLang === h.tgtLang) state.history.shift();
  state.history.unshift({ ...h, t: Date.now() });
  state.history = state.history.slice(0, 100);
  store.set("history", state.history);
}
function renderHistory() {
  $("#histEmpty").hidden = state.history.length > 0;
  $("#histList").innerHTML = state.history.map((h, i) => `<li data-i="${i}">
      <span class="lang-pill">${h.srcLang.toUpperCase()}→${h.tgtLang.toUpperCase()}</span>
      <div class="wl-main"><div class="wl-front">${esc(h.src)}</div><div class="wl-back">${esc(h.tgt)}</div></div>
      <span class="hist-meta">${new Date(h.t).toLocaleDateString()}<br>${esc(ENGINES[h.engine]?.label.split(" · ")[0] || "")}</span></li>`).join("");
}
$("#histList").addEventListener("click", (e) => {
  const li = e.target.closest("li"); if (!li) return;
  const h = state.history[+li.dataset.i];
  setEngine(h.engine, { silent: true }); setLangs(h.srcLang, h.tgtLang);
  el.srcText.value = h.src; updateCount();
  state.last = { ...h };
  renderOutput({ text: h.tgt });
  showTab("translate");
});
$("#clearHistBtn").addEventListener("click", () => {
  if (!state.history.length || !confirm("Clear translation history?")) return;
  state.history = []; store.set("history", []); renderHistory();
});

// ------------------------------------------------------------------ examples + phrase of the day
function renderExamples() {
  const ex = EXAMPLES[state.src] || [];
  $("#exampleChips").innerHTML = ex.map((s) => `<button class="chip">${esc(s)}</button>`).join("");
}
$("#exampleChips").addEventListener("click", (e) => {
  const c = e.target.closest(".chip"); if (!c) return;
  el.srcText.value = c.textContent; updateCount(); doTranslate();
});
function renderPotd() {
  const [de, en, uz] = phraseOfTheDay();
  const ann = annotate(de, "de").map((t) => (t.cls ? `<span class="${t.cls}">${esc(t.t)}</span>` : esc(t.t))).join("");
  $("#potd .potd-body").innerHTML = `
    <div class="de">${ann}</div>
    <div class="tr"><b>EN</b> ${esc(en)}<br><b>UZ</b> ${esc(uz)}</div>
    <div class="potd-actions">
      <button class="btn sm" data-act="speak"><i data-icon="volume"></i>Listen</button>
      <button class="btn sm" data-act="save"><i data-icon="star"></i>Save</button>
    </div>`;
  hydrateIcons($("#potd"));
  $("#potd").onclick = (e) => {
    const a = e.target.closest("[data-act]")?.dataset.act;
    if (a === "speak") speak(de, "de");
    if (a === "save") addWord(de, "de", `EN: ${en} · UZ: ${uz}`, "sentence");
  };
}

// ------------------------------------------------------------------ about: scores
async function renderScores() {
  try {
    const v = await (await fetch("model/vocab.json")).json();
    const s = v.scores;
    if (!s) return;
    $("#scoreRow").innerHTML = `<div class="score">Test BLEU<b>${s.bleu.toFixed(1)}</b></div>` +
      `<div class="score">chrF<b>${s.chrf.toFixed(1)}</b></div>` +
      `<div class="score">Exact match<b>${s.exact.toFixed(0)}%</b></div>`;
  } catch {}
}

// ------------------------------------------------------------------ PWA
let deferredPrompt = null;
addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferredPrompt = e; $("#installBtn").hidden = false; });
$("#installBtn").addEventListener("click", async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt(); await deferredPrompt.userChoice; deferredPrompt = null; $("#installBtn").hidden = true;
});
if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

// ------------------------------------------------------------------ init
hydrateIcons();
applyThemeIcon();
setLangs(state.src, state.tgt);
setEngine(state.engine, { silent: true });
renderPotd();
renderScores();
updateBadges();
updateCount();
const initialTab = location.hash.slice(1);
showTab(["translate", "words", "history", "about"].includes(initialTab) ? initialTab : "translate");
window.__translator = { state, mine, nllb, online };
