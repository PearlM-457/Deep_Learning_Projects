// ============================================================================
// Translation engines
//   online : MyMemory public API (no key)                  — any pair
//   nllb   : Meta NLLB-200 in a Web Worker (Transformers.js) — any pair, offline
//   mine   : my own Seq2Seq Transformer (ONNX Runtime Web)  — EN → UZ only
// ============================================================================

export const LANGS = {
  en: { name: "English", nllb: "eng_Latn", tts: "en-US", asr: "en-US" },
  uz: { name: "O'zbekcha", nllb: "uzn_Latn", tts: "uz-UZ", asr: "uz-UZ" },
  de: { name: "Deutsch", nllb: "deu_Latn", tts: "de-DE", asr: "de-DE" },
};

// Split text into sentence-ish chunks (keeps punctuation) so each request stays small.
export function splitSentences(text, max = 400) {
  const parts = text.match(/[^.!?\n]+[.!?]*\s*|\n+/g) || [text];
  const out = [];
  let cur = "";
  for (const p of parts) {
    if ((cur + p).length > max && cur) { out.push(cur); cur = ""; }
    cur += p;
  }
  if (cur) out.push(cur);
  return out;
}

// ---------------------------------------------------------------- online
export const online = {
  id: "online",
  label: "Online · MyMemory",
  supports: () => true,
  ready: () => true,
  async translate(text, src, tgt) {
    const chunks = splitSentences(text, 450);
    const out = [];
    for (const c of chunks) {
      if (!c.trim()) { out.push(c); continue; }
      const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(c.trim())}&langpair=${src}|${tgt}`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(`Online engine error (${r.status})`);
      const j = await r.json();
      const t = j?.responseData?.translatedText ?? "";
      if (j.responseStatus && Number(j.responseStatus) !== 200) throw new Error(t || "Online engine error");
      if (/MYMEMORY WARNING/i.test(t)) throw new Error("Daily free quota reached — switch to On-device AI.");
      out.push(decodeEntities(t) + (/\s$/.test(c) ? " " : ""));
    }
    return { text: out.join("").trim() };
  },
};

function decodeEntities(s) {
  const el = document.createElement("textarea");
  el.innerHTML = s;
  return el.value;
}

// ---------------------------------------------------------------- NLLB (worker)
export const nllb = {
  id: "nllb",
  label: "On-device · NLLB-200",
  worker: null,
  state: "idle", // idle | loading | ready | error
  pending: new Map(),
  seq: 0,
  onProgress: null,
  onState: null,
  supports: () => true,
  ready() { return this.state === "ready"; },

  async isCached() {
    try {
      if (!("caches" in window)) return false;
      const c = await caches.open("transformers-cache");
      const keys = await c.keys();
      return keys.some((k) => k.url.includes("nllb-200") && k.url.includes("decoder_model_merged"));
    } catch { return false; }
  },

  load() {
    if (this.worker) return this.readyPromise;
    this.setState("loading");
    this.worker = new Worker(new URL("../nllb-worker.js", import.meta.url), { type: "module" });
    this.readyPromise = new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") this.onProgress?.(m);
        else if (m.type === "ready") { this.setState("ready"); resolve(); }
        else if (m.type === "error" && m.id == null) { this.setState("error"); this.worker = null; reject(new Error(m.message)); }
        else if (m.type === "result" || m.type === "error") {
          const p = this.pending.get(m.id);
          if (!p) return;
          this.pending.delete(m.id);
          m.type === "result" ? p.resolve({ text: m.text }) : p.reject(new Error(m.message));
        }
      };
      this.worker.onerror = (e) => { this.setState("error"); this.worker = null; reject(new Error(e.message || "Worker failed")); };
    });
    this.worker.postMessage({ type: "load" });
    return this.readyPromise;
  },

  setState(s) { this.state = s; this.onState?.(s); },

  async translate(text, src, tgt) {
    if (!this.ready()) await this.load();
    const chunks = splitSentences(text, 300).filter((c) => c.trim());
    const results = [];
    for (const c of chunks) {
      const id = ++this.seq;
      const p = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
      this.worker.postMessage({ type: "translate", id, text: c.trim(), src: LANGS[src].nllb, tgt: LANGS[tgt].nllb });
      results.push((await p).text);
    }
    return { text: results.join(" ") };
  },
};

// ---------------------------------------------------------------- my model (ONNX)
const APOS = /[ʻʼ’‘`]/g;
const TOKEN_RE = /[\p{L}\p{N}_]+(?:'[\p{L}\p{N}_]+)*'?|[^\p{L}\p{N}_\s]/gu;
export const tokenize = (s) => (s.replace(APOS, "'").toLowerCase().match(TOKEN_RE) || []);

function detok(words) {
  let s = words.join(" ").replace(/ ([.,?!;:])/g, "$1");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const mine = {
  id: "mine",
  label: "My model · Seq2Seq",
  session: null,
  vocab: null,
  stoi: null,
  supports: (src, tgt) => src === "en" && tgt === "uz",
  ready() { return !!this.session; },

  async load() {
    if (this.session) return;
    if (this._loading) return this._loading;
    this._loading = (async () => {
      for (let i = 0; i < 100 && !window.ort; i++) await new Promise((r) => setTimeout(r, 100));
      if (!window.ort) throw new Error("ONNX Runtime failed to load");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
      ort.env.wasm.numThreads = 1;
      const base = new URL("../model/", import.meta.url);
      this.vocab = await (await fetch(new URL("vocab.json", base))).json();
      this.stoi = new Map(this.vocab.itos.map((w, i) => [w, i]));
      this.proper = new Set(this.vocab.proper || []);
      this.session = await ort.InferenceSession.create(new URL("tarjimon.onnx", base).href, { executionProviders: ["wasm"] });
    })();
    try { await this._loading; } finally { this._loading = null; }
  },

  // The ONNX graph has static shapes [1, T]. Thanks to the causal mask, padding
  // tgt_in with <pad> doesn't change the logits of the real positions.
  async logits(srcIds, tgtIds) {
    const { T, pad_id } = this.vocab;
    const tgt = tgtIds.slice(0, T);
    while (tgt.length < T) tgt.push(pad_id);
    const t = (a) => new ort.Tensor("int64", BigInt64Array.from(a.map(BigInt)), [1, a.length]);
    const out = await this.session.run({ src: t(srcIds), tgt_in: t(tgt) });
    return out.logits; // [1, T, V]
  },

  lastRow(logits, row) {
    const V = logits.dims[2];
    return logits.data.subarray(row * V, (row + 1) * V);
  },

  logSoftmax(row) {
    let m = -Infinity;
    for (const v of row) if (v > m) m = v;
    let s = 0;
    for (const v of row) s += Math.exp(v - m);
    const lse = m + Math.log(s);
    return Array.from(row, (v) => v - lse);
  },

  async translateSentence(sentence, beam = 4) {
    const { T, pad_id, sos_id, eos_id, unk_id } = this.vocab;
    const toks = tokenize(sentence).slice(0, T);
    const unknown = toks.filter((w) => !this.stoi.has(w));
    const src = toks.map((w) => this.stoi.get(w) ?? unk_id);
    while (src.length < T) src.push(pad_id);

    let beams = [{ seq: [sos_id], score: 0 }];
    const done = [];
    for (let step = 0; step < T - 1; step++) {
      const cand = [];
      for (const b of beams) {
        const lg = await this.logits(src, b.seq);
        const lp = this.logSoftmax(this.lastRow(lg, b.seq.length - 1));
        const idx = lp.map((v, i) => [v, i]).sort((a, c) => c[0] - a[0]).slice(0, beam);
        for (let [v, i] of idx) {
          if (i === b.seq[b.seq.length - 1]) v -= 2.0; // repetition penalty
          cand.push({ seq: [...b.seq, i], score: b.score + v });
        }
      }
      cand.sort((a, c) => c.score - a.score);
      beams = [];
      for (const c of cand) {
        (c.seq[c.seq.length - 1] === eos_id ? done : beams).push(c);
        if (beams.length === beam) break;
      }
      if (done.length >= beam || !beams.length) break;
    }
    const all = done.concat(beams);
    const best = all.reduce((a, c) => (c.score / c.seq.length ** 0.7 > a.score / a.seq.length ** 0.7 ? c : a));

    // per-token confidence from one teacher-forced pass
    const tgtIn = best.seq.at(-1) === eos_id ? best.seq.slice(0, -1) : best.seq;
    const lg = await this.logits(src, tgtIn);
    const words = [];
    const conf = [];
    for (let k = 1; k < best.seq.length; k++) {
      const id = best.seq[k];
      if (id === eos_id || id === pad_id) break;
      const row = this.logSoftmax(this.lastRow(lg, k - 1));
      const w = this.vocab.itos[id];
      words.push(this.proper.has(w) ? w.charAt(0).toUpperCase() + w.slice(1) : w);
      conf.push(Math.exp(row[id]));
    }
    return { words, conf, unknown };
  },

  async translate(text) {
    await this.load();
    const sents = (text.match(/[^.!?\n]+[.!?]*/g) || [text]).map((s) => s.trim()).filter(Boolean);
    const out = { text: "", tokens: [], unknown: [] };
    for (const s of sents) {
      const r = await this.translateSentence(s);
      const w = [...r.words];
      if (w.length) w[0] = w[0].charAt(0).toUpperCase() + w[0].slice(1);
      out.tokens.push(...w.map((word, i) => ({ word, p: r.conf[i] })));
      out.unknown.push(...r.unknown);
      out.text += (out.text ? " " : "") + detok(r.words);
    }
    return out;
  },
};

export const ENGINES = { online, nllb, mine };
