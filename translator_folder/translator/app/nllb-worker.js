// Web Worker: runs Meta's NLLB-200 (distilled 600M, 8-bit) with Transformers.js.
// The model files (~900 MB) are downloaded once from the Hugging Face Hub and
// cached by the browser (Cache Storage "transformers-cache").
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL = "Xenova/nllb-200-distilled-600M";
let translator = null;

self.onmessage = async (e) => {
  const { type, id, text, src, tgt } = e.data;
  try {
    if (type === "load") {
      if (!translator) {
        translator = await pipeline("translation", MODEL, {
          dtype: "q8",
          progress_callback: (p) => self.postMessage({ type: "progress", ...p }),
        });
      }
      self.postMessage({ type: "ready" });
    } else if (type === "translate") {
      const out = await translator(text, { src_lang: src, tgt_lang: tgt, max_new_tokens: 256 });
      self.postMessage({ type: "result", id, text: out[0].translation_text });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: type === "translate" ? id : null, message: String(err?.message || err) });
  }
};
