// Offline app shell. Model weights for NLLB are cached separately by Transformers.js.
const CACHE = "translator-v1";
const SHELL = ["./", "index.html", "styles.css", "app.js", "js/icons.js", "js/engines.js", "js/german.js",
  "nllb-worker.js", "manifest.webmanifest", "icons/icon.svg", "icons/icon-192.png",
  "model/vocab.json", "model/translator.onnx"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  const cdn = url.hostname === "cdn.jsdelivr.net" || url.hostname.endsWith("fonts.gstatic.com") || url.hostname === "fonts.googleapis.com";
  if (e.request.method !== "GET" || (url.origin !== location.origin && !cdn)) return;
  // stale-while-revalidate
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(e.request);
    const net = fetch(e.request).then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; }).catch(() => hit);
    return hit || net;
  }));
});
