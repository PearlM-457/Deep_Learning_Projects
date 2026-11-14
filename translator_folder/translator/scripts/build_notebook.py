"""Builds notebooks/Translator_Seq2Seq.ipynb (homework notebook, Uzbek)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 🌐 Translator — Seq2Seq (Transformer) bilan Ingliz → O'zbek mini translator

**Mavzu:** Seq2Seq, Encoding va Decoding · **Turi:** Amaliy vazifa · **Muallif:** PearlyMe

<a href="https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/translator/blob/main/notebooks/Translator_Seq2Seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>

Ushbu notebookda:

| # | Vazifa | Bo'lim |
|---|--------|--------|
| 1 | Lug'at va tokenlar (`<pad>`, `<sos>`, `<eos>`, padding T gacha) | §2 |
| 2 | `tgt_in` va `tgt_out` — qo'lda misol | §3 |
| 3 | O'qitish: `CrossEntropyLoss(ignore_index=pad_id)`, `AdamW`, ≥30 epoch, loss grafigi | §4–5 |
| 4 | `translate()` — `<sos>` dan boshlab argmax, `<eos>` da to'xtash | §6 |
| 5 | Tajriba: `C`, `nhead`, qatlamlar soni | §8 |
| ⭐ | Qo'shimcha: 10 ta gapda xatolar tahlili, nega so'z takrorlanadi | §7 |
| ❓ | Savol: mask olib tashlansa nima bo'ladi? (amalda isbotlangan) | §9 |
| 🚀 | Ijodiy qism: BLEU/chrF baholash, yaxshilangan **V2** model, beam search, ONNX eksport → brauzerdagi translator | §10–11 |

> Berilgan `Seq2Seq` klassi **o'zgartirilmagan** holda ishlatiladi.
""")

md(r"""
## 0. Encoding va Decoding — qisqacha

Translator odamni tasavvur qiling: avval gapni **o'qib tushunadi** (encoding), keyin ma'noni **boshqa tilda so'zma-so'z yozadi** (decoding).

```
src  (ingliz gapi)        ──► embed + pos ──► ENCODER (2 qatlam) ──┐   "ma'no" (memory)
                                                                   ▼
tgt_in (<sos> + o'zbek)   ──► embed + pos ──► DECODER (2 qatlam) ──► Linear ──► keyingi so'z
                                   ▲
                       mask: kelajakni ko'rish taqiqlangan
```
""")

code(r"""
# Colab'da kerakli kutubxonalar (torch Colab'da bor)
!pip -q install sacrebleu

import os, re, csv, math, random, time, json
import torch, torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))
print("torch", torch.__version__)
""")

md(r"""
## 1. Ma'lumot

* **Manba:** o'zim yozgan ro'yxat (vazifa shartida ruxsat berilgan) — `scripts/make_dataset.py` yordamida yaratilgan.
  Generator o'zbek tilining **morfologiyasini qoidalar bilan** quradi: fe'l zamonlari (-yman, -dim, -yapman, -may, -moqchi, -a olaman),
  kelishiklar (-ni, -ga/-ka/-qa, -da), egalik qo'shimchalari (mushuk → mushu**g**im) va ~100 ta qo'lda yozilgan kundalik iboralar.
* **Hajm:** ~1860 juftlik, har bir inglizcha gap **5–8 so'z**.
* Tatoeba'dan foydalanmoqchi bo'lsangiz — pastdagi ixtiyoriy katakchani ishlating.

**Avval fayl formatini tekshiramiz** (vazifa talabi):
""")

code(r"""
DATA_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/translator/main/data/en_uz.tsv"
DATA = "../data/en_uz.tsv" if os.path.exists("../data/en_uz.tsv") else "en_uz.tsv"
if not os.path.exists(DATA):
    !wget -q -O en_uz.tsv {DATA_URL}

with open(DATA, encoding="utf-8") as f:
    head = [next(f) for _ in range(4)]
print("Fayl boshi (xom ko'rinishda):")
for line in head: print(repr(line))

pairs = []
with open(DATA, encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t"); next(r)
    for en, uz in r:
        pairs.append((en, uz))
print("\nJuftliklar soni:", len(pairs))
lens = [len(en.split()) for en, _ in pairs]
print("Inglizcha gap uzunligi (so'z): min", min(lens), "max", max(lens))
for en, uz in random.sample(pairs, 5): print(f"  {en:45s} → {uz}")
""")

code(r"""
# (Ixtiyoriy) Tatoeba ma'lumotlari bilan almashtirish — Colab'da ishlaydi.
USE_TATOEBA = False
if USE_TATOEBA:
    !wget -q https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
    !wget -q https://downloads.tatoeba.org/exports/per_language/uzb/uzb_sentences.tsv.bz2
    !wget -q https://downloads.tatoeba.org/exports/links.tar.bz2
    !bunzip2 -f eng_sentences.tsv.bz2 uzb_sentences.tsv.bz2 && tar xjf links.tar.bz2
    eng = dict(l.rstrip("\n").split("\t")[0::2] for l in open("eng_sentences.tsv", encoding="utf-8"))
    uzb = dict(l.rstrip("\n").split("\t")[0::2] for l in open("uzb_sentences.tsv", encoding="utf-8"))
    tat = []
    for l in open("links.csv"):
        a, b = l.split()
        if a in eng and b in uzb and 5 <= len(eng[a].split()) <= 8:
            tat.append((eng[a], uzb[b]))
    print("Tatoeba juftliklari:", len(tat))
    pairs = tat[:2000]
""")

md(r"""
## 2. 1-vazifa — Lug'at va tokenlar

* Tokenizatsiya: kichik harf, so'zlar va tinish belgilari alohida token. O'zbekcha `o'`, `g'`, `'` (tutuq belgisi) so'z ichida qoladi.
* **Umumiy lug'at** (modelda `vocab` bitta, ya'ni ingliz va o'zbek so'zlari bitta ro'yxatda).
* Maxsus tokenlar: `<pad>`=0, `<sos>`=1, `<eos>`=2, `<unk>`=3 (notanish so'z uchun — qo'shimcha).
* Hamma gap `T` uzunlikkacha padding qilinadi.
""")

code(r"""
APOS = str.maketrans({"ʻ": "'", "ʼ": "'", "’": "'", "‘": "'", "`": "'"})
TOKEN_RE = re.compile(r"[\w]+(?:'[\w]+)*'?|[^\w\s]")

def tokenize(s):
    return TOKEN_RE.findall(s.translate(APOS).lower())

print(tokenize("My mother cooks very delicious plov."))
print(tokenize("Onam juda mazali palov pishiradi."))
print(tokenize("Men o'zbekcha gapira olaman, sizchi?"))

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
specials = [PAD, SOS, EOS, UNK]

# train / test bo'linishi (10% test) — baholash uchun
random.shuffle(pairs)
n_test = len(pairs) // 10
test_pairs, train_pairs = pairs[:n_test], pairs[n_test:]

words = sorted({w for en, uz in train_pairs for w in tokenize(en) + tokenize(uz)})
itos = specials + words
stoi = {w: i for i, w in enumerate(itos)}
pad_id, sos_id, eos_id, unk_id = (stoi[t] for t in specials)
vocab = len(itos)

max_src = max(len(tokenize(en)) for en, _ in pairs)
max_tgt = max(len(tokenize(uz)) for _, uz in pairs)
T = max(max_src, max_tgt + 1)          # +1: <sos> yoki <eos> uchun joy
print(f"vocab = {vocab},  eng uzun src = {max_src},  eng uzun tgt = {max_tgt},  T = {T}")
print("train:", len(train_pairs), " test:", len(test_pairs))
""")

code(r"""
def encode(tokens):
    return [stoi.get(w, unk_id) for w in tokens]

def pad_to(ids, T):
    assert len(ids) <= T, "gap T dan uzun!"
    return ids + [pad_id] * (T - len(ids))

def make_tensors(pairs):
    S, I, O = [], [], []
    for en, uz in pairs:
        s = encode(tokenize(en))
        u = encode(tokenize(uz))
        S.append(pad_to(s, T))
        I.append(pad_to([sos_id] + u, T))   # tgt_in  = <sos> + gap
        O.append(pad_to(u + [eos_id], T))   # tgt_out = gap + <eos>
    return torch.tensor(S), torch.tensor(I), torch.tensor(O)

src_tr, tin_tr, tout_tr = make_tensors(train_pairs)
src_te, tin_te, tout_te = make_tensors(test_pairs)
print(src_tr.shape, tin_tr.shape, tout_tr.shape)
""")

md(r"""
## 3. 2-vazifa — `tgt_in` va `tgt_out` (qo'lda misol)

`tgt_in = <sos> + gap`, `tgt_out = gap + <eos>`. Model **har pozitsiyada keyingi so'zni** bashorat qiladi, shuning uchun `tgt_out` bir qadam chapga siljigan:
""")

code(r"""
def show(ids):
    return " ".join(itos[i] for i in ids if i != pad_id)

gap = "men seni sevaman"
u = [stoi.get(w, unk_id) for w in gap.split()]
print("Gap:     ", gap)
print("tgt_in:  ", show([sos_id] + u))
print("tgt_out: ", show(u + [eos_id]))

print("\nDatasetdan haqiqiy misol (pad bilan):")
k = 0
print("src     :", [itos[i] for i in src_tr[k].tolist()])
print("tgt_in  :", [itos[i] for i in tin_tr[k].tolist()])
print("tgt_out :", [itos[i] for i in tout_tr[k].tolist()])
print("\nPozitsiya bo'yicha: decoder ko'radi → bashorat qilishi kerak")
for a, b in zip(tin_tr[k].tolist(), tout_tr[k].tolist()):
    if b == pad_id: break
    print(f"  {itos[a]:>14s}  →  {itos[b]}")
""")

md(r"""
## 4. Berilgan model (o'zgartirilmagan)

O'qituvchi eslatmasiga ko'ra `torch.arange(...)` CPU'da yaratiladi, shuning uchun **berilgan modelni CPU'da o'qitamiz** (model kichik — CPU'da ham tez).
""")

code(r"""
class Seq2Seq(nn.Module):
    def __init__(self, vocab, T, C):
        super().__init__()
        self.src_tok = nn.Embedding(vocab, C)
        self.tgt_tok = nn.Embedding(vocab, C)
        self.pos = nn.Embedding(T, C)
        self.tf = nn.Transformer(d_model=C, nhead=2, num_encoder_layers=2,
                                 num_decoder_layers=2, dim_feedforward=4 * C,
                                 batch_first=True)
        self.out = nn.Linear(C, vocab)

    def embed(self, tok, ids):
        return tok(ids) + self.pos(torch.arange(ids.size(1)))

    def forward(self, src, tgt_in):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))
        s = self.embed(self.src_tok, src)
        t = self.embed(self.tgt_tok, tgt_in)
        h = self.tf(s, t, tgt_mask=mask)
        return self.out(h)
""")

md(r"""
## 5. 3-vazifa — O'qitish

* Loss: `CrossEntropyLoss(ignore_index=pad_id)` — `<pad>` pozitsiyalari hisobga olinmaydi.
* Optimizer: `AdamW`. Epochlar: **40** (≥30).
* Har epochda **test loss** ham hisoblanadi — overfitting'ni ko'rish uchun.
""")

code(r"""
def batches(n, bs, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]

def train_model(model, epochs=40, lr=5e-4, bs=32, loss_fn=None, verbose=True, device="cpu"):
    model.to(device)
    loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = {"train": [], "test": []}
    for ep in range(1, epochs + 1):
        model.train(); tot = cnt = 0
        for b in batches(len(src_tr), bs):
            s, ti, to = src_tr[b].to(device), tin_tr[b].to(device), tout_tr[b].to(device)
            logits = model(s, ti)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), to.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b); cnt += len(b)
        hist["train"].append(tot / cnt)
        model.eval()
        with torch.no_grad():
            lg = model(src_te.to(device), tin_te.to(device))
            hist["test"].append(nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tout_te.to(device).reshape(-1), ignore_index=pad_id).item())
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"epoch {ep:3d} | train loss {hist['train'][-1]:.3f} | test loss {hist['test'][-1]:.3f}")
    return hist

C = 128
torch.manual_seed(SEED)
model = Seq2Seq(vocab, T, C)
print("Parametrlar soni:", sum(p.numel() for p in model.parameters()))
t0 = time.time()
hist = train_model(model, epochs=40)
print(f"O'qitish vaqti: {time.time() - t0:.0f} s")
""")

code(r"""
os.makedirs("../assets", exist_ok=True) if os.path.exists("../data") else None
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(hist["train"]) + 1), hist["train"], label="train loss", lw=2)
plt.plot(range(1, len(hist["test"]) + 1), hist["test"], label="test loss", lw=2, ls="--")
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.title("Seq2Seq (C=128, nhead=2, 2+2 qatlam) — loss grafigi")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
if os.path.exists("../assets"): plt.savefig("../assets/loss.png", dpi=150)
plt.show()
""")

md(r"""
## 6. 4-vazifa — `translate()` (greedy, argmax)

`<sos>` dan boshlaymiz → har qadamda modelni chaqiramiz → **oxirgi pozitsiyaning** logitlaridan `argmax` → `tgt` ga `torch.cat` bilan qo'shamiz → `<eos>` chiqsa to'xtaymiz.
""")

code(r"""
@torch.no_grad()
def translate(model, src_ids, sos_id, eos_id, max_len):
    model.eval()
    tgt = torch.tensor([[sos_id]])
    for _ in range(max_len):
        logits = model(src_ids, tgt)                     # (1, len, vocab)
        next_id = logits[0, -1].argmax().item()          # oxirgi pozitsiyadan argmax
        tgt = torch.cat([tgt, torch.tensor([[next_id]])], dim=1)
        if next_id == eos_id:
            break
    return tgt

def src_tensor(sentence):
    return torch.tensor([pad_to(encode(tokenize(sentence))[:T], T)])

def detok(ids):
    words = [itos[i] for i in ids if i not in (sos_id, eos_id, pad_id)]
    s = " ".join(words)
    s = re.sub(r" ([.,?!])", r"\1", s)
    return s[:1].upper() + s[1:]

def tr(model, sentence, max_len=T - 1):
    out = translate(model, src_tensor(sentence), sos_id, eos_id, max_len)
    return detok(out[0].tolist())

for s in ["I drink tea every morning.", "My father works in the hospital.", "We will go to Tashkent tomorrow."]:
    print(f"{s:40s} → {tr(model, s)}")
""")

md(r"""
## 7. Natijalar: 10 ta tarjima namunasi (to'g'ri va xato) + ⭐ xatolar tahlili

Avval **test to'plamidan** (model o'qitishda ko'rmagan gaplar) tarjimalar, keyin o'zim yozgan "qiyin" gaplar.
""")

code(r"""
import sacrebleu

def evaluate(model, pairs, fn=None):
    fn = fn or (lambda s: tr(model, s))
    hyps = [fn(en) for en, _ in pairs]
    refs = [uz for _, uz in pairs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    exact = sum(h.lower() == r.lower() for h, r in zip(hyps, refs)) / len(refs) * 100
    return bleu, chrf, exact, hyps

bleu, chrf, exact, hyps = evaluate(model, test_pairs)
print(f"TEST:  BLEU = {bleu:.1f}   chrF = {chrf:.1f}   aniq moslik = {exact:.1f}%\n")

ok  = [(en, uz, h) for (en, uz), h in zip(test_pairs, hyps) if h.lower() == uz.lower()]
bad = [(en, uz, h) for (en, uz), h in zip(test_pairs, hyps) if h.lower() != uz.lower()]
print("✅ TO'G'RI tarjimalar:")
for en, uz, h in ok[:5]:
    print(f"   EN : {en}\n   MODEL: {h}\n")
print("❌ XATO tarjimalar:")
for en, uz, h in bad[:5]:
    print(f"   EN : {en}\n   MODEL: {h}\n   TO'G'RI: {uz}\n")
""")

code(r"""
# ⭐ Qo'shimcha: 10 ta o'zim yozgan gap (ba'zilari ataylab qiyin)
my10 = [
    "I read a book every evening.",
    "My sister is learning German now.",
    "Do you want to go to Berlin?",
    "The children are playing in the park.",
    "We did not watch the film yesterday.",
    "My brother has a new red car.",
    "The weather is very cold in Tashkent.",
    "She can speak English and German.",          # "and" lug'atda yo'q
    "My grandmother lives in a small village.",   # "grandmother" — notanish so'z
    "I love my beautiful country very much.",     # o'qitishda bunday tuzilma yo'q
]
for s in my10:
    unk = [w for w in tokenize(s) if w not in stoi]
    print(f"{s:45s} → {tr(model, s):40s} {'  [notanish: ' + ', '.join(unk) + ']' if unk else ''}")
""")

md(r"""
### ⭐ Xatolar tahlili: model nega ba'zan bitta so'zni takrorlaydi?

1. **Greedy (argmax) — "ochko'z" qaror.** Har qadamda faqat eng ehtimolli so'z tanlanadi; bir marta xato qilinsa, keyingi qadamlar shu xatoga tayanadi va qaytib tuzatib bo'lmaydi.
2. **Exposure bias.** O'qitishda decoder har doim **to'g'ri** oldingi so'zlarni ko'radi (teacher forcing), lekin `translate()` da **o'zining** (ba'zan xato) so'zlarini ko'radi. Notanish holatga tushgan model "xavfsiz" deb bilgan so'zni (masalan, `har`, `kuni`, fe'l) qayta-qayta tanlaydi va sikl hosil bo'ladi.
3. **Model "nimani yozib bo'ldim" ni hisoblamaydi** (coverage yo'q) — u `<eos>` ni kech bashorat qilsa, oxirgi so'zlarni takrorlaydi.
4. **Kichik ma'lumot va `<unk>`.** Lug'atda yo'q so'z kelsa (`grandmother`), encoder ma'noni yo'qotadi va decoder ko'p uchraydigan so'zlarga "yopishadi".
5. **Berilgan modelda padding mask yo'q** — `<pad>` tokenlar ham attention'da qatnashadi; bu shovqin qo'shadi.

**Yechimlar** (V2 da qo'llangan): beam search, label smoothing, padding mask, ko'proq ma'lumot, takrorlash jarimasi (repetition penalty).
""")

md(r"""
## 8. 5-vazifa — Tajriba: `C`, `nhead`, qatlamlar soni

Berilgan klass `nhead=2` va 2 qatlamni qattiq yozgan, shuning uchun tajriba uchun **xuddi shu arxitekturaning parametrlangan nusxasini** yaratamiz (forward bir xil).
Har bir sozlama 30 epoch o'qitiladi va test to'plamida baholanadi.
""")

code(r"""
class Seq2SeqExp(Seq2Seq):
    def __init__(self, vocab, T, C, nhead=2, layers=2):
        super().__init__(vocab, T, C)
        self.tf = nn.Transformer(d_model=C, nhead=nhead, num_encoder_layers=layers,
                                 num_decoder_layers=layers, dim_feedforward=4 * C, batch_first=True)

configs = [  # (C, nhead, layers)
    (64, 2, 2), (128, 2, 2), (256, 2, 2),
    (128, 4, 2), (128, 8, 2),
    (128, 2, 1), (128, 2, 3),
]
results = []
for C_, h_, L_ in configs:
    torch.manual_seed(SEED)
    m = Seq2SeqExp(vocab, T, C_, h_, L_)
    t0 = time.time()
    hst = train_model(m, epochs=30, verbose=False)
    b, c, e, _ = evaluate(m, test_pairs)
    n_par = sum(p.numel() for p in m.parameters())
    results.append(dict(C=C_, nhead=h_, layers=L_, params=n_par, train_loss=hst["train"][-1],
                        test_loss=min(hst["test"]), bleu=b, chrf=c, exact=e, sec=time.time() - t0, hist=hst))
    print(f"C={C_:3d} nhead={h_} layers={L_} | params={n_par/1e6:.2f}M | train {hst['train'][-1]:.3f} "
          f"| test {min(hst['test']):.3f} | BLEU {b:5.1f} | chrF {c:5.1f} | {time.time()-t0:.0f}s")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for r in results:
    lab = f"C={r['C']}, h={r['nhead']}, L={r['layers']}"
    ax[0].plot(r["hist"]["train"], label=lab); ax[1].plot(r["hist"]["test"], label=lab)
ax[0].set_title("train loss"); ax[1].set_title("test loss")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[1].legend(fontsize=8); plt.tight_layout()
if os.path.exists("../assets"): plt.savefig("../assets/experiments.png", dpi=150)
plt.show()

print(f"{'C':>4} {'nhead':>5} {'L':>2} {'params':>8} {'BLEU':>6} {'chrF':>6} {'aniq%':>6}")
for r in results:
    print(f"{r['C']:>4} {r['nhead']:>5} {r['layers']:>2} {r['params']/1e6:>7.2f}M {r['bleu']:>6.1f} {r['chrf']:>6.1f} {r['exact']:>6.1f}")
""")

md(r"""
### Tajriba xulosasi (3–4 jumla)

*Quyidagi xulosa yuqoridagi jadval natijalariga asoslangan — notebookni qayta ishga tushirsangiz, raqamlar biroz farq qilishi mumkin.*

EXPERIMENT_SUMMARY
""")

md(r"""
## 9. ❓ Savol: `mask` ni olib tashlasak, loss nega juda tez tushadi, lekin `translate()` ishlamaydi?

**Javob.** `tgt_out` — bu `tgt_in` ning bir qadam chapga siljigan nusxasi: `tgt_out[i] = tgt_in[i+1]`. Mask bo'lmasa, decoder `i`-pozitsiyada **butun** `tgt_in` ni, jumladan `i+1`-pozitsiyadagi so'zni ham ko'radi — ya'ni **javobni ko'rib, ko'chirib yozadi**. Model tarjima qilishni emas, "qo'shni pozitsiyadan nusxa olishni" o'rganadi, shuning uchun loss deyarli nolga tez tushadi.

`translate()` da esa kelajakdagi so'zlar **mavjud emas** — biz ularni endi yaratyapmiz. Model esa doim kelajakka qarab javob berishga o'rgangan; kelajak bo'lmagach (oxirgi pozitsiyada "keyingi so'z" yo'q), u nimani chiqarishni bilmaydi va ma'nosiz/takroriy so'zlar chiqaradi. Bu **o'qitish va inferens orasidagi nomuvofiqlik** (train/test mismatch).

Keling, buni **amalda isbotlaymiz**:
""")

code(r"""
class Seq2SeqNoMask(Seq2Seq):
    def forward(self, src, tgt_in):                      # xuddi shu, faqat tgt_mask YO'Q
        s = self.embed(self.src_tok, src)
        t = self.embed(self.tgt_tok, tgt_in)
        return self.out(self.tf(s, t))

torch.manual_seed(SEED)
nomask = Seq2SeqNoMask(vocab, T, 128)
hist_nm = train_model(nomask, epochs=20, verbose=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, key, title in [(ax[0], "train", "train loss"), (ax[1], "test", "test loss (teacher forcing)")]:
    a.plot(range(1, 21), hist[key][:20], label="mask BOR", lw=2)
    a.plot(range(1, 21), hist_nm[key], label="mask YO'Q", lw=2)
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(title); a.grid(alpha=.3, which="both"); a.legend()
fig.suptitle("Mask'siz model javobni 'ko'chirib' yozadi — loss nolga yaqinlashadi (log shkala)")
plt.tight_layout()
if os.path.exists("../assets"): plt.savefig("../assets/nomask.png", dpi=150)
plt.show()

print(f"20-epoch train loss:  mask bilan = {hist['train'][19]:.4f},  mask'siz = {hist_nm['train'][-1]:.4f}")
print(f"20-epoch test  loss:  mask bilan = {hist['test'][19]:.4f},  mask'siz = {hist_nm['test'][-1]:.4f}")
print("→ Mask'siz model hatto TEST to'plamida ham loss'i past — chunki u tarjima qilmaydi, keyingi so'zni tgt_in dan ko'chiradi.\n")
for s in ["I drink tea every morning.", "My father works in the hospital.", "We will go to Tashkent tomorrow."]:
    print(f"{s:36s} | mask bilan: {tr(model, s):38s} | mask'siz: {tr(nomask, s)}")
""")

md(r"""
---
# 🚀 Ijodiy qism (vazifadan tashqari)

## 10. V2 — yaxshilangan model

Berilgan arxitektura saqlangan, lekin amaliy muammolar tuzatilgan:

| Yaxshilanish | Nima uchun |
|---|---|
| `device=ids.device` | GPU'da ham ishlaydi (o'qituvchi eslatmasi) |
| **padding mask** (`src/tgt/memory_key_padding_mask`) | `<pad>` tokenlar attention'ga aralashmaydi |
| embedding × √C | pozitsiya va token signali muvozanatli |
| **weight tying** (tgt_tok ↔ out) | parametrlar kamayadi, kichik ma'lumotda yaxshiroq umumlashtiradi |
| **label smoothing 0.1** | model haddan tashqari ishonchli bo'lmaydi |
| LR warmup + cosine | barqaror o'qitish |
| **beam search** | greedy xatolarini kamaytiradi, takrorlanishni kamaytiradi |
""")

code(r"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class Seq2SeqV2(nn.Module):
    def __init__(self, vocab, T, C=256, nhead=4, layers=3, dropout=0.15, pad_id=0):
        super().__init__()
        self.C, self.pad_id = C, pad_id
        self.src_tok = nn.Embedding(vocab, C)
        self.tgt_tok = nn.Embedding(vocab, C)
        self.pos = nn.Embedding(T, C)
        self.tf = nn.Transformer(d_model=C, nhead=nhead, num_encoder_layers=layers,
                                 num_decoder_layers=layers, dim_feedforward=4 * C,
                                 dropout=dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(C, vocab)
        self.out.weight = self.tgt_tok.weight                 # weight tying

    def embed(self, tok, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        return tok(ids) * math.sqrt(self.C) + self.pos(pos)

    def forward(self, src, tgt_in):
        L = tgt_in.size(1)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=src.device), diagonal=1)
        sp, tp = src == self.pad_id, tgt_in == self.pad_id
        h = self.tf(self.embed(self.src_tok, src), self.embed(self.tgt_tok, tgt_in),
                    tgt_mask=mask, src_key_padding_mask=sp, tgt_key_padding_mask=tp,
                    memory_key_padding_mask=sp)
        return self.out(h)

def train_v2(model, epochs=60, lr=1e-3, bs=32, warmup=3):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(src_tr) / bs); wsteps = warmup * math.ceil(len(src_tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / wsteps) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    hist = {"train": [], "test": []}
    for ep in range(1, epochs + 1):
        model.train(); tot = cnt = 0
        for b in batches(len(src_tr), bs):
            s, ti, to = src_tr[b].to(device), tin_tr[b].to(device), tout_tr[b].to(device)
            lg = model(s, ti)
            loss = loss_fn(lg.reshape(-1, lg.size(-1)), to.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b); cnt += len(b)
        model.eval()
        with torch.no_grad():
            lg = model(src_te.to(device), tin_te.to(device))
            hist["test"].append(nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                tout_te.to(device).reshape(-1), ignore_index=pad_id).item())
        hist["train"].append(tot / cnt)
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:3d} | train {hist['train'][-1]:.3f} | test {hist['test'][-1]:.3f}")
    return hist

torch.manual_seed(SEED)
v2 = Seq2SeqV2(vocab, T, pad_id=pad_id)
print("V2 parametrlari:", sum(p.numel() for p in v2.parameters()))
t0 = time.time(); hist_v2 = train_v2(v2); print(f"{time.time()-t0:.0f}s")
""")

code(r"""
@torch.no_grad()
def beam_translate(model, sentence, beam=4, max_len=T - 1, alpha=0.7):
    model.eval()
    src = src_tensor(sentence).to(device)
    beams = [([sos_id], 0.0)]
    done = []
    for _ in range(max_len):
        cand = []
        for seq, sc in beams:
            lg = model(src, torch.tensor([seq], device=device))[0, -1]
            lp = torch.log_softmax(lg, -1)
            top = lp.topk(beam)
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                if i == seq[-1]:      # bitta so'zni ketma-ket takrorlashga jarima
                    v -= 2.0
                cand.append((seq + [i], sc + v))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for seq, sc in cand:
            (done if seq[-1] == eos_id else beams).append((seq, sc))
            if len(beams) == beam: break
        if len(done) >= beam or not beams: break
    done += beams
    best = max(done, key=lambda x: x[1] / (len(x[0]) ** alpha))   # uzunlik normalizatsiyasi
    return detok(best[0])

def greedy_v2(sentence):
    out = translate(v2.cpu(), src_tensor(sentence), sos_id, eos_id, T - 1); v2.to(device)
    return detok(out[0].tolist())

b1, c1, e1, _ = evaluate(model, test_pairs)
v2.cpu(); device_backup, device = device, "cpu"
b2, c2, e2, _ = evaluate(v2, test_pairs, fn=greedy_v2)
b3, c3, e3, _ = evaluate(v2, test_pairs, fn=lambda s: beam_translate(v2, s))
device = device_backup; v2.to(device)

print(f"{'Model':32s} {'BLEU':>6} {'chrF':>6} {'aniq%':>6}")
print(f"{'Berilgan Seq2Seq + greedy':32s} {b1:6.1f} {c1:6.1f} {e1:6.1f}")
print(f"{'V2 + greedy':32s} {b2:6.1f} {c2:6.1f} {e2:6.1f}")
print(f"{'V2 + beam search (k=4)':32s} {b3:6.1f} {c3:6.1f} {e3:6.1f}")
SCORES = dict(base=dict(bleu=b1, chrf=c1, exact=e1), v2_greedy=dict(bleu=b2, chrf=c2, exact=e2),
              v2_beam=dict(bleu=b3, chrf=c3, exact=e3))
""")

code(r"""
v2.cpu(); device = "cpu"
print(f"{'Inglizcha':45s} | {'Berilgan model':38s} | V2 + beam")
for s in my10:
    print(f"{s:45s} | {tr(model, s):38s} | {beam_translate(v2, s)}")
""")

md(r"""
## 11. Saqlash va ONNX eksport → brauzerdagi translator

V2 modeli ONNX formatiga o'tkaziladi va **Translator** veb-ilovasida ("Mening modelim" rejimi) to'g'ridan-to'g'ri brauzerda, serversiz ishlaydi.
""")

code(r"""
os.makedirs("export", exist_ok=True)
torch.save({"state_dict": v2.state_dict(), "itos": itos, "T": T,
            "config": dict(C=256, nhead=4, layers=3)}, "export/translator_v2.pt")

v2.eval().cpu()
try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass
# Brauzer uchun ikkala kirish ham T uzunlikda (statik shakl). Causal mask tufayli k-pozitsiyadagi
# natija faqat tgt[0..k] ga bog'liq, shuning uchun tgt_in ni <pad> bilan to'ldirish natijani o'zgartirmaydi.
dummy_src = src_tensor("I drink tea every morning.")
dummy_tgt = torch.tensor([pad_to([sos_id, stoi["men"]], T)])
torch.onnx.export(v2, (dummy_src, dummy_tgt), "export/translator_v2_fp32.onnx",
                  input_names=["src", "tgt_in"], output_names=["logits"],
                  opset_version=17, dynamo=False)

# int8 kvantizatsiya: fayl ~4 baravar kichrayadi (veb uchun qulay)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("export/translator_v2_fp32.onnx", "export/translator_v2.onnx", weight_type=QuantType.QInt8)

with open("export/vocab.json", "w", encoding="utf-8") as f:
    json.dump({"itos": itos, "T": T, "pad_id": pad_id, "sos_id": sos_id, "eos_id": eos_id,
               "unk_id": unk_id, "scores": SCORES}, f, ensure_ascii=False)
for fn in sorted(os.listdir("export")):
    print(f"{fn:28s} {os.path.getsize('export/' + fn)/1e6:6.1f} MB")
""")

code(r"""
# ONNX ↔ PyTorch: natijalar bir xilmi? (fp32 aniq, int8 — taxminiy) + int8 model bilan greedy tarjimalar mosligi
import importlib.util
import onnxruntime as ort, numpy as np
s32 = ort.InferenceSession("export/translator_v2_fp32.onnx")
s8 = ort.InferenceSession("export/translator_v2.onnx")
for L in (1, 3, 6):
    t = [sos_id] + torch.randint(4, vocab, (L - 1,)).tolist()
    ref = v2(dummy_src, torch.tensor([t]))[0].detach().numpy()            # padding'siz PyTorch
    padded = np.array([pad_to(t, T)])
    a = s32.run(None, {"src": dummy_src.numpy(), "tgt_in": padded})[0][0, :L]
    print(f"tgt_len={L}: fp32 max |farq| = {np.abs(a - ref).max():.2e}")

def ort_greedy(sess, sentence):
    src = src_tensor(sentence).numpy(); seq = [sos_id]
    for _ in range(T - 1):
        lg = sess.run(None, {"src": src, "tgt_in": np.array([pad_to(seq, T)])})[0][0, len(seq) - 1]
        seq.append(int(lg.argmax()))
        if seq[-1] == eos_id: break
    return detok(seq)

same = sum(ort_greedy(s8, en) == greedy_v2(en) for en, _ in test_pairs)
print(f"\nint8 ONNX va PyTorch greedy tarjimalari bir xil: {same}/{len(test_pairs)} ({same/len(test_pairs)*100:.1f}%)")
b8, c8, _, _ = evaluate(None, test_pairs, fn=lambda s: ort_greedy(s8, s))
print(f"int8 ONNX greedy: BLEU {b8:.1f}  chrF {c8:.1f}")
""")

md(r"""
## Xulosa

* Berilgan Encoder–Decoder Transformer **o'zgartirilmasdan** o'qitildi, `tgt_in`/`tgt_out` siljishi qo'lda ko'rsatildi, `translate()` greedy decoding bilan yozildi.
* Mask'ning roli **eksperiment bilan isbotlandi**: mask'siz model loss'i tez tushadi, lekin tarjima qila olmaydi.
* Tajribalar `C`, `nhead`, qatlamlar soni ta'sirini ko'rsatdi.
* Ijodiy qism: V2 (padding mask, label smoothing, weight tying, beam search) BLEU ni oshirdi va ONNX orqali **brauzerda ishlaydigan** haqiqiy translatorga aylantirildi → **Translator** veb-ilovasi (EN ⇄ UZ ⇄ DE).
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}
nbf.write(nb, "notebooks/Translator_Seq2Seq.ipynb")
print("ok")
