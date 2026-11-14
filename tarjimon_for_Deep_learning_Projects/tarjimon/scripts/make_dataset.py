"""
Tarjimon — English → Uzbek parallel dataset generator.

The assignment allows "o'zingiz yozgan ro'yxat" (your own list). This script
builds one with *rule-based Uzbek morphology* so every pair is grammatical:

  * verb conjugation  (present-future, past, continuous, negative, want, can)
  * person suffixes   (-man, -siz, -di, -miz, -dilar ...)
  * case suffixes     (accusative -ni, dative -ga/-ka/-qa, locative -da)
  * possessive suffixes with consonant mutation (mushuk -> mushugim)

plus a hand-written list of everyday phrases. Only pairs whose English side
has 5–8 words are kept, as the assignment requires.

Usage:  python scripts/make_dataset.py  ->  data/en_uz.tsv
"""
import random, re, csv, os, itertools

random.seed(42)
VOWELS = set("aeiou")

# ---------------------------------------------------------------- subjects
# (english, uzbek, person, is_pronoun)
SUBJECTS = [
    ("I", "Men", "1s"), ("you", "Siz", "2"), ("he", "U", "3s"), ("she", "U", "3s"),
    ("we", "Biz", "1p"), ("they", "Ular", "3p"),
    ("my mother", "Onam", "3s"), ("my father", "Otam", "3s"),
    ("my brother", "Akam", "3s"), ("my sister", "Opam", "3s"),
    ("my friend", "Do'stim", "3s"), ("the teacher", "O'qituvchi", "3s"),
    ("the students", "Talabalar", "3p"), ("the children", "Bolalar", "3p"),
    ("Ali", "Ali", "3s"), ("Laylo", "Laylo", "3s"), ("Anvar", "Anvar", "3s"),
]

# ---------------------------------------------------------------- uzbek morphology
PRES = {"1s": "man", "2": "siz", "3s": "di", "1p": "miz", "3p": "dilar"}
PAST = {"1s": "dim", "2": "dingiz", "3s": "di", "1p": "dik", "3p": "dilar"}
CONT = {"1s": "man", "2": "siz", "3s": "ti", "1p": "miz", "3p": "tilar"}
WANT = {"1s": "man", "2": "siz", "3s": "", "1p": "miz", "3p": ""}
COPULA = {"1s": "man", "2": "siz", "3s": "", "1p": "miz", "3p": ""}


def last_word(stem):
    return stem.split()[-1]


def pres_stem(stem):          # o'qi -> o'qiy ; yoz -> yoza
    return stem + ("y" if last_word(stem)[-1] in VOWELS else "a")


def uz_present(stem, p):      return pres_stem(stem) + PRES[p]
def uz_past(stem, p):         return stem + PAST[p]
def uz_cont(stem, p):         return stem + "yap" + CONT[p]
def uz_neg_present(stem, p):  return stem + "may" + PRES[p]
def uz_neg_past(stem, p):     return stem + "ma" + PAST[p]
def uz_want(stem, p):         return stem + "moqchi" + WANT[p]
def uz_can(stem, p):          return pres_stem(stem) + " ola" + PRES[p]
def uz_cannot(stem, p):       return pres_stem(stem) + " olmay" + PRES[p]


def possessive(noun, p):
    """kitob -> kitobim, mashina -> mashinam, mushuk -> mushugim"""
    vowel_end = noun[-1] in VOWELS
    suf = {"1s": ("m", "im"), "2": ("ngiz", "ingiz"), "3s": ("si", "i"),
           "1p": ("miz", "imiz"), "3p": ("si", "i")}[p]
    if vowel_end:
        return noun + suf[0]
    if noun.endswith("k"):
        noun = noun[:-1] + "g"
    elif noun.endswith("q"):
        noun = noun[:-1] + "g'"
    return noun + suf[1]


# ---------------------------------------------------------------- english morphology
BE = {"1s": "am", "2": "are", "3s": "is", "1p": "are", "3p": "are"}
DO = {"1s": "do", "2": "do", "3s": "does", "1p": "do", "3p": "do"}
HAVE = {"1s": "have", "2": "have", "3s": "has", "1p": "have", "3p": "have"}


def a_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


# ---------------------------------------------------------------- verbs with objects
# objects: (en_indef or None, uz_indef or None, en_def or None, uz_def or None)
def obj(en, uz, en_def=None, uz_def=None):
    return (en, uz, en_def, uz_def)


VERBS = {
    "read": dict(en=("read", "reads", "read", "reading"), uz="o'qi", objs=[
        obj("a book", "kitob", "the book", "kitobni"), obj("a newspaper", "gazeta", "the newspaper", "gazetani"),
        obj("a letter", "xat", "the letter", "xatni"), obj("a magazine", "jurnal", "this magazine", "bu jurnalni")]),
    "write": dict(en=("write", "writes", "wrote", "writing"), uz="yoz", objs=[
        obj("a letter", "xat", "the letter", "xatni"), obj("a story", "hikoya", "this story", "bu hikoyani"),
        obj("a poem", "she'r", "the poem", "she'rni")]),
    "drink": dict(en=("drink", "drinks", "drank", "drinking"), uz="ich", objs=[
        obj("water", "suv", "the water", "suvni"), obj("tea", "choy", "the tea", "choyni"),
        obj("milk", "sut", "the milk", "sutni"), obj("coffee", "qahva", "the coffee", "qahvani"),
        obj("juice", "sharbat", "the juice", "sharbatni"), obj("green tea", "ko'k choy", None, None)]),
    "eat": dict(en=("eat", "eats", "ate", "eating"), uz="ye", objs=[
        obj("bread", "non", "the bread", "nonni"), obj("an apple", "olma", "the apple", "olmani"),
        obj("soup", "sho'rva", "the soup", "sho'rvani"), obj("meat", "go'sht", "the meat", "go'shtni"),
        obj("fruit", "meva", "the fruit", "mevani"), obj("plov", "palov", "the plov", "palovni")]),
    "watch": dict(en=("watch", "watches", "watched", "watching"), uz="ko'r", objs=[
        obj("a film", "film", "the film", "filmni"), obj("TV", "televizor", None, None),
        obj("a cartoon", "multfilm", "this cartoon", "bu multfilmni"),
        obj("football", "futbol", "the match", "o'yinni")]),
    "buy": dict(en=("buy", "buys", "bought", "buying"), uz="sotib ol", objs=[
        obj("bread", "non", "the bread", "nonni"), obj("apples", "olma", "the apples", "olmalarni"),
        obj("a book", "kitob", "this book", "bu kitobni"), obj("milk", "sut", "the milk", "sutni"),
        obj("a car", "mashina", "the car", "mashinani"), obj("a phone", "telefon", "this phone", "bu telefonni"),
        obj("flowers", "gul", "the flowers", "gullarni")]),
    "clean": dict(en=("clean", "cleans", "cleaned", "cleaning"), uz="tozala", objs=[
        obj(None, None, "the room", "xonani"), obj(None, None, "the house", "uyni"),
        obj(None, None, "the kitchen", "oshxonani"), obj(None, None, "the table", "stolni"),
        obj(None, None, "the classroom", "sinfni")]),
    "wash": dict(en=("wash", "washes", "washed", "washing"), uz="yuv", objs=[
        obj(None, None, "the dishes", "idishlarni"), obj(None, None, "the car", "mashinani"),
        obj(None, None, "the clothes", "kiyimlarni"), obj(None, None, "the apples", "olmalarni")]),
    "cook": dict(en=("cook", "cooks", "cooked", "cooking"), uz="pishir", objs=[
        obj("soup", "sho'rva", "the soup", "sho'rvani"), obj("plov", "palov", "the plov", "palovni"),
        obj("dinner", "kechki ovqat", "the dinner", "kechki ovqatni"),
        obj("breakfast", "nonushta", None, None), obj("meat", "go'sht", "the meat", "go'shtni")]),
    "learn": dict(en=("learn", "learns", "learned", "learning"), uz="o'rgan", objs=[
        obj("English", "ingliz tilini"), obj("German", "nemis tilini"), obj("Uzbek", "o'zbek tilini"),
        obj("Russian", "rus tilini"), obj("math", "matematikani"), obj("new words", "yangi so'zlarni"),
        obj("programming", "dasturlashni")]),
    "like": dict(en=("like", "likes", "liked", None), uz="yoqtir", objs=[
        obj("music", "musiqani"), obj("tea", "choyni"), obj("coffee", "qahvani"), obj("football", "futbolni"),
        obj("chess", "shaxmatni"), obj("apples", "olmani"), obj("this city", "bu shaharni"),
        obj("cold water", "sovuq suvni"), obj("this song", "bu qo'shiqni")]),
    "love": dict(en=("love", "loves", "loved", None), uz="sev", objs=[
        obj("music", "musiqani"), obj("this city", "bu shaharni"), obj("flowers", "gullarni"),
        obj("my mother", "onamni"), obj("books", "kitoblarni")]),
    "play": dict(en=("play", "plays", "played", "playing"), uz="o'yna", objs=[
        obj("football", "futbol"), obj("chess", "shaxmat"), obj("tennis", "tennis"),
        obj("basketball", "basketbol"), obj("volleyball", "voleybol"), obj("the piano", "pianino")]),
    "listen": dict(en=("listen to", "listens to", "listened to", "listening to"), uz="tingla", objs=[
        obj("music", "musiqa", "this song", "bu qo'shiqni"), obj("the radio", "radio"),
        obj(None, None, "the teacher", "o'qituvchini"), obj(None, None, "my father", "otamni")]),
    "open": dict(en=("open", "opens", "opened", "opening"), uz="och", objs=[
        obj(None, None, "the door", "eshikni"), obj(None, None, "the window", "derazani"),
        obj(None, None, "the box", "qutini"), obj(None, None, "the shop", "do'konni")]),
    "close": dict(en=("close", "closes", "closed", "closing"), uz="yop", objs=[
        obj(None, None, "the door", "eshikni"), obj(None, None, "the window", "derazani"),
        obj(None, None, "the box", "qutini"), obj(None, None, "the book", "kitobni")]),
    "help": dict(en=("help", "helps", "helped", "helping"), uz="yordam ber", objs=[
        obj("my mother", "onamga"), obj("my father", "otamga"), obj("the teacher", "o'qituvchiga"),
        obj("my friend", "do'stimga"), obj("the children", "bolalarga"), obj("my sister", "opamga")]),
    "call": dict(en=("call", "calls", "called", "calling"), uz="qo'ng'iroq qil", objs=[
        obj("my mother", "onamga"), obj("my father", "otamga"), obj("my friend", "do'stimga"),
        obj("the doctor", "shifokorga"), obj("my brother", "akamga")]),
    "wait": dict(en=("wait for", "waits for", "waited for", "waiting for"), uz="kut", objs=[
        obj("the bus", "avtobusni"), obj("my friend", "do'stimni"), obj("the teacher", "o'qituvchini"),
        obj("the train", "poyezdni"), obj("my brother", "akamni")]),
    "speak": dict(en=("speak", "speaks", "spoke", "speaking"), uz="gapir", objs=[
        obj("English", "inglizcha"), obj("German", "nemischa"), obj("Uzbek", "o'zbekcha"),
        obj("Russian", "ruscha")]),
    "draw": dict(en=("draw", "draws", "drew", "drawing"), uz="chiz", objs=[
        obj("a picture", "rasm", "this picture", "bu rasmni"), obj("a cat", "mushuk", None, None),
        obj("a map", "xarita", "the map", "xaritani")]),
}

COMPANIONS = [("with my friend", "do'stim bilan"), ("with my brother", "akam bilan"),
              ("with my family", "oilam bilan"), ("with my sister", "opam bilan"),
              ("with the children", "bolalar bilan")]

ADV = {
    "present": [("every day", "har kuni", "end"), ("every morning", "har kuni ertalab", "end"),
                ("every evening", "har kuni kechqurun", "end"), ("in the morning", "ertalab", "end"),
                ("in the evening", "kechqurun", "end"), ("on Sunday", "yakshanba kuni", "end"),
                ("often", "tez-tez", "pre"), ("sometimes", "ba'zan", "pre"), ("usually", "odatda", "pre"),
                ("", "", "end")],
    "past": [("yesterday", "kecha", "end"), ("last week", "o'tgan hafta", "end"),
             ("this morning", "bugun ertalab", "end"), ("last night", "kecha kechqurun", "end"),
             ("on Sunday", "yakshanba kuni", "end")],
    "cont": [("now", "hozir", "end"), ("at the moment", "ayni paytda", "end"), ("right now", "hozir", "end")],
    "future": [("tomorrow", "ertaga", "end"), ("next week", "keyingi hafta", "end"),
               ("tonight", "bugun kechqurun", "end"), ("on Sunday", "yakshanba kuni", "end")],
    "none": [("", "", "end")],
}


def cap(s):
    s = s.strip()
    return s[0].upper() + s[1:] if s else s


def clean_spaces(s):
    return re.sub(r"\s+", " ", s).strip()


def build(en_parts, uz_parts, q=False):
    end = "?" if q else "."
    en = cap(clean_spaces(" ".join(en_parts))) + end
    uz = cap(clean_spaces(" ".join(uz_parts))) + end
    return en, uz


def subj_en(s, initial=True):
    return s[0]


# ---------------------------------------------------------------- family: verbs + objects
def gen_verb_sentences():
    out = []
    for (sen, suz, p), (vk, v), tense in itertools.product(SUBJECTS, VERBS.items(),
            ["present", "past", "cont", "future", "neg_present", "neg_past", "q_present", "q_past",
             "q_cont", "want", "can", "cannot", "q_can"]):
        base, s3, past, ing = v["en"]
        stem = v["uz"]
        if ing is None and tense in ("cont", "q_cont", "want", "can", "cannot", "q_can", "future"):
            continue
        for (ei, ui, ed, ud) in v["objs"]:
            for use_def in (False, True):
                eo, uo = (ed, ud) if use_def else (ei, ui)
                if eo is None:
                    continue
                if eo.replace("my ", "") in sen.replace("my ", "") and eo.startswith("my"):
                    continue  # "my mother helps my mother"
                adv_key = {"present": "present", "past": "past", "cont": "cont", "future": "future",
                           "neg_present": "present", "neg_past": "past", "q_present": "present",
                           "q_past": "past", "q_cont": "cont", "want": "future", "can": "none",
                           "cannot": "none", "q_can": "none"}[tense]
                for (ea, ua, pos) in ADV[adv_key]:
                    comp = random.random() < 0.15
                    ec, uc = random.choice(COMPANIONS) if comp else ("", "")
                    if comp and ec.split()[-1] in sen:
                        ec, uc = "", ""
                    tail = [eo, ec] + ([ea] if pos == "end" else [])
                    pre = [ea] if pos == "pre" else []
                    q = False
                    if tense == "present":
                        en = [sen] + pre + [s3 if p == "3s" else base] + tail
                        uzv = uz_present(stem, p)
                    elif tense == "past":
                        en = [sen] + pre + [past] + tail
                        uzv = uz_past(stem, p)
                    elif tense == "cont":
                        en = [sen, BE[p], ing] + tail
                        uzv = uz_cont(stem, p)
                    elif tense == "future":
                        en = [sen, "will", base] + tail
                        uzv = uz_present(stem, p)
                    elif tense == "neg_present":
                        if pos == "pre":
                            continue
                        en = [sen, DO[p], "not", base] + tail
                        uzv = uz_neg_present(stem, p)
                    elif tense == "neg_past":
                        en = [sen, "did not", base] + tail
                        uzv = uz_neg_past(stem, p)
                    elif tense == "q_present":
                        if pos == "pre" or p == "1s":
                            continue
                        en = [DO[p], sen] + [base] + tail
                        uzv = uz_present(stem, p) + "mi"; q = True
                    elif tense == "q_past":
                        if p == "1s":
                            continue
                        en = ["did", sen, base] + tail
                        uzv = uz_past(stem, p) + "mi"; q = True
                    elif tense == "q_cont":
                        if p == "1s":
                            continue
                        en = [BE[p], sen, ing] + tail
                        uzv = uz_cont(stem, p) + "mi"; q = True
                    elif tense == "want":
                        en = [sen, "wants to" if p == "3s" else "want to", base] + tail
                        uzv = uz_want(stem, p)
                    elif tense == "can":
                        en = [sen, "can", base] + tail
                        uzv = uz_can(stem, p)
                    elif tense == "cannot":
                        en = [sen, "cannot", base] + tail
                        uzv = uz_cannot(stem, p)
                    elif tense == "q_can":
                        if p == "1s":
                            continue
                        en = ["can", sen, base] + tail
                        uzv = uz_can(stem, p) + "mi"; q = True
                    uz = [suz, ua, uc, uo, uzv]
                    if q and en[0] in ("do", "does", "did", "can", "are", "is", "am"):
                        pass
                    out.append(("verb_" + tense,) + build(en, uz, q))
    return out


# ---------------------------------------------------------------- family: movement
PLACES = [  # en_to, uz_dat, en_at, uz_loc
    ("to school", "maktabga", "at school", "maktabda"), ("to work", "ishga", "at work", "ishda"),
    ("home", "uyga", "at home", "uyda"), ("to the market", "bozorga", "at the market", "bozorda"),
    ("to the park", "parkka", "in the park", "parkda"),
    ("to the library", "kutubxonaga", "in the library", "kutubxonada"),
    ("to university", "universitetga", "at university", "universitetda"),
    ("to the hospital", "kasalxonaga", "in the hospital", "kasalxonada"),
    ("to the city", "shaharga", "in the city", "shaharda"),
    ("to the village", "qishloqqa", "in the village", "qishloqda"),
    ("to the shop", "do'konga", "in the shop", "do'konda"),
    ("to the cinema", "kinoteatrga", "at the cinema", "kinoteatrda"),
    ("to Tashkent", "Toshkentga", "in Tashkent", "Toshkentda"),
    ("to Samarkand", "Samarqandga", "in Samarkand", "Samarqandda"),
    ("to Germany", "Germaniyaga", "in Germany", "Germaniyada"),
    ("to Berlin", "Berlinga", "in Berlin", "Berlinda"),
    ("to the garden", "bog'ga", "in the garden", "bog'da"),
    ("to the kitchen", "oshxonaga", "in the kitchen", "oshxonada"),
]
TRANSPORT = [("by bus", "avtobusda"), ("by car", "mashinada"), ("by train", "poyezdda"),
             ("on foot", "piyoda"), ("by taxi", "taksida"), ("by metro", "metroda")]
MOVE = {"go": (("go", "goes", "went", "going"), "bor"), "come": (("come", "comes", "came", "coming"), "kel")}


def gen_movement():
    out = []
    for (sen, suz, p), (mk, (forms, stem)), place, tense in itertools.product(
            SUBJECTS, MOVE.items(), PLACES, ["present", "past", "cont", "future", "neg_past", "q_present", "want"]):
        base, s3, past, ing = forms
        en_to, uz_dat = place[0], place[1]
        if mk == "come" and en_to not in ("home", "to school", "to work", "to Tashkent", "to the park", "to the village"):
            continue
        adv_key = {"present": "present", "past": "past", "cont": "cont", "future": "future",
                   "neg_past": "past", "q_present": "present", "want": "future"}[tense]
        for (ea, ua, pos) in ADV[adv_key]:
            if pos == "pre" and tense in ("q_present",):
                continue
            tr = random.choice(TRANSPORT) if random.random() < 0.3 else ("", "")
            pre = [ea] if pos == "pre" else []
            tail = [en_to, tr[0]] + ([ea] if pos == "end" else [])
            q = False
            if tense == "present":
                en = [sen] + pre + [s3 if p == "3s" else base] + tail; v = uz_present(stem, p)
            elif tense == "past":
                en = [sen] + pre + [past] + tail; v = uz_past(stem, p)
            elif tense == "cont":
                en = [sen, BE[p], ing] + tail; v = uz_cont(stem, p)
            elif tense == "future":
                en = [sen, "will", base] + tail; v = uz_present(stem, p)
            elif tense == "neg_past":
                en = [sen, "did not", base] + tail; v = uz_neg_past(stem, p)
            elif tense == "q_present":
                if p == "1s":
                    continue
                en = [DO[p], sen, base] + tail; v = uz_present(stem, p) + "mi"; q = True
            elif tense == "want":
                en = [sen, "wants to" if p == "3s" else "want to", base] + tail; v = uz_want(stem, p)
            out.append(("move_" + tense,) + build(en, [suz, ua, uz_dat, tr[1], v], q))
    return out


# ---------------------------------------------------------------- family: activity at a place
LOC_VERBS = {
    "work": (("work", "works", "worked", "working"), "ishla",
             ["in the city", "in the village", "in the hospital", "at school", "at the market", "in Tashkent",
              "at home", "in Germany", "in the library"]),
    "live": (("live", "lives", "lived", "living"), "yasha",
             ["in Tashkent", "in Samarkand", "in the city", "in the village", "in Germany", "in Berlin"]),
    "study": (("study", "studies", "studied", "studying"), "o'qi",
              ["at school", "at university", "in the library", "at home", "in Germany", "in Tashkent"]),
    "play": (("play", "plays", "played", "playing"), "o'yna",
             ["in the park", "in the garden", "at home", "at school"]),
    "sleep": (("sleep", "sleeps", "slept", "sleeping"), "uxla", ["at home", "in the garden"]),
}
LOC_MAP = {p[2]: p[3] for p in PLACES}


def gen_location_activity():
    out = []
    for (sen, suz, p), (vk, (forms, stem, places)), tense in itertools.product(
            SUBJECTS, LOC_VERBS.items(), ["present", "past", "cont", "neg_present", "q_present", "want"]):
        base, s3, past, ing = forms
        for pl in places:
            adv_key = {"present": "present", "past": "past", "cont": "cont", "neg_present": "present",
                       "q_present": "present", "want": "future"}[tense]
            for (ea, ua, pos) in ADV[adv_key]:
                if pos == "pre" and tense in ("neg_present", "q_present"):
                    continue
                if vk == "live" and ea not in ("", "now", "at the moment", "usually"):
                    continue
                pre = [ea] if pos == "pre" else []
                tail = [pl] + ([ea] if pos == "end" else [])
                q = False
                if tense == "present":
                    en = [sen] + pre + [s3 if p == "3s" else base] + tail; v = uz_present(stem, p)
                elif tense == "past":
                    en = [sen] + pre + [past] + tail; v = uz_past(stem, p)
                elif tense == "cont":
                    en = [sen, BE[p], ing] + tail; v = uz_cont(stem, p)
                elif tense == "neg_present":
                    en = [sen, DO[p], "not", base] + tail; v = uz_neg_present(stem, p)
                elif tense == "q_present":
                    if p == "1s":
                        continue
                    en = [DO[p], sen, base] + tail; v = uz_present(stem, p) + "mi"; q = True
                elif tense == "want":
                    en = [sen, "wants to" if p == "3s" else "want to", base] + tail; v = uz_want(stem, p)
                out.append(("loc_" + tense,) + build(en, [suz, ua, LOC_MAP[pl], v], q))
    return out


# ---------------------------------------------------------------- family: X is at/in place
def gen_be_location():
    out = []
    for (sen, suz, p) in SUBJECTS:
        for (_, _, en_at, uz_loc) in PLACES:
            for (ea, ua) in [("now", "hozir"), ("today", "bugun"), ("every morning", "har kuni ertalab"),
                             ("in the evening", "kechqurun"), ("", "")]:
                en = [sen, BE[p], en_at, ea]
                out.append(("be_loc",) + build(en, [suz, ua, uz_loc + COPULA[p]]))
                if p != "1s":
                    out.append(("be_loc_q",) + build([BE[p], sen, en_at, ea], [suz, ua, uz_loc + COPULA[p] + "mi"], True))
                out.append(("be_loc_neg",) + build([sen, BE[p], "not", en_at, ea],
                                                   [suz, ua, uz_loc + " emas" + COPULA[p]]))
    return out


# ---------------------------------------------------------------- family: adjectives
ADJ = {"big": "katta", "small": "kichik", "new": "yangi", "old": "eski", "beautiful": "chiroyli",
       "clean": "toza", "dirty": "iflos", "fast": "tez", "expensive": "qimmat", "cheap": "arzon",
       "interesting": "qiziqarli", "boring": "zerikarli", "long": "uzun", "good": "yaxshi",
       "sweet": "shirin", "red": "qizil", "delicious": "mazali", "hot": "issiq", "cold": "sovuq",
       "fresh": "yangi", "black": "qora", "white": "oq", "kind": "mehribon", "young": "yosh",
       "strict": "qattiqqo'l", "green": "yashil", "warm": "iliq", "bad": "yomon", "difficult": "qiyin",
       "easy": "oson", "quiet": "tinch", "tall": "baland"}
NOUNS = {  # en: (uz, adjs, can_my)
    "house": ("uy", ["big", "small", "new", "old", "beautiful", "clean"], True),
    "car": ("mashina", ["new", "old", "fast", "expensive", "cheap", "beautiful"], True),
    "city": ("shahar", ["big", "beautiful", "old", "clean", "quiet"], False),
    "room": ("xona", ["big", "small", "clean", "dirty", "cold", "warm"], True),
    "book": ("kitob", ["interesting", "new", "old", "expensive", "cheap", "boring"], True),
    "film": ("film", ["interesting", "new", "boring", "long", "good"], False),
    "apple": ("olma", ["sweet", "red", "big", "delicious", "cheap"], False),
    "tea": ("choy", ["hot", "cold", "sweet", "delicious"], False),
    "water": ("suv", ["cold", "hot", "clean"], False),
    "bread": ("non", ["fresh", "delicious", "cheap", "hot"], False),
    "soup": ("sho'rva", ["hot", "delicious", "cold"], False),
    "dog": ("it", ["big", "small", "black", "white"], True),
    "teacher": ("o'qituvchi", ["good", "kind", "young", "strict"], True),
    "garden": ("bog'", ["big", "beautiful", "green"], True),
    "lesson": ("dars", ["interesting", "difficult", "easy", "long", "boring"], False),
    "question": ("savol", ["difficult", "easy", "interesting"], False),
    "building": ("bino", ["tall", "new", "old", "beautiful"], False),
    "street": ("ko'cha", ["long", "quiet", "clean", "beautiful"], False),
}


def gen_adjectives():
    out = []
    for en_n, (uz_n, adjs, can_my) in NOUNS.items():
        dets = [("this", "bu", None), ("that", "o'sha", None), ("the", "", None)]
        if can_my:
            dets.append(("my", "mening", "my"))
        for (ed, ud, flag) in dets:
            uzn = possessive(uz_n, "1s") if flag else uz_n
            for a in adjs:
                ua = ADJ[a]
                for inten in ["very", "", "not", "not very", "really"] + ([] if flag else ["q", "q_very"]):
                    if inten == "very":
                        en = [ed, en_n, "is very", a]; uz = [ud, uzn, "juda", ua]; q = False
                    elif inten == "really":
                        en = [ed, en_n, "is really", a]; uz = [ud, uzn, "haqiqatan ham", ua]; q = False
                    elif inten == "":
                        en = [ed, en_n, "is", a]; uz = [ud, uzn, ua]; q = False
                    elif inten == "not":
                        en = [ed, en_n, "is not", a]; uz = [ud, uzn, ua, "emas"]; q = False
                    elif inten == "not very":
                        en = [ed, en_n, "is not very", a]; uz = [ud, uzn, "unchalik", ua, "emas"]; q = False
                    elif inten == "q":
                        en = ["is", ed, en_n, a]; uz = [ud, uzn, ua + "mi"]; q = True
                    else:
                        en = ["is", ed, en_n, "very", a]; uz = [ud, uzn, "juda", ua + "mi"]; q = True
                    out.append(("adj",) + build(en, uz, q))
                # attributive: "This is a very big house."
                if flag is None and ed in ("this", "that") and en_n not in ("tea", "water", "bread", "soup"):
                    en = [ed, "is", a_an("very"), "very", a, en_n]
                    uz = ["Bu" if ed == "this" else "U", "juda", ua, uz_n]
                    out.append(("adj_attr",) + build(en, uz))
                    en = [ed, "is", a_an(a), a, en_n]
                    out.append(("adj_attr",) + build(en, ["Bu" if ed == "this" else "U", ua, uz_n]))
    for a in ["cold", "hot", "warm", "good", "bad"]:
        for (ea, ua) in [("today", "Bugun"), ("in Tashkent", "Toshkentda"), ("in Berlin", "Berlinda"),
                         ("in winter", "Qishda"), ("in summer", "Yozda")]:
            out.append(("weather",) + build(["the weather is very", a, ea], [ua, "havo juda", ADJ[a]]))
            out.append(("weather",) + build(["the weather is", a, ea], [ua, "havo", ADJ[a]]))
            out.append(("weather",) + build(["the weather is not", a, ea], [ua, "havo", ADJ[a], "emas"]))
    return out


# ---------------------------------------------------------------- family: professions
PROF = {"doctor": "shifokor", "teacher": "o'qituvchi", "engineer": "muhandis", "driver": "haydovchi",
        "student": "talaba", "programmer": "dasturchi", "cook": "oshpaz", "farmer": "dehqon",
        "pilot": "uchuvchi", "singer": "xonanda", "artist": "rassom", "journalist": "jurnalist"}
PROF_ADJ = {"good": "yaxshi", "young": "yosh", "famous": "mashhur", "new": "yangi", "very good": "juda yaxshi"}
YOUR = {"my mother": ("your mother", "Onangiz"), "my father": ("your father", "Otangiz"),
        "my brother": ("your brother", "Akangiz"), "my sister": ("your sister", "Opangiz"),
        "my friend": ("your friend", "Do'stingiz")}


def gen_professions():
    out = []
    singular = [s for s in SUBJECTS if s[2] in ("1s", "2", "3s")]
    for (sen, suz, p) in singular:
        for en_p, uz_p in PROF.items():
            for adj in [None] + list(PROF_ADJ):
                if adj:
                    en = [sen, BE[p], a_an(adj), adj, en_p]; uz = [suz, PROF_ADJ[adj], uz_p + COPULA[p]]
                else:
                    en = [sen, BE[p], a_an(en_p), en_p]; uz = [suz, uz_p + COPULA[p]]
                out.append(("prof",) + build(en, uz))
            out.append(("prof_neg",) + build([sen, BE[p], "not", a_an(en_p), en_p], [suz, uz_p, "emas" + COPULA[p]]))
            if sen in YOUR:
                ye, yu = YOUR[sen]
                out.append(("prof_q",) + build(["is", ye, a_an(en_p), en_p], [yu, uz_p + "mi"], True))
                out.append(("prof_q",) + build(["is", ye, "a good", en_p], [yu, "yaxshi", uz_p + "mi"], True))
            # "My father works as a doctor"
            out.append(("prof",) + build([sen, "works as" if p == "3s" else "work as", a_an(en_p), en_p],
                                         [suz, uz_p, "bo'lib", uz_present("ishla", p)]))
            out.append(("prof",) + build([sen, "wants to be" if p == "3s" else "want to be", a_an(en_p), en_p],
                                         [suz, uz_p, "bo'lishni", uz_present("xohla", p)]))
    return out


# ---------------------------------------------------------------- family: possession
POSS = [("I", "Mening", "1s"), ("you", "Sizning", "2"), ("he", "Uning", "3s"), ("she", "Uning", "3s"),
        ("we", "Bizning", "1p"), ("they", "Ularning", "3p"), ("my brother", "Akamning", "3s"),
        ("my friend", "Do'stimning", "3s"), ("Ali", "Alining", "3s"), ("Laylo", "Laylonung", "3s"),
        ("my sister", "Opamning", "3s")]
POSS = [x if x[0] != "Laylo" else ("Laylo", "Layloning", "3s") for x in POSS]
HAVE_N = {"car": ("mashina", "cars"), "house": ("uy", "houses"), "dog": ("it", "dogs"),
          "computer": ("kompyuter", "computers"), "phone": ("telefon", "phones"),
          "bicycle": ("velosiped", "bicycles"), "book": ("kitob", "books"), "cat": ("mushuk", "cats"),
          "garden": ("bog'", "gardens"), "brother": ("aka", "brothers"), "sister": ("opa", "sisters"),
          "friend": ("do'st", "friends"), "question": ("savol", "questions"), "idea": ("fikr", "ideas")}
HAVE_ADJ = ["new", "old", "big", "small", "red", "black", "white", "beautiful", "good"]
NUMS = [("two", "ikkita"), ("three", "uchta"), ("four", "to'rtta"), ("five", "beshta"), ("many", "ko'p")]


def gen_possession():
    out = []
    for (pen, puz, p) in POSS:
        for n, (uzn, pl) in HAVE_N.items():
            un = possessive(uzn, p)
            human = n in ("brother", "sister", "friend")
            adjs = [] if human else HAVE_ADJ
            if n in ("question", "idea"):
                adjs = ["good", "interesting", "new"] if n == "idea" else ["difficult", "interesting"]
            for adj in [None] + adjs:
                ua = ADJ.get(adj, "") if adj else ""
                if adj == "young":
                    ua = "yosh"
                if adj:
                    en_np = [a_an(adj), adj, n]
                else:
                    en_np = [a_an(n), n]
                out.append(("have",) + build([pen, HAVE[p]] + en_np, [puz, ua, un, "bor"]))
            out.append(("have_neg",) + build([pen, DO[p], "not have", a_an(n), n], [puz, un, "yo'q"]))
            if p != "1s":
                out.append(("have_q",) + build([DO[p], pen, "have", a_an(n), n], [puz, un, "bormi"], True))
            for (en_num, uz_num) in NUMS:
                for adj in [None] + (["good"] if human else ["big", "new", "small"]):
                    ua = ADJ[adj] if adj else ""
                    out.append(("have_num",) + build([pen, HAVE[p], en_num, adj or "", pl],
                                                     [puz, uz_num, ua, un, "bor"]))
    return out


# ---------------------------------------------------------------- hand-written everyday phrases
HAND = """Good morning, how are you today?	Xayrli tong, bugun qalaysiz?
What is your name, my friend?	Ismingiz nima, do'stim?
My name is Diyorbek, nice to meet you.	Mening ismim Diyorbek, tanishganimdan xursandman.
Where is the nearest bus stop?	Eng yaqin avtobus bekati qayerda?
How much does this book cost?	Bu kitob qancha turadi?
I do not understand this word.	Men bu so'zni tushunmayman.
Can you speak a little slower?	Biroz sekinroq gapira olasizmi?
Please write it on the paper.	Iltimos, buni qog'ozga yozing.
Thank you very much for your help.	Yordamingiz uchun katta rahmat.
I am very happy to see you.	Sizni ko'rganimdan juda xursandman.
What time is it now, please?	Iltimos, hozir soat necha?
I am learning German every single day.	Men har kuni nemis tilini o'rganaman.
German is a very beautiful language.	Nemis tili juda chiroyli til.
I want to travel to Germany.	Men Germaniyaga sayohat qilmoqchiman.
Where do you live now, Ali?	Ali, hozir qayerda yashaysiz?
I live in Tashkent with my family.	Men oilam bilan Toshkentda yashayman.
How old is your little brother?	Ukangiz necha yoshda?
My little brother is ten years old.	Ukam o'n yoshda.
I was born in Samarkand.	Men Samarqandda tug'ilganman.
What do you do on weekends?	Dam olish kunlari nima qilasiz?
I usually read books on weekends.	Men dam olish kunlari odatda kitob o'qiyman.
Let us go to the park together.	Keling, birga parkka boramiz.
Please open the window, it is hot.	Iltimos, derazani oching, issiq.
Please close the door behind you.	Iltimos, orqangizdan eshikni yoping.
Do not be late for the lesson.	Darsga kechikmang.
The lesson starts at nine o'clock.	Dars soat to'qqizda boshlanadi.
The shop opens at eight in the morning.	Do'kon ertalab soat sakkizda ochiladi.
The train leaves at ten tonight.	Poyezd bugun kechqurun soat o'nda jo'naydi.
I am hungry, let us eat something.	Men ochman, keling, biror narsa yeymiz.
I am very tired after work today.	Bugun ishdan keyin juda charchadim.
She is my best friend at school.	U maktabdagi eng yaxshi do'stim.
This is the best day of my life.	Bu hayotimdagi eng yaxshi kun.
Where can I buy a train ticket?	Poyezd chiptasini qayerdan sotib olsam bo'ladi?
I would like a cup of tea.	Men bir piyola choy istardim.
The food in this restaurant is delicious.	Bu restorandagi ovqat mazali.
My favorite color is green.	Mening sevimli rangim yashil.
My favorite subject is mathematics.	Mening sevimli fanim matematika.
What is your favorite film?	Sizning sevimli filmingiz qaysi?
It is raining outside right now.	Hozir tashqarida yomg'ir yog'yapti.
It is snowing in the mountains today.	Bugun tog'larda qor yog'yapti.
The sun is shining very brightly today.	Bugun quyosh juda charaqlab turibdi.
Happy birthday to you, my dear friend.	Tug'ilgan kuningiz bilan, aziz do'stim.
I am sorry, I am late again.	Kechirasiz, men yana kechikdim.
Excuse me, where is the library?	Kechirasiz, kutubxona qayerda?
Turn left at the next street.	Keyingi ko'chada chapga buriling.
Go straight and then turn right.	To'g'riga yuring, keyin o'ngga buriling.
The hospital is not far from here.	Kasalxona bu yerdan uzoq emas.
My house is near the big park.	Mening uyim katta park yaqinida.
We will meet at the station tomorrow.	Biz ertaga bekatda uchrashamiz.
I need to finish my homework today.	Men bugun uy vazifamni tugatishim kerak.
Our teacher gave us a difficult task.	O'qituvchimiz bizga qiyin vazifa berdi.
The students are writing a test now.	Talabalar hozir test yozishyapti.
I passed the exam with a good mark.	Men imtihondan yaxshi baho bilan o'tdim.
He wants to become a famous programmer.	U mashhur dasturchi bo'lmoqchi.
Artificial intelligence is changing the world.	Sun'iy intellekt dunyoni o'zgartiryapti.
I am building a translator with Python.	Men Python bilan tarjimon yaratyapman.
This model translates English into Uzbek.	Bu model inglizchani o'zbekchaga tarjima qiladi.
Machine learning is very interesting for me.	Mashinali o'qitish men uchun juda qiziqarli.
My computer is old but fast.	Mening kompyuterim eski, lekin tez.
Please call me in the evening.	Iltimos, menga kechqurun qo'ng'iroq qiling.
I will send you a message tomorrow.	Men sizga ertaga xabar yuboraman.
She sends me a message every day.	U menga har kuni xabar yuboradi.
My mother cooks very delicious plov.	Onam juda mazali palov pishiradi.
We eat plov together every Sunday.	Biz har yakshanba birga palov yeymiz.
Uzbek bread is the most delicious bread.	O'zbek noni eng mazali non.
Tashkent is the capital of Uzbekistan.	Toshkent O'zbekistonning poytaxti.
Berlin is the capital of Germany.	Berlin Germaniyaning poytaxti.
Samarkand is a very old city.	Samarqand juda qadimiy shahar.
There are many tourists in Samarkand.	Samarqandda sayyohlar ko'p.
There is a big tree in our garden.	Bizning bog'imizda katta daraxt bor.
There are three rooms in our house.	Bizning uyimizda uchta xona bor.
There is no water in the bottle.	Shishada suv yo'q.
I drink a glass of water every morning.	Men har kuni ertalab bir stakan suv ichaman.
He runs in the park every morning.	U har kuni ertalab parkda yuguradi.
We often walk along the river.	Biz tez-tez daryo bo'ylab sayr qilamiz.
The children are swimming in the river.	Bolalar daryoda suzishyapti.
The cat is sleeping on the sofa.	Mushuk divanda uxlayapti.
The dog is playing in the garden.	It bog'da o'ynayapti.
The birds are singing in the tree.	Qushlar daraxtda sayrayapti.
I am afraid of big dogs.	Men katta itlardan qo'rqaman.
My sister is afraid of the dark.	Opam qorong'ilikdan qo'rqadi.
I feel much better today, thank you.	Bugun o'zimni ancha yaxshi his qilyapman, rahmat.
I have a headache this morning.	Bugun ertalab boshim og'riyapti.
You should drink more water every day.	Siz har kuni ko'proq suv ichishingiz kerak.
I go to bed at eleven o'clock.	Men soat o'n birda uxlashga yotaman.
I wake up at seven every morning.	Men har kuni ertalab soat yettida uyg'onaman.
Winter in Tashkent is not very cold.	Toshkentda qish unchalik sovuq emas.
Summer in Uzbekistan is very hot.	O'zbekistonda yoz juda issiq.
My favorite season is spring.	Mening sevimli faslim bahor.
I like to listen to music.	Men musiqa tinglashni yoqtiraman.
She likes to draw pictures of cats.	U mushuklarning rasmini chizishni yoqtiradi.
They like to play football after school.	Ular darsdan keyin futbol o'ynashni yoqtiradilar.
We like to travel in summer.	Biz yozda sayohat qilishni yoqtiramiz.
I do not like cold weather.	Men sovuq havoni yoqtirmayman.
Do you want some more tea?	Yana choy ichasizmi?
Yes, please, with a little sugar.	Ha, iltimos, ozgina shakar bilan.
No, thank you, I am full.	Yo'q, rahmat, men to'qman.
Can I help you with something?	Sizga biror narsada yordam bera olamanmi?
I am looking for a good book.	Men yaxshi kitob qidiryapman.
This dress is too expensive for me.	Bu ko'ylak men uchun juda qimmat.
Do you have a smaller size?	Sizda kichikroq o'lcham bormi?
I will take this one, please.	Iltimos, men buni olaman.
See you tomorrow at school, Laylo.	Ertaga maktabda ko'rishamiz, Laylo.
Have a nice day, my friend.	Kuningiz yaxshi o'tsin, do'stim.
Good night and sweet dreams.	Xayrli tun, shirin tushlar ko'ring.
"""


def main():
    pairs = []
    fams = {"verb": gen_verb_sentences(), "move": gen_movement(), "loc": gen_location_activity(),
            "be_loc": gen_be_location(), "adj": gen_adjectives(), "prof": gen_professions(),
            "have": gen_possession()}

    def nwords(s):
        return len(re.findall(r"[A-Za-z']+", s))

    quota = {"verb": 760, "move": 210, "loc": 170, "be_loc": 110, "adj": 230, "prof": 120, "have": 160}
    seen = set()
    for k, lst in fams.items():
        lst = [x for x in lst if 5 <= nwords(x[1]) <= 8]
        random.shuffle(lst)
        # stratify by sub-family so every tense appears
        by = {}
        for x in lst:
            by.setdefault(x[0], []).append(x)
        picked, i = [], 0
        keys = list(by)
        while len(picked) < quota[k] and any(by.values()):
            key = keys[i % len(keys)]; i += 1
            if by[key]:
                x = by[key].pop()
                if x[1].lower() not in seen:
                    seen.add(x[1].lower()); picked.append(x)
        pairs += [(en, uz) for _, en, uz in picked]
        print(f"{k:8s} candidates={len(lst):6d} picked={len(picked)}")
    hand = [l.split("\t") for l in HAND.strip().splitlines()]
    hand = [(e, u) for e, u in hand if 5 <= nwords(e) <= 8 and e.lower() not in seen]
    print("hand    ", len(hand))
    pairs += hand
    random.shuffle(pairs)
    os.makedirs("data", exist_ok=True)
    with open("data/en_uz.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["en", "uz"])
        w.writerows(pairs)
    print("total", len(pairs), "-> data/en_uz.tsv")


if __name__ == "__main__":
    main()
