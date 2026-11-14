// German learning helpers: noun-gender coloring + phrase of the day.

const ARTICLES = { der: "g-m", die: "g-f", das: "g-n" };

/**
 * Split text into word / non-word tokens and attach a gender class for German.
 * "das kleine Haus" -> das (g-n) kleine Haus (g-n): the article colors the next
 * capitalized word (the noun), skipping up to 3 lowercase adjectives in between.
 */
export function annotate(text, lang) {
  const parts = text.match(/[\p{L}\p{N}'’ʻ-]+|[^\p{L}\p{N}'’ʻ-]+/gu) || [];
  let pending = null, gap = 0;
  return parts.map((t, i) => {
    const isWord = /[\p{L}\p{N}]/u.test(t);
    const tok = { t, isWord, cls: "" };
    if (!isWord || lang !== "de") return tok;
    const low = t.toLowerCase();
    if (ARTICLES[low]) {
      tok.cls = ARTICLES[low]; tok.article = low; pending = ARTICLES[low]; gap = 0;
      return tok;
    }
    if (pending) {
      const first = i === 0;
      if (/^\p{Lu}/u.test(t) && !first) { tok.cls = pending; tok.gender = pending; pending = null; }
      else if (++gap > 3) pending = null;
    }
    return tok;
  });
}

export const GENDER_ARTICLE = { "g-m": "der", "g-f": "die", "g-n": "das" };

// 30 everyday phrases (DE / EN / UZ)
export const PHRASES = [
  ["Guten Morgen! Wie geht es dir?", "Good morning! How are you?", "Xayrli tong! Qalaysan?"],
  ["Ich lerne seit einem Monat Deutsch.", "I have been learning German for a month.", "Men bir oydan beri nemis tilini o'rganyapman."],
  ["Können Sie das bitte wiederholen?", "Could you repeat that, please?", "Iltimos, buni takrorlay olasizmi?"],
  ["Ich verstehe das nicht.", "I don't understand that.", "Men buni tushunmayapman."],
  ["Wie heißt das auf Deutsch?", "What is this called in German?", "Bu nemischa nima deyiladi?"],
  ["Ich komme aus Usbekistan.", "I come from Uzbekistan.", "Men O'zbekistondanman."],
  ["Wo ist der Bahnhof?", "Where is the train station?", "Vokzal qayerda?"],
  ["Was kostet das?", "How much does this cost?", "Bu qancha turadi?"],
  ["Ich hätte gern einen Kaffee.", "I would like a coffee.", "Men bitta qahva istardim."],
  ["Die Rechnung, bitte.", "The bill, please.", "Hisobni bering, iltimos."],
  ["Entschuldigung, ich habe mich verspätet.", "Sorry, I am late.", "Kechirasiz, kechikdim."],
  ["Das Wetter ist heute schön.", "The weather is nice today.", "Bugun havo yaxshi."],
  ["Ich habe keine Zeit.", "I don't have time.", "Mening vaqtim yo'q."],
  ["Hast du am Wochenende Zeit?", "Do you have time on the weekend?", "Dam olish kunlari vaqting bormi?"],
  ["Ich freue mich auf das Wochenende.", "I am looking forward to the weekend.", "Dam olish kunlarini intiqlik bilan kutyapman."],
  ["Das Buch ist sehr interessant.", "The book is very interesting.", "Kitob juda qiziqarli."],
  ["Mein Bruder ist Ingenieur.", "My brother is an engineer.", "Akam muhandis."],
  ["Ich wohne in Taschkent.", "I live in Tashkent.", "Men Toshkentda yashayman."],
  ["Wie spät ist es?", "What time is it?", "Soat necha?"],
  ["Ich bin müde.", "I am tired.", "Men charchadim."],
  ["Viel Glück bei der Prüfung!", "Good luck with the exam!", "Imtihonda omad!"],
  ["Ich möchte nach Berlin fahren.", "I want to go to Berlin.", "Men Berlinga bormoqchiman."],
  ["Die Stadt ist sehr groß.", "The city is very big.", "Shahar juda katta."],
  ["Ich trinke jeden Morgen Tee.", "I drink tea every morning.", "Men har kuni ertalab choy ichaman."],
  ["Kannst du mir helfen?", "Can you help me?", "Menga yordam bera olasanmi?"],
  ["Das ist eine gute Idee.", "That is a good idea.", "Bu yaxshi fikr."],
  ["Ich spreche ein bisschen Deutsch.", "I speak a little German.", "Men biroz nemischa gapiraman."],
  ["Wir sehen uns morgen.", "See you tomorrow.", "Ertaga ko'rishamiz."],
  ["Der Hund spielt im Garten.", "The dog is playing in the garden.", "It bog'da o'ynayapti."],
  ["Gute Nacht und schöne Träume!", "Good night and sweet dreams!", "Xayrli tun, shirin tushlar!"],
];

export function phraseOfTheDay(d = new Date()) {
  const day = Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 864e5);
  return PHRASES[day % PHRASES.length];
}

export const EXAMPLES = {
  en: ["I drink tea every morning.", "My sister is learning German now.", "We will go to Tashkent tomorrow.",
       "My father works in the hospital.", "Do you have a new phone?"],
  uz: ["Men har kuni kitob o'qiyman.", "Bugun havo juda sovuq.", "Akam Germaniyada ishlaydi.",
       "Ertaga maktabda ko'rishamiz."],
  de: ["Ich lerne jeden Tag Deutsch.", "Das kleine Haus ist sehr alt.", "Die Katze schläft auf dem Sofa.",
       "Wo ist der nächste Supermarkt?"],
};
