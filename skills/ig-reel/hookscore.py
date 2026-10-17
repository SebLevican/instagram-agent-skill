#!/usr/bin/env python3
"""
hookscore.py - score the first line of a Reel on the five things strong hooks
have in common, and rank a batch of them against each other.

What this is:  five local heuristics, computed on your machine from the text
alone. They measure properties that hooks which hold attention tend to share -
a length you can say in under three seconds, a concrete marker, something at stake,
the payload at the front rather than the back, and a viewer to aim it at.

What this is NOT:  a view predictor. It was tested against 74 real short-form
hooks, transcribed from the first three seconds of the top eight and bottom
eight performers on five channels. Separating a real hook from a deliberately
bad one it does well: AUC 0.83, and 9 of 10 written-to-be-bad hooks scored
below the real median. Separating a good creator's hits from that same
creator's misses it barely does at all: AUC 0.56, where 0.50 is a coin flip.

So use it for what it measured well. It catches greetings, preambles, hooks
with nothing concrete in them and hooks that take five seconds to say. It will
not tell you which of two decent hooks will travel, and nothing that reads text
can, because that is decided by your face, your edit, your audio and who
Instagram shows it to. Trust the retention graph over this script.

Each check returns 0 to 100. Higher is better.

Spanish hooks get their own word lists, number formats (37.075, 9,4%) and a
faster speaking rate. The language is detected from the text, or forced with
--lang. A batch is detected once, so every hook in it is ranked on one scale.

Usage
  python3 hookscore.py hooks.txt              # one hook per line, ranked
  python3 hookscore.py --hook "I lost $18,000 on one missing contract."
  python3 hookscore.py --hook "9 de cada 10 pegas esconden el sueldo." --lang es
  pbpaste | python3 hookscore.py -
  python3 hookscore.py hooks.txt --json
"""

import argparse
import json
import re
import statistics
import sys

WORD_RE = re.compile(r"[A-Za-z0-9$%'’-]+")
NUMBER_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?"                       # money, whole
    r"|\b\d[\d,]*(?:\.\d+)?\s?"                      # a figure, with or
    r"(?:%|k\b|x\b|hrs?\b|hours?\b|mins?\b|minutes?\b"  # without a unit
    r"|days?\b|weeks?\b|months?\b|years?\b)?",
    re.IGNORECASE)
PROPER_RE = re.compile(r"(?<!^)\b[A-Z][a-z]{2,}\b")
HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

# Spoken hooks say their numbers out loud. "Zero dollars" and "twenty grand"
# are as concrete as "$0" and "$20,000", and counting only digits missed them.
# "one" and "first" are deliberately absent: they are filler far more often
# than they are a quantity.
SPOKEN_NUMBERS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen", "half", "twice", "triple",
}
MONEY_WORDS = {
    "dollars", "dollar", "bucks", "grand", "percent", "cents",
    "millionaire", "billionaire", "revenue", "profit", "salary", "rent",
}

# Words that put something on the line. A hook with none of these is a
# statement; a hook with one is a reason to keep watching.
STAKES = {
    "stop", "never", "wrong", "mistake", "mistakes", "lost", "lose", "losing",
    "cost", "costs", "broke", "broken", "failed", "failure", "fail", "nobody",
    "no", "not", "don't", "dont", "doesn't", "didn't", "can't", "won't",
    "quit", "quitting", "fired", "deleted", "delete", "killed", "kills", "kill",
    "replaced", "replaces", "cut", "beat", "free", "paid", "charged", "hired",
    "saved", "first",
    "banned", "illegal", "worst", "hate", "hated", "wasted", "waste", "scam",
    "lie", "lied", "lying", "truth", "secret", "hidden", "stole", "stolen",
    "before", "until", "instead", "but", "except", "unless", "problem",
    "risk", "danger", "warning", "regret", "wish", "should", "shouldn't",
    "still", "already", "only", "without", "versus", "vs", "actually",
}

# Openers that spend the first second saying nothing.
WEAK_OPENERS = [
    "so", "ok", "okay", "hey", "hi", "hello", "guys", "yo", "alright",
    "welcome", "today", "basically", "honestly", "look", "listen", "um",
    "just", "let", "lets", "let's", "i wanted", "i want", "one of",
    "have you", "did you", "do you", "are you", "in this", "in today",
    "the thing", "a lot", "there is", "there are", "this is", "it is",
    "as a", "when it", "if you've", "you know",
]

# Imperatives that earn the front position.
IMPERATIVES = {
    "stop", "steal", "copy", "delete", "try", "watch", "read", "save",
    "use", "build", "make", "write", "send", "take", "start", "quit",
    "never", "always", "don't", "dont", "do", "put", "run", "check",
}

DEALBREAKERS = [
    (re.compile(r"(?i)^\s*(?:stop scrolling|don'?t scroll)"),
     "Opens with \"stop scrolling\". Asking for attention proves you have not earned it."),
    (re.compile(r"(?i)\b(?:in (?:this|today'?s) (?:video|reel)|i'?m going to show you|i'?ll show you how)\b"),
     "Video preamble. Delete it and open on the payoff."),
    (re.compile(r"(?i)^\s*(?:hey |hi |what'?s up |welcome )"),
     "Greeting. Nobody came to the feed to be greeted."),
    (HASHTAG_RE,
     "Hashtag in the hook. Hashtags belong at the bottom of the caption, if anywhere."),
    (EMOJI_RE,
     "Emoji in the hook. On-screen text at hook size has room for words or for an emoji, not both."),
]


# ---------------------------------------------------------------------------
# Spanish. Same five checks, its own vocabulary. Written for Latin American
# Spanish, Chilean in particular ("pega", "lucas"), and neutral enough for the
# rest. Spoken Spanish runs faster in words per second than English and needs
# more function words ("de", "en", "la") to say the same thing, so the length
# window is wider and the clock runs at ~190 wpm instead of 165.

WORD_RE_ES = re.compile(r"[\w$%'’-]+")
NUMBER_RE_ES = re.compile(
    r"\$\s?\d[\d.]*(?:,\d+)?"                        # $1.500.000
    r"|\b\d[\d.]*(?:,\d+)?\s?"                       # 37.075, 9,4
    r"(?:%|k\b|x\b|hrs?\b|horas?\b|min(?:utos)?\b"
    r"|d[ií]as?\b|semanas?\b|mes(?:es)?\b|años?\b|lucas?\b|palos?\b|uf\b)?",
    re.IGNORECASE)
PROPER_RE_ES = re.compile(r"(?<!^)\b[A-ZÁÉÍÓÚÑ][a-záéíóúñü]{2,}\b")

# "uno", "una" and "primero" are left out for the same reason "one" is.
SPOKEN_NUMBERS_ES = {
    "cero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "quince", "veinte", "treinta", "cuarenta",
    "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien", "ciento",
    "cientos", "mil", "miles", "millón", "millon", "millones", "docena",
    "mitad", "doble", "triple",
}
MONEY_WORDS_ES = {
    "pesos", "dólares", "dolares", "lucas", "palos", "porciento", "sueldo",
    "sueldos", "salario", "salarios", "arriendo", "plata", "ingresos",
    "ganancia", "ganancias", "uf",
}

STAKES_ES = {
    "deja", "dejes", "nunca", "jamás", "jamas", "mal", "error", "errores",
    "perdí", "perdi", "pierdes", "perder", "perdiste", "cuesta", "costó",
    "costo", "fracaso", "fracasé", "nadie", "no", "ni", "sin", "despedido",
    "despidieron", "despidos", "borra", "borré", "eliminé", "reemplazó",
    "reemplaza", "gratis", "pagué", "pagas", "cobran", "contrataron",
    "contratan", "ahorré", "prohibido", "ilegal", "peor", "odio", "odiaba",
    "desperdicio", "perdiendo", "estafa", "mentira", "mienten", "mintieron",
    "verdad", "secreto", "oculto", "esconden", "esconde", "robó", "antes",
    "hasta", "solo", "sólo", "pero", "excepto", "salvo", "problema", "riesgo",
    "peligro", "cuidado", "arrepiento", "ojalá", "debería", "deberías",
    "todavía", "aún", "ya", "versus", "vs", "contra", "realmente", "trampa",
    "cayó", "cae", "caen", "cayeron", "caído", "cayendo", "bajó", "desplomó",
    "desapareció", "murió", "muerto", "rechazan", "rechazaron",
    "ignoran", "engaña", "vendieron", "primero", "basta", "olvida",
}

WEAK_OPENERS_ES = [
    "hola", "holi", "oye", "bueno", "ok", "okey", "chicos", "chiquillos",
    "amigos", "gente", "hoy", "en este", "en el video", "básicamente",
    "honestamente", "escucha", "eh", "solo quería", "quiero", "quería",
    "les quiero", "te quiero", "una de", "uno de", "has", "sabías",
    "alguna vez", "hay", "esto es", "es que", "como", "cuando", "sabes",
    "bienvenidos", "bienvenidas", "qué tal", "a ver",
]

IMPERATIVES_ES = {
    "deja", "roba", "copia", "borra", "prueba", "mira", "lee",
    "guarda", "usa", "arma", "haz", "escribe", "manda", "toma", "empieza",
    "nunca", "siempre", "no", "pon", "revisa", "cambia", "postula",
    "pregunta", "olvida", "sube", "agrega", "elimina", "busca", "evita",
}

# Spanish drops the pronoun, so "postulas a 10" talks to the viewer without
# ever saying "tú". Second-person verbs are too varied to catch by ending
# ("-as" also ends "ofertas"), so the common ones are listed.
ADDRESS_ES = re.compile(
    r"\b(tú|tu|tus|te|ti|contigo|usted|ustedes|vos|eres|estás|tienes"
    r"|puedes|sabes|quieres|buscas|postulas|ganas|trabajas|necesitas|crees"
    r"|haces|pones|mandas|pierdes|pagas|vives|llevas|ves|vas)\b")
FIRST_PERSON_ES = re.compile(
    r"\b(yo|mi|mis|me|nosotros|nosotras|nuestro|nuestra|llevo|tengo|veo"
    r"|\w{3,}é)\b")                      # revisé, analicé, miré

DEALBREAKERS_ES = [
    (re.compile(r"(?i)^\s*[¿¡]?\s*(?:deja de scrollear|no scrollees|para de scrollear|no sigas bajando)"),
     "Abre con \"deja de scrollear\". Pedir atención demuestra que no te la ganaste."),
    (re.compile(r"(?i)\b(?:en este (?:video|reel)|en el video de hoy|te voy a mostrar|les voy a mostrar"
                r"|hoy (?:te|les) traigo)\b"),
     "Preámbulo de video. Bórralo y abre con el resultado."),
    (re.compile(r"(?i)^\s*[¿¡]?\s*(?:hola|holi|qué tal|que tal|bienvenid)"),
     "Saludo. Nadie entró al feed para que lo saluden."),
    (HASHTAG_RE,
     "Hashtag en el hook. Los hashtags van al final del caption, si es que van."),
    (EMOJI_RE,
     "Emoji en el hook. El texto en pantalla a tamaño de hook tiene espacio para palabras o para un emoji, no para ambos."),
]

# Accents are often skipped when typing ("desplomo", "pagan mas"), so Spanish
# words are compared without them. The ñ stays: "año" and "ano" are not the
# same word.
_FOLD = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")


def fold(s):
    return s.translate(_FOLD)


def _folded(items):
    return type(items)(fold(x) for x in items)


LANGS = {
    "en": {
        "word_re": WORD_RE, "number_re": NUMBER_RE, "proper_re": PROPER_RE,
        "thousands": ",", "spoken": SPOKEN_NUMBERS, "money": MONEY_WORDS,
        "stakes": STAKES, "weak": WEAK_OPENERS, "imperatives": IMPERATIVES,
        "dealbreakers": DEALBREAKERS, "wps": 2.75, "max_words": 12,
        "max_chars": 60, "fold": lambda s: s,
    },
    "es": {
        "word_re": WORD_RE_ES, "number_re": NUMBER_RE_ES, "proper_re": PROPER_RE_ES,
        "thousands": ".", "spoken": _folded(SPOKEN_NUMBERS_ES),
        "money": _folded(MONEY_WORDS_ES), "stakes": _folded(STAKES_ES),
        "weak": _folded(WEAK_OPENERS_ES), "imperatives": _folded(IMPERATIVES_ES),
        "dealbreakers": DEALBREAKERS_ES, "wps": 3.2, "max_words": 14,
        "max_chars": 70, "fold": fold,
    },
}

ES_HINTS = {"de", "la", "el", "que", "en", "los", "las", "del", "por", "con",
            "para", "una", "es", "se", "te", "tu", "y", "no", "más", "pero"}
EN_HINTS = {"the", "of", "and", "to", "in", "is", "you", "your", "that", "it",
            "for", "on", "with", "this", "my", "i", "was", "are", "but", "not"}


def detect_lang(text):
    """Spanish if it carries Spanish marks or more Spanish function words."""
    if re.search(r"[ñ¿¡áéíóú]", text.lower()):
        return "es"
    toks = re.findall(r"\w+", text.lower())
    es = sum(t in ES_HINTS for t in toks)
    en = sum(t in EN_HINTS for t in toks)
    return "es" if es > en else "en"


def clamp(n):
    return max(0.0, min(100.0, n))


def words(text, p=LANGS["en"]):
    # "$18,000" is one word when it is spoken, so it is one word here too.
    # In Spanish the thousands separator is the dot: "37.075" is one word.
    sep = re.escape(p["thousands"])
    return p["word_re"].findall(re.sub(rf"(?<=\d){sep}(?=\d)", "", text))


def check_length(text, p=LANGS["en"]):
    """A hook has to land before the thumb moves. Roughly two seconds."""
    w = words(text, p)
    n = len(w)
    secs = n / p["wps"]                  # ~165 wpm in English, ~190 in Spanish
    chars = len(text.strip())
    top = p["max_words"]
    if 5 <= n <= top:
        score = 100.0
    elif n < 5:
        score = clamp(100 - (5 - n) * 20)
    else:
        score = clamp(100 - (n - top) * 11)
    if chars > p["max_chars"]:           # two lines of big on-screen text
        score -= 12
    return clamp(score), f"{n} words, {chars} chars, ~{secs:.1f}s spoken (want 5-{top} words)"


def check_specificity(text, p=LANGS["en"]):
    """One concrete thing beats three abstract ones."""
    nums = [n.strip() for n in p["number_re"].findall(text) if n.strip()]
    propers = set(p["proper_re"].findall(text))
    low = [p["fold"](w.lower().strip("'’")) for w in words(text, p)]
    spoken = [w for w in low if w in p["spoken"] or w in p["money"]]
    hits = len(nums) + len(propers) + len(spoken)
    score = 15.0 if hits == 0 else clamp(45 + hits * 30)
    found = ", ".join(nums[:2] + sorted(propers)[:2] + spoken[:2])
    return score, (f"{hits} concrete marker(s)" + (f": {found}" if found else
                   " - no number, no name, nothing checkable"))


def check_stakes(text, p=LANGS["en"]):
    """Tension, cost, negation. Something the viewer might lose."""
    w = [p["fold"](x.lower().strip("'’")) for x in words(text, p)]
    hits = [x for x in w if x in p["stakes"]]
    markers = sorted(set(hits))
    if re.search(r"\$\s?\d", text):
        markers.append("a price")
    n = len(markers)
    score = {0: 20.0, 1: 70.0}.get(n, 100.0)
    detail = f"{n} tension marker(s)" + (f": {', '.join(markers[:4])}" if markers else
                                         " - nothing is at stake in this line")
    return clamp(score), detail


def check_frontload(text, p=LANGS["en"]):
    """The interesting word cannot be in position nine."""
    w = words(text, p)
    if not w:
        return 0.0, "empty"
    low = [p["fold"](x.lower().strip("'’")) for x in w]
    penalty = 0
    hit_opener = None
    if p is LANGS["en"]:
        opener = " ".join(low[:2])
        matches = lambda weak: opener.startswith(weak) or low[0] == weak
    else:
        # Whole words only, so "hay" does not catch "hayan" and "como" does
        # not catch "comodín".
        opener = " ".join(low[:3]) + " "
        matches = lambda weak: opener.startswith(weak + " ")
    for weak in p["weak"]:
        if matches(weak):
            penalty, hit_opener = 30, weak
            break
    payload = None
    for i, token in enumerate(low):
        if (token in p["stakes"] or token in p["spoken"] or token in p["money"]
                or p["number_re"].match(w[i]) or (i and p["proper_re"].match(w[i]))):
            payload = i
            break
    if payload is None:
        base = 30.0
        where = "no payload word anywhere in the line"
    elif payload <= 3:
        base = 100.0
        where = f"payload at word {payload + 1}"
    elif payload <= 6:
        base = 70.0
        where = f"payload at word {payload + 1}, could move forward"
    else:
        base = 40.0
        where = f"payload at word {payload + 1}, too late"
    detail = where + (f"; weak opener \"{hit_opener}\"" if hit_opener else "")
    return clamp(base - penalty), detail


def check_address(text, p=LANGS["en"]):
    """Aimed at one viewer, or floating in the air."""
    low = text.lower()
    w = [p["fold"](x.lower().strip("'’")) for x in words(text, p)]
    spanish = p is LANGS["es"]
    you = ADDRESS_ES if spanish else re.compile(r"\b(you|your|you're|youre|yourself)\b")
    me = FIRST_PERSON_ES if spanish else re.compile(r"\b(i|my|me|we|our)\b")
    if you.search(low):
        return 100.0, "speaks to the viewer"
    if w and w[0] in p["imperatives"]:
        return 90.0, f"imperative opener (\"{w[0]}\")"
    if me.search(low):
        return 70.0, "first person, no viewer named"
    return 35.0, "third person, nobody in the room"


CHECKS = ["LENGTH", "SPECIFICITY", "STAKES", "FRONTLOAD", "ADDRESS"]


def run(text, lang=None):
    p = LANGS[lang or detect_lang(text)]
    results = {
        "LENGTH": check_length(text, p),
        "SPECIFICITY": check_specificity(text, p),
        "STAKES": check_stakes(text, p),
        "FRONTLOAD": check_frontload(text, p),
        "ADDRESS": check_address(text, p),
    }
    flags = [msg for pattern, msg in p["dealbreakers"] if pattern.search(text)]
    scores = [results[c][0] for c in CHECKS]
    # The weakest property caps the hook, same logic as detect.py: one bad
    # property is enough for the thumb to keep moving.
    overall = statistics.mean(scores) * 0.6 + min(scores) * 0.4 - len(flags) * 15
    overall = clamp(overall)
    verdict = "STRONG" if overall >= 70 and min(scores) >= 55 and not flags else (
        "OK" if overall >= 50 else "WEAK")
    return results, overall, verdict, flags


def bar(score, width=24):
    filled = round(score / 100 * width)
    return "#" * filled + "." * (width - filled)


def render_one(text, results, overall, verdict, flags, out=sys.stdout):
    print("\nHOOK SCORE", file=out)
    print("=" * 62, file=out)
    print(f"  \"{text.strip()}\"\n", file=out)
    for name in CHECKS:
        score, detail = results[name]
        print(f"  {name:<13} {bar(score)} {score:5.1f}", file=out)
        print(f"  {'':<13} {detail}", file=out)
    print("-" * 62, file=out)
    print(f"  {'HOOK SCORE':<13} {bar(overall)} {overall:5.1f}   {verdict}", file=out)
    for f in flags:
        print(f"\n  DEALBREAKER  {f}", file=out)
    if verdict != "STRONG":
        weakest = min(CHECKS, key=lambda c: results[c][0])
        print(f"\n  Weakest property: {weakest}. Fix that one and re-run.", file=out)
    print("", file=out)


def render_table(rows, out=sys.stdout):
    print("\nHOOK RANKING\n" + "=" * 78, file=out)
    for i, r in enumerate(rows, 1):
        mark = "->" if i == 1 else "  "
        hook = r["hook"] if len(r["hook"]) <= 62 else r["hook"][:59] + "..."
        print(f"{mark} {r['score']:5.1f} {r['verdict']:<7} {hook}", file=out)
        print(f"        weakest: {r['weakest']} ({r['checks'][r['weakest']]['score']:.0f})", file=out)
        for f in r["flags"]:
            print(f"        dealbreaker: {f}", file=out)
    print("\nShoot the top one. If the top one is under 50, none of these are the hook.\n",
          file=out)


def main():
    ap = argparse.ArgumentParser(description="Score a Reel hook on five properties.")
    ap.add_argument("input", nargs="?", default="-", help="file with one hook per line, or -")
    ap.add_argument("--hook", help="score a single hook given on the command line")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lang", choices=["auto", "en", "es"], default="auto",
                    help="language of the hooks (default: detect)")
    args = ap.parse_args()

    if args.hook:
        lines = [args.hook]
    else:
        raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        print("nothing to score", file=sys.stderr)
        sys.exit(2)

    lang = detect_lang(" ".join(lines)) if args.lang == "auto" else args.lang

    payload = []
    for line in lines:
        results, overall, verdict, flags = run(line, lang)
        payload.append({
            "hook": line,
            "checks": {k: {"score": round(v[0], 1), "detail": v[1]} for k, v in results.items()},
            "weakest": min(CHECKS, key=lambda c: results[c][0]),
            "flags": flags,
            "score": round(overall, 1),
            "verdict": verdict,
        })

    if args.json:
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2, ensure_ascii=False))
        return

    if len(payload) == 1:
        results, overall, verdict, flags = run(lines[0], lang)
        render_one(lines[0], results, overall, verdict, flags)
    else:
        render_table(sorted(payload, key=lambda r: -r["score"]))

    sys.exit(0 if max(p["score"] for p in payload) >= 70 else 1)


if __name__ == "__main__":
    main()
