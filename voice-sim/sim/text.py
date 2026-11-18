"""Text helpers: how numbers are SPOKEN (for the synthetic user) and how speech recognition is SCORED."""
import re

import jiwer

_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_WORDS_BY_DIGIT = {v: k for k, v in _DIGIT_WORDS.items() if k != "oh"}


def spoken(text: str) -> str:
    """Rewrites digit runs as spoken digits ("5550100" -> "five five five zero one zero zero") so the TTS voice reads a phone
    number the way a person does, not as "five million five hundred ...". Dashes and spaces inside the number are kept out."""
    def digits(m: re.Match) -> str:
        return " ".join(_WORDS_BY_DIGIT[d] for d in re.sub(r"\D", "", m.group(0)))

    return re.sub(r"\+?\d[\d\-\s().]{2,}\d|\d{2,}", digits, text)


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, spoken digits -> digits, merge adjacent digit tokens: "five five five" == "555" == "5 5 5"."""
    tokens = [_DIGIT_WORDS.get(t, t) for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()]
    out: list[str] = []
    for t in tokens:
        if out and t.isdigit() and out[-1].isdigit():
            out[-1] += t
        else:
            out.append(t)
    return out


def wer(said: str, heard: str) -> dict:
    """Word error of what the agent HEARD (its own transcript of the user) against what the user SAID."""
    ref, hyp = normalize(said), normalize(heard)
    if not ref:
        return {"wer": None, "errors": 0, "words": 0}
    if not hyp:
        return {"wer": 1.0, "errors": len(ref), "words": len(ref)}
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    errors = o.substitutions + o.deletions + o.insertions
    return {"wer": errors / len(ref), "errors": errors, "words": len(ref)}
