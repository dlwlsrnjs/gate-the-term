"""Deterministic language-consistency check by script.

The target component of the multilingual setting is "did the model answer in the
language it was asked in". Measuring that with a language-ID classifier would put a
noisy learned component at the centre of the paper, and every number would inherit
its error rate. We avoid that entirely by restricting to languages written in a
script that no other language in the set uses, so consistency is a deterministic
function of Unicode code points:

    lang(o) = 1[ share of o's letters that lie in the target script >= THRESH ]

There is nothing to train, nothing to tune, and the measurement is reproducible to
the character. The cost is that the study covers non-Latin-script languages only,
which we state as a limitation rather than papering over with a classifier.
"""

# Unicode letter ranges that uniquely identify each script in our language set.
SCRIPTS = {
    "ru": [(0x0400, 0x04FF), (0x0500, 0x052F)],            # Cyrillic
    "bn": [(0x0980, 0x09FF)],                              # Bengali
    "te": [(0x0C00, 0x0C7F)],                              # Telugu
    "th": [(0x0E00, 0x0E7F)],                              # Thai
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF),             # Hiragana, Katakana
           (0x4E00, 0x9FFF)],                              # + Han
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],            # Han
}
LATIN = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]
THRESH = 0.5


def _in(cp, ranges):
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_share(text, lang):
    """fraction of script-bearing characters that belong to the target script.

    Digits, punctuation and whitespace are ignored: a numeric answer must not count
    as evidence for either language. If a response contains no script-bearing
    character at all (a bare number), the share is undefined and we return None so
    callers can decide, rather than silently scoring it as consistent."""
    ranges = SCRIPTS[lang]
    tgt = other = 0
    for ch in text:
        cp = ord(ch)
        if _in(cp, ranges):
            tgt += 1
        elif _in(cp, LATIN) or ch.isalpha():
            other += 1
    total = tgt + other
    return (tgt / total) if total else None


def is_consistent(text, lang, thresh=THRESH):
    """1.0 if the response is in the target language, 0.0 otherwise.

    A response with no letters at all (just a number) is scored 0: it does not
    answer in the target language, and treating it as consistent would reward the
    degenerate strategy of emitting a bare digit."""
    s = script_share(text, lang)
    return 0.0 if s is None else float(s >= thresh)


if __name__ == "__main__":
    tests = [
        ("Джанет продаёт 9 яиц по 2 доллара, итого 18 долларов.", "ru", 1.0),
        ("Janet sells 9 eggs at $2 each, so 18 dollars.", "ru", 0.0),
        ("জেনেট প্রতিদিন 9টি ডিম বিক্রি করেন, মোট 18 ডলার।", "bn", 1.0),
        ("Janet sells 9 eggs. The answer is 18.", "bn", 0.0),
        ("18", "ru", 0.0),
        ("คำตอบคือ 18 บาท", "th", 1.0),
    ]
    ok = True
    for text, lang, want in tests:
        got = is_consistent(text, lang)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        sh = script_share(text, lang)
        print(f"  {flag} lang={lang} want={want} got={got} "
              f"share={'None' if sh is None else f'{sh:.3f}'}  {text[:44]}")
    print("ALL PASS" if ok else "SOME FAILED")
