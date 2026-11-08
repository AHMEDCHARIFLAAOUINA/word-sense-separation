"""Batches per word until every sentence of the word has been used."""
import random


def build_cover(by_word, B=16, seed=0):
    rng = random.Random(seed)
    words = sorted(w for w, s in by_word.items()
                   if sum(len(v) >= 2 for v in s.values()) >= 2)
    rng.shuffle(words)
    batches = []
    for w in words:
        senses = {k: list(v) for k, v in by_word[w].items() if len(v) >= 2}
        for v in senses.values():
            rng.shuffle(v)
        remaining = {k: list(v) for k, v in senses.items()}
        while any(remaining.values()):
            other = max(remaining, key=lambda k: len(remaining[k]))
            anchor_senses = [k for k in senses if k != other]
            unused = [k for k in anchor_senses if remaining[k]]
            if unused:
                anchor = remaining[unused[0]].pop()
            else:
                anchor = rng.choice(senses[rng.choice(anchor_senses)])
            rest, remaining[other] = remaining[other][:B - 1], remaining[other][B - 1:]
            batches.append([anchor] + rest)
    return batches


def attach_cover(sep_data, train_rng, B, seed):
    batches = build_cover(sep_data.by_word, B, seed)
    covered = {id(x) for b in batches for x in b}
    total = sum(len(v) for s in sep_data.by_word.values() for v in s.values() if len(v) >= 2)
    state = {"i": 0}
    original = sep_data.sample

    def sample(n=None, rng=None):
        if rng is train_rng:
            b = batches[state["i"] % len(batches)]
            state["i"] += 1
            return b
        return original(n, rng) if rng is not None else original(n)

    sep_data.sample = sample
    sep_data.cover = batches
    print(f"[cover] {len(batches)} batches (= training steps) over "
          f"{len({b[0]['word'] for b in batches})} words; {len(covered)}/{total} "
          f"sentences covered; B <= {B}")
    return len(batches)
