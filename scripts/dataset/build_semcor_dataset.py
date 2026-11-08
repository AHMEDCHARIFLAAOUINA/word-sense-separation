"""Build the pool of ambiguous words from SemCor 3.0 (two most frequent WordNet senses).

    python scripts/dataset/build_semcor_dataset.py --pos nvar --min-per-sense 4 \
        --max-per-sense 20 --n-words 3000 --out data/semcor_pool.json"""
import argparse, collections, json, os, random

import nltk
nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nltk_data"))
nltk.data.path.insert(0, "data/nltk_data")
from nltk.corpus import semcor
from nltk.tree import Tree

# auxiliaries and light verbs: their WordNet senses are not lexical ambiguity
STOP_LEMMAS = {
    "be", "have", "do", "not", "will", "would", "can", "could", "shall", "should",
    "may", "might", "must", "get", "go", "come", "take", "give", "put", "let",
    "person", "group", "location", "time_period",
}


def collect(pos_keep, max_len=60):
    """occurrences[lemma][synset] = list of (sent_id, token_index, tokens)"""
    occ = collections.defaultdict(lambda: collections.defaultdict(list))
    n_sent = 0
    for sid, sent in enumerate(semcor.tagged_sents(tag="sem")):
        n_sent += 1
        tokens, spans = [], []
        for ch in sent:
            leaves = ch.leaves() if isinstance(ch, Tree) else ch
            start = len(tokens)
            tokens.extend(leaves)
            if isinstance(ch, Tree):
                spans.append((ch.label(), start, len(leaves)))
        if len(tokens) > max_len:
            continue
        for lab, start, length in spans:
            if length != 1 or not hasattr(lab, "synset"):
                continue   # multiword or unparsed label
            syn = lab.synset()
            lemma = lab.name().lower()
            if syn.pos() not in pos_keep or lemma in STOP_LEMMAS or "_" in lemma:
                continue
            if start < 1:
                continue   # target must not be sentence-initial
            occ[lemma][syn.name()].append((sid, start, tokens))
    return occ, n_sent


def build(occ, min_per_sense, n_words, max_per_sense, seed=0):
    rng = random.Random(seed)
    cand = []
    for lemma, senses in occ.items():
        ok = [(s, o) for s, o in senses.items()
              if len({x[0] for x in o}) >= min_per_sense]
        if len(ok) < 2:
            continue
        ok.sort(key=lambda kv: len({x[0] for x in kv[1]}), reverse=True)
        (s1, o1), (s2, o2) = ok[0], ok[1]   # the two most frequent senses
        cand.append((min(len(o1), len(o2)), lemma, s1, o1, s2, o2))
    cand.sort(reverse=True)

    words = []
    for _, lemma, s1, o1, s2, o2 in cand[:n_words]:
        entry = {"word": lemma, "senses": [s1, s2], "sentences": []}
        for sense, occs in ((s1, o1), (s2, o2)):
            seen, kept = set(), []
            for sid, idx, tokens in occs:
                if sid in seen:
                    continue   # one occurrence per sentence
                seen.add(sid)
                kept.append({"sense": sense, "target_index": idx,
                             "target": tokens[idx], "text": " ".join(tokens)})
            rng.shuffle(kept)
            entry["sentences"].extend(kept[:max_per_sense])
        words.append(entry)
    return words


def report(words):
    per_word = [len(w["sentences"]) for w in words]
    lens = [len(s["text"].split()) for w in words for s in w["sentences"]]
    pairs_same = pairs_diff = 0
    for w in words:
        c = collections.Counter(s["sense"] for s in w["sentences"])
        n = list(c.values())
        pairs_same += sum(k * (k - 1) // 2 for k in n)
        pairs_diff += n[0] * n[1] if len(n) > 1 else 0
    def q(v, p):
        v = sorted(v); return v[int(p * (len(v) - 1))]
    print(f"words: {len(words)}")
    print(f"sentences: {sum(per_word)}")
    print(f"sentences per word: min {min(per_word)}, median {q(per_word,.5)}, max {max(per_word)}")
    print(f"sentence length (words): median {q(lens,.5)}, 10th {q(lens,.1)}, 90th {q(lens,.9)}")
    print(f"pairs: {pairs_same} same-sense, {pairs_diff} different-sense")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-words", type=int, default=1000)
    ap.add_argument("--min-per-sense", type=int, default=5)
    ap.add_argument("--max-per-sense", type=int, default=50)
    ap.add_argument("--pos", default="nv", help="n = nouns, v = verbs, nv = both")
    ap.add_argument("--out", default="data/semcor_1000.json")
    ap.add_argument("--max-sent-words", type=int, default=60)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()

    occ, n_sent = collect(set(a.pos), a.max_sent_words)
    print(f"SemCor: {n_sent} sentences, {len(occ)} lemmas after filtering")
    for k in (2, 3, 5, 10):
        n = sum(1 for _, d in occ.items()
                if sum(1 for o in d.values() if len({x[0] for x in o}) >= k) >= 2)
        print(f"  words with 2+ senses having >={k} sentences each: {n}")

    words = build(occ, a.min_per_sense, a.n_words, a.max_per_sense)
    report(words)
    if not a.stats_only:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"source": "SemCor 3.0 (WordNet senses)",
                       "min_per_sense": a.min_per_sense,
                       "max_per_sense": a.max_per_sense,
                       "pos": a.pos, "words": words}, f, indent=1)
        print("written:", a.out)
