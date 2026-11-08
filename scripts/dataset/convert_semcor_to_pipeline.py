"""Write the 1,000 words of the pool as per-word files and configs.

    python scripts/dataset/convert_semcor_to_pipeline.py --pool data/semcor_pool.json \
        --n-words 1000 --out-dir data/raw/semcor_words"""
import argparse, collections, json, os, re

PREFIX = "Example sentence: "
DESIGN = ("SemCor 3.0 sentences with WordNet sense annotations; one surface form "
          "per word, the two most frequent senses, balanced classes")


def detokenise(tokens):
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$])\s+", r"\1", s)
    s = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s[0].upper() + s[1:] if s else s


def pick_form(word):
    """Best single surface form: the one whose weaker sense has most sentences."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in word["sentences"]:
        by[s["target"].lower()][s["sense"]].append(s)
    best = None
    for form, senses in by.items():
        if len(senses) < 2:
            continue
        two = sorted(senses.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
        score = min(len(two[0][1]), len(two[1][1]))
        if best is None or score > best[0]:
            best = (score, form, two)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/semcor_pool.json")
    ap.add_argument("--n-words", type=int, default=1000)
    ap.add_argument("--min-per-sense", type=int, default=4)
    ap.add_argument("--max-per-sense", type=int, default=20)
    ap.add_argument("--out-dir", default="data/raw/semcor_words")
    a = ap.parse_args()

    pool = json.load(open(a.pool))["words"]
    cand = []
    for w in pool:
        best = pick_form(w)
        if best and best[0] >= a.min_per_sense:
            cand.append((best[0], w["word"], best[1], best[2]))
    cand.sort(key=lambda t: (-t[0], t[1]))
    EXCLUDE = {'bank','bat','crane','seal','plant','pupil','club'}
    seen, ded = set(), []
    for c in cand:
        if c[2] in EXCLUDE or c[2] in seen:
            continue
        seen.add(c[2]); ded.append(c)
    cand = ded[:a.n_words]

    os.makedirs(a.out_dir, exist_ok=True)
    manifest, n_sent = [], 0
    for score, lemma, form, two in cand:
        n = min(score, a.max_per_sense)   # balanced classes
        sentences, labels, rel_pos = [], [], []
        for syn, items in two:
            for s in items[:n]:
                sentences.append(PREFIX + detokenise(s["text"].split()))
                labels.append(syn)
                rel_pos.append(s["target_index"] / max(1, len(s["text"].split())))
        data = {"target_word": form,
                "sentences": sentences,
                "labels": labels,
                "stratum": ["semcor"] * len(sentences),
                "prefix": PREFIX,
                "design": DESIGN}
        with open(os.path.join(a.out_dir, f"{form}.json"), "w") as f:
            json.dump(data, f, indent=1)
        n_sent += len(sentences)
        manifest.append({"word": form, "lemma": lemma,
                         "senses": [two[0][0], two[1][0]],
                         "n_per_sense": n,
                         "mean_target_rel_pos": sum(rel_pos) / len(rel_pos)})

    with open(os.path.join(a.out_dir, "_manifest.json"), "w") as f:
        json.dump({"n_words": len(manifest), "n_sentences": n_sent,
                   "min_per_sense": a.min_per_sense, "words": manifest}, f, indent=1)

    per = [m["n_per_sense"] for m in manifest]
    print(f"words: {len(manifest)}  sentences: {n_sent}")
    print(f"sentences per sense: min {min(per)}, median {sorted(per)[len(per)//2]}, max {max(per)}")
    print(f"pairs per word (same/diff): about {2*min(per)*(min(per)-1)//2}--{2*max(per)*(max(per)-1)//2} / "
          f"{min(per)**2}--{max(per)**2}")
    print("manifest:", os.path.join(a.out_dir, "_manifest.json"))


if __name__ == "__main__":
    main()
