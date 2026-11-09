"""Write our sentences and sense labels in the input format of the OpenHowNet project code.

    cd <OpenHowNet project folder>; python scripts/make_ta_input.py"""
import os
import json
import re
from pathlib import Path

PIPE = Path(os.environ.get("WSS_PIPELINE", "."))
OUT = Path(__file__).resolve().parents[1] / "data" / "ta_input"
CONTROLLED = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]


def controlled_file(word):
    """Dataset file named in configs/words/<word>.yaml (same as the Part 1 pipeline)."""
    for line in (PIPE / "configs/words" / f"{word}.yaml").read_text().splitlines():
        if line.startswith("dataset:"):
            return PIPE / line.split(":", 1)[1].strip().strip('"')
    raise SystemExit(f"no dataset line in configs/words/{word}.yaml")


def write(name, files):
    OUT.mkdir(parents=True, exist_ok=True)
    n_rec = n_skip = 0
    words = 0
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for path in files:
            d = json.load(open(path, encoding="utf-8"))
            w = d["target_word"]
            words += 1
            pat = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
            for i, (s, lab) in enumerate(zip(d["sentences"], d["labels"])):
                m = pat.search(s)
                if m is None:
                    n_skip += 1
                    continue
                ann = {"occurrence_id": f"{w}:{i}", "record_index": n_rec, "word": s[m.start():m.end()],
                       "dictionary_word": w.lower(), "sentence": s, "start_char": m.start(), "end_char": m.end(),
                       "primary_sememe": str(lab), "selected_sememes": [str(lab)], "selected_sense_id": str(lab),
                       "confidence": 1.0, "rationale": "gold label from SemCor",
                       "annotation_status": "ok"}
                f.write(json.dumps({"id": f"{w}:{i}", "openhownet_annotations": [ann]}, ensure_ascii=False) + "\n")
                n_rec += 1
    print(f"{name}: {words} words, {n_rec} sentences written, {n_skip} skipped (target not found) -> {OUT / (name + '.jsonl')}")


write("controlled7", [controlled_file(w) for w in CONTROLLED])
write("semcor", sorted(p for p in (PIPE / "data/raw/semcor_words").glob("*.json") if not p.name.startswith("_")))
