"""Run the project's extract_embeddings + analyze_similarity (unchanged) for 12 models.

    cd ta_sememe_project; python our_scripts/run_ta_task1.py semcor"""
import os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = os.environ.get("WSS_MODELS", "models")
MODELS = {
    "llama-7b": str(ROOT / "model_links/llama-7b"), "llama-2-7b": f"{M}/Llama/Llama-2-7b",
    "llama-3-8b": f"{M}/Llama/Llama-3-8b", "llama-3.1-8b": f"{M}/Llama/Llama-3.1-8B",
    "qwen1.5-7b": f"{M}/Qwen/qwen1.5-7b", "qwen2-7b": f"{M}/Qwen/qwen2-7b",
    "qwen2.5-7b": f"{M}/Qwen/qwen2.5-7b", "qwen3-8b": f"{M}/Qwen/qwen3-8b",
    "bert-base": f"{M}/Bert/bert-base", "roberta-base": f"{M}/Bert/roberta-base",
    "spanbert-base-cased": f"{M}/Bert/spanbert-base-cased", "xlm-roberta-base": f"{M}/Bert/xlm-roberta-base",
}
DATASETS = ["controlled7", "semcor"] if len(sys.argv) < 2 or sys.argv[1] == "all" else [sys.argv[1]]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def free_gpu():
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = [int(i) for i, m in (l.split(",") for l in out.strip().splitlines()) if int(m) < 2000]
        if free:
            return free[-1]
        time.sleep(60)


def run(cmd, log, gpu):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                              stdout=f, stderr=subprocess.STDOUT).returncode


failed = []
for ds in DATASETS:
    for name, path in MODELS.items():
        emb = ROOT / "outputs" / ds / "embeddings" / name
        ana = ROOT / "outputs" / ds / "analysis" / name
        if (ana / "mean_cosine_by_layer.csv").exists():
            print(f"{ds} {name}: done, skipped", flush=True)
            continue
        log = ROOT / "outputs" / ds / f"log_{name}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        if emb.exists():
            shutil.rmtree(emb)   # unfinished earlier attempt
        g = free_gpu()
        print(f"{time.strftime('%H:%M')} {ds} {name}: extract on GPU {g}", flush=True)
        rc = run(["python", "-m", "sememe_project.extract_embeddings", "--input", f"data/ta_input/{ds}.jsonl",
                  "--output-dir", str(emb), "--models", path, "--batch-size", "4"], log, g)
        if rc == 0:
            rc = run(["python", "-m", "sememe_project.analyze_similarity", "--embeddings-dir", str(emb),
                      "--output-dir", str(ana)], log, g)
        if rc != 0:
            failed.append(f"{ds} {name}")
            print(f"{time.strftime('%H:%M')} {ds} {name}: FAILED (see {log})", flush=True)
            continue
        for d in emb.iterdir():   # free the disk, keep model.json
            for s in d.glob("shard_*"):
                s.unlink()
        print(f"{time.strftime('%H:%M')} {ds} {name}: done", flush=True)
print(f"all done (FAILED: {len(failed)}) {failed}", flush=True)
