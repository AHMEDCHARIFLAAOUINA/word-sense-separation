"""Other-benchmark evaluation of all GSM8K models (pretrained, baseline, pull on 4 / 8 / 10 layers).

    python scripts/sense_pull_finetuning/run_broad_full.py"""
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("llama-3-8b", "llama-3.1-8b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b")
VARIANTS = ("baseline", "ta_pull", "ta_pull_last8", "ta_pull_last10")
SEEDS = (0, 1, 2)

JOBS = []
for m in MODELS:
    name = "pretrained" if m == "llama-3-8b" else f"pretrained_{m}"
    JOBS.append((f"broadfull_{name}", m, "none"))
    for v in VARIANTS:
        for s in SEEDS:
            a = f"adapters/final_gsm8k_full_{m}_{v}_s{s}"
            JOBS.append((f"broadfull_{Path(a).name}", m, a))


def log(name):
    return ROOT / "logs" / f"{name}.log"


def done(name):
    p = log(name)
    return p.exists() and "ALL DONE" in p.read_text(errors="ignore")


def ready(adapter):
    return adapter == "none" or (ROOT / adapter / "adapter_config.json").exists()


def free_gpus(busy):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(i) for i, mem in (l.split(",") for l in out.strip().splitlines())
            if int(mem) < 2000 and int(i) not in busy]


todo = [j for j in JOBS if not done(j[0]) and ready(j[2])]
missing = [j[0] for j in JOBS if not ready(j[2])]
print(f"{len(JOBS)} models, {len(todo)} to do, {len(missing)} adapters not trained yet (skipped)", flush=True)
running, failed = {}, 0
while todo or running:
    for g, (j, p) in list(running.items()):
        if p.poll() is not None:
            ok = done(j[0])
            failed += not ok
            print(f"{time.strftime('%H:%M')} {'done' if ok else 'FAILED'} {j[0]}", flush=True)
            del running[g]
    for g in free_gpus(running)[:max(0, 7 - len(running))]:
        if not todo:
            break
        j = todo.pop(0)
        cmd = ["python", "scripts/sense_pull_finetuning/eval_broad.py", "--model", j[1], "--adapter", j[2]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        running[g] = (j, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log(j[0]), "w"),
                                          stderr=subprocess.STDOUT))
        print(f"{time.strftime('%H:%M')} start {j[0]} on GPU {g}", flush=True)
        time.sleep(20)
    time.sleep(30)
print(f"all done (FAILED: {failed}; not trained yet: {len(missing)})", flush=True)
