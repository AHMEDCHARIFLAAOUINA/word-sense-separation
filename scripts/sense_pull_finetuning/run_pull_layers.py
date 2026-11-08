"""All 5 models with the pull on the last 8 and last 10 layers, 3 seeds. Skips finished jobs.

    python scripts/sense_pull_finetuning/run_pull_layers.py"""
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_LAYERS = {"llama-3-8b": 32, "llama-3.1-8b": 32, "qwen1.5-7b": 32, "qwen2-7b": 28, "qwen2.5-7b": 28}
LAST = (8, 10)
SEEDS = (0, 1, 2)


def layers(m, k):
    n = N_LAYERS[m]
    return f"{n - k + 1}-{n}"


def train_job(m, k, s):
    tag = f"{m}_ta_pull_last{k}"
    cmd = ["python", "-m", "src.finetune.ta_semantic_regulariser", "--model", m, "--task", "gsm8k_full",
           "--seed", str(s), "--layers", layers(m, k), "--loss", "ta_pull", "--span", "mean", "--tag", tag,
           "--save-adapter", f"adapters/final_gsm8k_full_{tag}_s{s}"]
    return f"final_gsm8k_full_{tag}_s{s}", cmd, "saved ->"


def eval_job(m, k, s):
    tag = f"{m}_ta_pull_last{k}"
    cmd = ["python", "scripts/sense_pull_finetuning/eval_final.py", "--eval", "gsm8k_full", "--model", m,
           "--adapter", f"adapters/final_gsm8k_full_{tag}_s{s}", "--seed", str(s), "--out-dir", "final_eval_v2"]
    return f"v2eval_gsm8k_full__final_gsm8k_full_{tag}_s{s}", cmd, "stop-rule accuracy"


def log(name):
    return ROOT / "logs" / f"{name}.log"


def done(name, marker):
    p = log(name)
    return p.exists() and marker in p.read_text(errors="ignore")


def free_gpus(busy):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(i) for i, mem in (l.split(",") for l in out.strip().splitlines())
            if int(mem) < 2000 and int(i) not in busy]


def run_queue(jobs, title):
    todo = [j for j in jobs if not done(j[0], j[2])]
    print(f"{title}: {len(jobs)} jobs, {len(todo)} to do", flush=True)
    running, failed = {}, 0
    while todo or running:
        for g, (j, p) in list(running.items()):
            if p.poll() is not None:
                ok = done(j[0], j[2])
                failed += not ok
                print(f"{time.strftime('%H:%M')} {'done' if ok else 'FAILED'} {j[0]}", flush=True)
                del running[g]
        for g in free_gpus(running)[:max(0, 7 - len(running))]:
            if not todo:
                break
            j = todo.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
            running[g] = (j, subprocess.Popen(j[1], cwd=ROOT, env=env, stdout=open(log(j[0]), "w"),
                                              stderr=subprocess.STDOUT))
            print(f"{time.strftime('%H:%M')} start {j[0]} on GPU {g}", flush=True)
            time.sleep(20)
        time.sleep(30)
    return failed


JOBS = [(m, k, s) for m in N_LAYERS for k in LAST for s in SEEDS]
f1 = run_queue([train_job(*j) for j in JOBS], "PHASE 1 training")
f2 = run_queue([eval_job(*j) for j in JOBS], "PHASE 2 official-test evaluation")
print(f"all done (FAILED: training {f1}, evaluation {f2})", flush=True)
