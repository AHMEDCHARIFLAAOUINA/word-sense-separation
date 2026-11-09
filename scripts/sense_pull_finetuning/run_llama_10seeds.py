"""llama-3-8b, last 4 layers, seeds 3-9.

    python scripts/sense_pull_finetuning/run_llama_10seeds.py"""
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"llama-3-8b": "29-32"}   # last 4 layers
SEEDS = tuple(range(10))   # seeds 0-2 already done (skipped), 3-9 new
VARIANTS = {"baseline": ["--baseline"], "ta_pull": ["--loss", "ta_pull", "--span", "mean"]}


def train_job(m, v, s):
    tag = f"{m}_{v}"
    cmd = ["python", "-m", "src.finetune.sense_pull_regulariser", "--model", m, "--task", "gsm8k_full",
           "--seed", str(s), "--layers", MODELS[m], "--tag", tag,
           "--save-adapter", f"adapters/final_gsm8k_full_{tag}_s{s}"] + VARIANTS[v]
    return f"final_gsm8k_full_{tag}_s{s}", cmd, "saved ->"


def eval_job(m, v, s):
    adapter = "none" if v == "pretrained" else f"adapters/final_gsm8k_full_{m}_{v}_s{s}"
    pre = "pretrained" if m == "llama-3-8b" else f"pretrained_{m}"   # eval_final's naming for the pretrained model
    name = f"gsm8k_full__{pre}" if v == "pretrained" else f"gsm8k_full__final_gsm8k_full_{m}_{v}_s{s}"
    cmd = ["python", "scripts/sense_pull_finetuning/eval_final.py", "--eval", "gsm8k_full", "--model", m,
           "--adapter", adapter, "--seed", str(s), "--out-dir", "final_eval_v2"]
    return f"v2eval_{name}", cmd, "stop-rule accuracy"


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


f1 = run_queue([train_job(m, v, s) for m in MODELS for v in VARIANTS for s in SEEDS], "PHASE 1 training")
evals = [eval_job(m, "pretrained", 0) for m in MODELS]
evals += [eval_job(m, v, s) for m in MODELS for v in VARIANTS for s in SEEDS]
f2 = run_queue(evals, "PHASE 2 official-test evaluation")
print(f"all done (FAILED: training {f1}, evaluation {f2})", flush=True)
