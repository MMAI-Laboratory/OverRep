import itertools
import logging
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


class Style:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'


class CustomLogFormatter(logging.Formatter):
    def format(self, record):
        timestamp = self.formatTime(record, "%H:%M:%S")
        icon = getattr(record, 'icon', "•")
        color = getattr(record, 'color', Style.RESET)
        return f"{Style.BOLD}[{timestamp}]{Style.RESET} {color}{icon} {record.getMessage()}{Style.RESET}"


def setup_logger():
    logger = logging.getLogger("GPUScheduler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(CustomLogFormatter())
        logger.addHandler(ch)
    return logger


LOGGER = setup_logger()

CONFIG = {
    "2-7b": "meta-llama/Llama-2-7b-hf",
    "2-13b": "meta-llama/Llama-2-13b-hf",
    "3-3b": "meta-llama/Llama-3.2-3B",
    "3-8b": "meta-llama/Llama-3.1-8B",
    "q3-4b": "Qwen/Qwen3-4B-Base",
    "q3-8b": "Qwen/Qwen3-8B-Base",
    "q3-14b": "Qwen/Qwen3-14B-Base",
    "q3moe-30b": "Qwen/Qwen3-30B-A3B-Base",
}

MODEL_KEY = {
    "2-7b": "llama2_7B",
    "2-13b": "llama2_13B",
    "3-3b": "llama3_3B",
    "3-8b": "llama3_8B",
    "q3-4b": "qwen3_4B",
    "q3-8b": "qwen3_8B",
    "q3-14b": "qwen3_14B",
    "q3moe-30b": "qwen3moe_30B_A3B",
}

# 25% pruning, target_layer:layer_interval
INCLUDE_LAYER_25 = {
    "2-7b": "22:8", "2-13b": "28:10",
    "3-3b": "17:9", "3-8b": "21:9",
    "q3-4b": "23:11", "q3-8b": "24:10",
    "q3-14b": "18:10", "q3moe-30b": "34:12",
}

# 50% pruning
INCLUDE_LAYER_50 = {
    "2-7b": "14:16", "2-13b": "18:20",
    "3-3b": "10:16", "3-8b": "12:18",
    "q3-4b": "13:21", "q3-8b": "14:20",
    "q3-14b": "18:20", "q3moe-30b": "22:24",
}

MIXED_PRECISION = {
    "2-7b": "no", "2-13b": "no",
    "3-3b": "bf16", "3-8b": "bf16",
    "q3-4b": "bf16", "q3-8b": "bf16",
    "q3-14b": "bf16", "q3moe-30b": "bf16",
}


def backbone_arch(backbone: str) -> str:
    return 'qwen' if backbone.startswith('q') else 'llama'


def dataset_alias(backbone: str) -> str:
    return f"{backbone_arch(backbone)}{backbone.replace('q', '')}"


def model_name_for(backbone: str, preset: str) -> str:
    return f"{MODEL_KEY[backbone]}_{preset}"


def include_layer_for(backbone: str, pruning_ratio: str) -> str:
    return INCLUDE_LAYER_25[backbone] if pruning_ratio == "Qtr" else INCLUDE_LAYER_50[backbone]


def dict_product(grid: Dict[str, List]):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


@dataclass(frozen=True)
class Job:
    script: str
    args: List[str]
    name: str
    num_gpus: int
    stage: str
    config_name: str = None
    include_layer: str = None


def parse_gpu_list(gpus: str) -> List[int]:
    return sorted(int(x.strip()) for x in gpus.split(",") if x.strip())


def _acquire_gpus(free_gpus, condition, n: int) -> List[int]:
    with condition:
        while len(free_gpus) < n:
            condition.wait()
        return [free_gpus.pop(0) for _ in range(n)]


def _release_gpus(free_gpus, condition, gpu_ids: List[int]) -> None:
    with condition:
        free_gpus.extend(gpu_ids)
        free_gpus[:] = sorted(free_gpus)
        condition.notify_all()


_CONSOLE_TO_MODULE = {
    'overrep-train': 'overrep.train.mse',
    'overrep-train-kd': 'overrep.train.kd',
    'overrep-eval': 'overrep.eval.harness',
    'overrep-score': 'overrep.eval.score',
    'overrep-cache': 'overrep.features.dump_hidden_for_mse',
    'overrep-cache-kd': 'overrep.features.dump_hidden_for_kd',
}


def _resolve_command(script: str) -> str:
    import shutil
    import sys as _sys
    from pathlib import Path as _Path
    venv_exe = _Path(_sys.executable).parent / script
    if venv_exe.exists():
        return str(venv_exe)
    on_path = shutil.which(script)
    if on_path:
        return on_path
    module = _CONSOLE_TO_MODULE.get(script)
    if module:
        return f"{_sys.executable} -m {module}"
    return script


def _run_job(job: Job, gpu_ids, free_gpus, condition, task_id, total_tasks, failed, failed_logs):
    visible_devices = ",".join(map(str, gpu_ids))
    cmd = ' '.join([_resolve_command(job.script), *job.args])
    full_cmd = f"CUDA_VISIBLE_DEVICES={visible_devices} {cmd}"
    captured = deque(maxlen=500)
    try:
        LOGGER.info(
            f"[{job.stage.upper()}] Starting Task #{task_id}/{total_tasks} on GPUs [{visible_devices}] | {job.name}",
            extra={"icon": "🚀", "color": Style.CYAN},
        )
        LOGGER.info(
            f"Command: {full_cmd}",
            extra={"icon": "  ↳", "color": Style.BLUE},
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

        proc = subprocess.Popen(
            shlex.split(cmd), env=env,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        for line in iter(proc.stderr.readline, ''):
            sys.stderr.write(line)
            sys.stderr.flush()
            captured.append(line)
        proc.stderr.close()
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        LOGGER.info(
            f"[{job.stage.upper()}] Finished Task #{task_id} on GPUs [{visible_devices}] | {job.name}",
            extra={"icon": "✅", "color": Style.GREEN},
        )
    except subprocess.CalledProcessError as e:
        LOGGER.info(
            f"[{job.stage.upper()}] Failed Task #{task_id} on GPUs [{visible_devices}] | {job.name} | {e}",
            extra={"icon": "❌", "color": Style.RED},
        )
        failed.append(cmd)
        log_entry = (
            f"{'=' * 80}\n"
            f"JOB:   {job.name}\n"
            f"STAGE: {job.stage}\n"
            f"CMD:   {full_cmd}\n"
            f"EXIT:  {e.returncode}\n"
            f"{'-' * 80}\n"
            f"{''.join(captured)}"
            f"{'=' * 80}\n\n"
        )
        failed_logs.append(log_entry)
    finally:
        _release_gpus(free_gpus, condition, gpu_ids)


def schedule_jobs(jobs: Sequence[Job], title: str, free_gpus, condition, failed, failed_logs):
    LOGGER.info("=" * 60)
    LOGGER.info(f"{Style.BOLD}      {title}{Style.RESET}")
    LOGGER.info(f"{Style.BOLD}      Tasks: {len(jobs)} | Total GPUs: {len(free_gpus)}{Style.RESET}")
    LOGGER.info("=" * 60 + "\n")

    processes = []
    for i, job in enumerate(jobs, start=1):
        gpu_ids = _acquire_gpus(free_gpus, condition, job.num_gpus)
        p = mp.Process(target=_run_job, args=(job, gpu_ids, free_gpus, condition, i, len(jobs), failed, failed_logs))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    LOGGER.info("All tasks scheduled. Waiting for completion...", extra={"icon": "💤", "color": Style.MAGENTA})
    for p in processes:
        p.join()


def _run_evals_for_train(train_job: Job, eval_jobs: Sequence[Job], free_gpus, condition, failed, failed_logs):
    train_failed = any(
        train_job.script in cmd and train_job.name in cmd
        for cmd in list(failed)
    )
    if train_failed:
        LOGGER.info(
            f"[EVAL] Skipping {len(eval_jobs)} evals — train failed | {train_job.name}",
            extra={"icon": "⏭️", "color": Style.YELLOW},
        )
        return

    procs = []
    for j, eval_job in enumerate(eval_jobs, start=1):
        gpu_ids = _acquire_gpus(free_gpus, condition, eval_job.num_gpus)
        p = mp.Process(
            target=_run_job,
            args=(eval_job, gpu_ids, free_gpus, condition, j, len(eval_jobs), failed, failed_logs),
        )
        p.start()
        procs.append(p)
        time.sleep(0.1)
    for p in procs:
        p.join()


def _run_train_supervisor(train_job: Job, eval_jobs: Sequence[Job], gpu_ids,
                          free_gpus, condition, task_id, total_tasks, failed, failed_logs):
    _run_job(train_job, gpu_ids, free_gpus, condition, task_id, total_tasks, failed, failed_logs)
    if eval_jobs:
        _run_evals_for_train(train_job, eval_jobs, free_gpus, condition, failed, failed_logs)


def schedule_interleaved(train_jobs: Sequence[Job], eval_jobs_by_train: Dict[str, Sequence[Job]],
                         free_gpus, condition, failed, failed_logs):
    LOGGER.info("=" * 60)
    LOGGER.info(f"{Style.BOLD}      INTERLEAVED TRAIN→EVAL SCHEDULER STARTING{Style.RESET}")
    n_evals = sum(len(v) for v in eval_jobs_by_train.values())
    LOGGER.info(
        f"{Style.BOLD}      Trains: {len(train_jobs)} | Evals: {n_evals} "
        f"| Total GPUs: {len(free_gpus)}{Style.RESET}"
    )
    LOGGER.info("=" * 60 + "\n")

    supervisors = []
    for i, train_job in enumerate(train_jobs, start=1):
        gpu_ids = _acquire_gpus(free_gpus, condition, train_job.num_gpus)
        evals = list(eval_jobs_by_train.get(train_job.name, ()))
        p = mp.Process(
            target=_run_train_supervisor,
            args=(train_job, evals, gpu_ids, free_gpus, condition, i, len(train_jobs), failed, failed_logs),
        )
        p.start()
        supervisors.append(p)
        time.sleep(0.1)

    LOGGER.info(
        "All train supervisors scheduled. Waiting for completion (incl. their evals)...",
        extra={"icon": "💤", "color": Style.MAGENTA},
    )
    for p in supervisors:
        p.join()


def group_evals_by_train(train_jobs: Sequence[Job], eval_jobs: Sequence[Job]) -> Dict[str, List[Job]]:
    by_name: Dict[str, List[Job]] = {t.name: [] for t in train_jobs}
    for ev in eval_jobs:
        parent = ev.name.split("::", 1)[0]
        by_name.setdefault(parent, []).append(ev)
    return by_name


def run_pipeline(gpus: str, train_jobs: Sequence[Job] = (), eval_jobs: Sequence[Job] = (),
                 eval_jobs_by_train: Dict[str, Sequence[Job]] = None):
    manager = mp.Manager()
    free_gpus = manager.list(parse_gpu_list(gpus))
    condition = manager.Condition()
    failed = manager.list()
    failed_logs = manager.list()

    if eval_jobs_by_train is not None:
        schedule_interleaved(train_jobs, eval_jobs_by_train, free_gpus, condition, failed, failed_logs)
        LOGGER.info("ALL TRAIN+EVAL TASKS COMPLETED!", extra={"icon": "🎉", "color": Style.GREEN})
    else:
        if train_jobs:
            schedule_jobs(train_jobs, "TRAIN JOB SCHEDULER STARTING", free_gpus, condition, failed, failed_logs)
            LOGGER.info("ALL TRAIN TASKS COMPLETED!", extra={"icon": "🎉", "color": Style.GREEN})

        if eval_jobs:
            schedule_jobs(eval_jobs, "EVALUATION JOB SCHEDULER STARTING", free_gpus, condition, failed, failed_logs)
            LOGGER.info("ALL EVALUATION TASKS COMPLETED!", extra={"icon": "🎉", "color": Style.GREEN})

    if failed:
        LOGGER.info("=" * 10 + " FAILED JOBS " + "=" * 10)
        for cmd in failed:
            LOGGER.info(cmd)

        logs_dir = Path("runner_logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmds_file = logs_dir / f"{timestamp}_failed_jobs"
        log_file = logs_dir / f"{timestamp}_failed_jobs.log"
        cmds_file.write_text("\n".join(failed) + "\n")
        log_file.write_text("".join(failed_logs))
        LOGGER.info(f"Saved {len(failed)} failed commands → {cmds_file}",
                    extra={"icon": "💾", "color": Style.YELLOW})
