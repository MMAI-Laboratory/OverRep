"""Train and evaluate every cell of the grid in `build_train_jobs` over a GPU pool.

    OVERREP_GPUS=0,1,2,3 overrep-runner
"""
import os
from typing import List

from overrep.runners.config import (
    CONFIG,
    MIXED_PRECISION,
    Job,
    dataset_alias,
    dict_product,
    group_evals_by_train,
    include_layer_for,
    model_name_for,
    run_pipeline,
)

GPUS = os.environ.get("OVERREP_GPUS", "0")
EVAL_GPU = 1

SINGLE_GPU_BACKBONES = ('3-3b', 'q3-4b')

EVAL_PROTOCOL = [
    (0, 'coqa'), (8, 'gsm8k'), (5, 'triviaqa'),
    (0, 'hellaswag'), (0, 'winogrande,arc_easy,mathqa,race'), (0, 'mmlu'),
    (0, 'arc_challenge,openbookqa,piqa,boolq'),
]


def build_train_jobs() -> List[Job]:
    grid = {
        "backbone": ['3-3b'],
        "model_type": ['OverRep'],
        "pruning_ratio": ["Half", "Qtr"],
    }
    jobs = []
    for exp in dict_product(grid):
        backbone = exp["backbone"]
        pruning_ratio = exp["pruning_ratio"]

        model_name = model_name_for(backbone, exp["model_type"])
        include_layer = include_layer_for(backbone, pruning_ratio)
        target_layer, interval = include_layer.split(':')

        single_gpu = backbone in SINGLE_GPU_BACKBONES
        batch_size, accum = (8, 1) if single_gpu else (1, 8)
        exp_name = f"{pruning_ratio}_{model_name}"

        jobs.append(Job(
            script="overrep-train",
            args=[
                f"--custom --config_name {CONFIG[backbone]} --model_name_or_path {model_name}",
                f"--dataset data/finewebedu_{dataset_alias(backbone)}_manifest "
                f"--target_layer {target_layer} --layer_interval {interval}",
                f"--epochs 20 --batch_size {batch_size} --gradient_accumulation_steps {accum} "
                f"--mixed_precision {MIXED_PRECISION[backbone]} --num_workers 4 --should_log",
                "--learning_rate 1e-4 --weight_decay 1e-2 --min_learning_rate 1e-5 --va --rep_norm",
                f"--name {exp_name}",
            ],
            config_name=CONFIG[backbone],
            include_layer=include_layer,
            name=exp_name,
            num_gpus=1 if single_gpu else 2,
            stage="train",
        ))
    return jobs


def build_eval_jobs(train_jobs: List[Job]) -> List[Job]:
    jobs = []
    for job in train_jobs:
        for n_shot, dataset in EVAL_PROTOCOL:
            jobs.append(Job(
                script="overrep-eval",
                args=[
                    f"deploy_model --custom --tokenizer-name {job.config_name} --force-nogen --hf-gen",
                    f"--name {job.name} --include-layer {job.include_layer}",
                    f"--checkpoint output/{job.name} --num-fewshot {n_shot} {dataset}",
                ],
                name=f"{job.name}::{dataset}",
                num_gpus=EVAL_GPU,
                stage="eval",
            ))
    return jobs


def main():
    train_jobs = build_train_jobs()
    eval_jobs = build_eval_jobs(train_jobs)
    run_pipeline(
        GPUS,
        train_jobs=train_jobs,
        eval_jobs_by_train=group_evals_by_train(train_jobs, eval_jobs),
    )


if __name__ == "__main__":
    main()
