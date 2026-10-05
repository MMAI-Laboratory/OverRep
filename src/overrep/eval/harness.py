import gc
import os
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Annotated

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import typer
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import get_max_memory
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config, make_table
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typer import Argument, Option

from overrep.common.utils import auto_dispatch_model, configure_cuda, convert_model_dtype
from overrep.models import create_model

configure_cuda()

app = typer.Typer(pretty_exceptions_enable=False)

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Model Parameters"
HELP_PANEL_NAME_3 = "Tokenizer Parameters"
HELP_PANEL_NAME_4 = "Evaluation Parameters"
GENERATION_TASKS = ['coqa', 'gsm8k', 'triviaqa']


def save_to_csv(name, param, results, file_name=None):
    if file_name is None:
        file_name = 'results.csv'
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("Param(B),Model,Task,Version,N-shot,Metric,Value,Stderr\n")

    if "groups" in results:
        column_name = 'groups'
    else:
        column_name = 'results'

    keys = results[column_name].keys()
    with open(file_name, "+a") as file:
        for key in keys:
            dic = results[column_name][key]
            version = results["versions"].get(key, "N/A")
            n_shot = results.get("n-shot", {}).get(key, 0)
            display_key = dic.get("alias", key)

            metric_items = sorted((k, v) for k, v in dic.items() if k != "alias")

            for (mf), v in metric_items:
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" + "," + f in dic:
                    se = dic[m + "_stderr" + "," + f]
                    file.write(f"{param},{name},{display_key},{version},{n_shot},{m},{v},{se}\n")


def task_manage(tasks):
    task_manager = TaskManager()
    task_list = tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task for task in task_list if task not in task_names and "*" not in task
    ]

    if task_missing:
        missing = ", ".join(task_missing)
        raise ValueError(
            f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
        )
    return task_names


def set_parallelization_kwargs(accelerator, max_memory_per_gpu=0.95):
    num_local_processes = accelerator.num_processes
    kwargs = dict()
    max_memory_all_gpus = get_max_memory()
    if "cpu" in max_memory_all_gpus:
        del max_memory_all_gpus["cpu"]

    max_memory_per_gpu_map = {
        k: v * max_memory_per_gpu
        for k, v in max_memory_all_gpus.items()
        if k % num_local_processes
           == (accelerator.process_index % num_local_processes)
    }
    kwargs["max_memory"] = max_memory_per_gpu_map
    kwargs["device_map"] = "auto"
    kwargs["offload_folder"] = "./offload"

    return kwargs


def check_results(name, tasks, num_fewshot, save_file=None):
    results_path = save_file or 'results.csv'
    if not os.path.exists(results_path):
        return tasks

    df = pd.read_csv(results_path)
    requested = [t.strip() for t in tasks.split(',') if t.strip()]
    remaining = [
        t for t in requested
        if df[(df['Model'] == name) & (df['Task'] == t) & (df['N-shot'] == num_fewshot)].empty
    ]

    if not remaining:
        print(f"Results for {tasks} already exist. SKIP")
        raise SystemExit(0)

    return ','.join(remaining)


@app.command()
def main(
        model_name: Annotated[str, Argument(help="Name of transformers model or your custom model)")],
        tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
        custom: Annotated[bool, Option(help="Whether use local model.", rich_help_panel=HELP_PANEL_NAME_2)] = False,
        attn: Annotated[
            str, Option(
                help="Attention mechanism for transformers models. Select ['eager', 'sdpa', 'flash_attention_2'].",
                rich_help_panel=HELP_PANEL_NAME_2)] = "sdpa",
        parallel: Annotated[
            bool, Option(help="Implement model parallelization when the model is too large to import on single GPU.",
                         rich_help_panel=HELP_PANEL_NAME_2)] = False,
        model_compile: Annotated[
            bool, Option(help="Compile model for fast inference.", rich_help_panel=HELP_PANEL_NAME_2)] = False,
        trust_remote_code: Annotated[
            bool, Option(help="Allow custom modeling code from the model repo.",
                         rich_help_panel=HELP_PANEL_NAME_2)] = True,
        checkpoint: Annotated[str, Option(help="Checkpoint of safetensors", rich_help_panel=HELP_PANEL_NAME_2)] = None,
        mixed_precision: Annotated[str, Option(help="Mixed precision type. [no | fp8 | fp16 | bf16]",
                                               rich_help_panel=HELP_PANEL_NAME_2)] = 'no',
        tokenizer_name: Annotated[
            str, Option(help="Name of tokenizer. If none, it will be model_name.",
                        rich_help_panel=HELP_PANEL_NAME_3)] = None,
        fast: Annotated[
            bool, Option(help="Use fast tokenizer.", rich_help_panel=HELP_PANEL_NAME_3)] = True,
        num_fewshot: Annotated[
            int, Option(help="Number of few shot samples", rich_help_panel=HELP_PANEL_NAME_4)] = 0,
        max_length: Annotated[
            int, Option(help="Max length", rich_help_panel=HELP_PANEL_NAME_4)] = None,
        batch_size: Annotated[
            str, Option(help="Number of batch size", rich_help_panel=HELP_PANEL_NAME_4)] = 'auto',
        log_samples: Annotated[
            bool, Option(
                help="Write out all model outputs and documents for per-sample measurement and post-hoc analysis",
                rich_help_panel=HELP_PANEL_NAME_4)] = False,
        name: Annotated[
            str, Option(help="Experimental name to save the results.", rich_help_panel=HELP_PANEL_NAME_1)] = None,
        save: Annotated[
            bool, Option(help="Whether to save the experimental results", rich_help_panel=HELP_PANEL_NAME_1)] = True,
        include_layer: Annotated[
            str, Option(help="Recovery-block mask as `target_layer:layer_interval`, e.g. `17:9`.",
                        rich_help_panel=HELP_PANEL_NAME_2)] = None,
        force_nogen: Annotated[
            bool, Option(help="Run generation tasks through the HF backend instead of vLLM (same as --hf-gen).",
                         rich_help_panel=HELP_PANEL_NAME_1)] = False,
        hf_gen: Annotated[
            bool, Option(help="Force the HF backend for ALL generation tasks (no vLLM).",
                         rich_help_panel=HELP_PANEL_NAME_1)] = False,
        gen_kwargs: Annotated[
            str, Option(help="Generation kwargs override, e.g. 'max_gen_toks=256' "
                             "(degenerate no-stop outputs cap).",
                        rich_help_panel=HELP_PANEL_NAME_4)] = None,
        limit: Annotated[
            int, Option(help="Limit samples per task (diagnostics only).",
                        rich_help_panel=HELP_PANEL_NAME_4)] = None,
        save_file: Annotated[
            str, Option(help="CSV path to append results to.", rich_help_panel=HELP_PANEL_NAME_1)] = None,
):
    """Evaluate a backbone or an OverRep checkpoint on the given lm_eval tasks."""
    if limit is None:
        tasks = check_results(name, tasks, num_fewshot, save_file)
    else:
        print(f"[diagnostic] --limit {limit}: skip-check and CSV save are disabled "
              f"(raw JSON still written, tagged with the limit).")

    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], mixed_precision=mixed_precision)

    parallel_kwargs = {}
    parallel = parallel or torch.cuda.device_count() > 1
    if parallel:
        if custom:
            parallel_kwargs = set_parallelization_kwargs(accelerator)
            parallel_kwargs["device_map"] = "cpu"
        else:
            parallel_kwargs = {"device_map": "auto", "torch_dtype": "auto"}

    if custom:
        config = AutoConfig.from_pretrained(tokenizer_name)
        model = create_model(model_name, checkpoint=checkpoint, parallel_kwargs=parallel_kwargs,
                             include_layer=include_layer, config_name=tokenizer_name)
        convert_model_dtype(model, config.torch_dtype)
        if parallel:
            model = auto_dispatch_model(model, model.dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn,
            **parallel_kwargs,
        )
    if model_compile:
        model = torch.compile(model)

    if accelerator.is_main_process:
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Model Size: {n_params / 1e6:.2f}M")
    else:
        n_params = 0

    tokenizer_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=fast, legacy=False, fix_mistral_regex=True)

    tasks_list = tasks.split(',')
    is_generation = any(t.strip() in GENERATION_TASKS for t in tasks_list)
    if force_nogen or hf_gen:
        is_generation = False
    save_path = None
    can_skip_save = (not custom) and (checkpoint is None) and (include_layer is None)
    try:
        if is_generation:
            if can_skip_save:
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                vllm_model_path = model_name
            else:
                save_path = Path('.cache_weight') / str(uuid.uuid4())
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved model to {save_path}")
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                vllm_model_path = str(save_path)

            model = 'vllm'
            model_args = f'pretrained={vllm_model_path},max_model_len=4096'
            if 'Llama-2' in tokenizer_name:
                model_args += ',tokenizer_mode=slow'
        else:
            if any(t.strip() in ['c4'] for t in tasks_list):
                max_length = 2048

            _dmap = getattr(model, "hf_device_map", None)
            if _dmap and len(set(_dmap.values())) > 1:
                model.eval()
            else:
                model = accelerator.prepare_model(model)
                model.eval()
            # "auto" is not passed to HFLM (batch size 1): its auto-detection runs out of memory on generate_until
            hflm_kwargs = {} if batch_size == "auto" else {"batch_size": int(batch_size)}
            model = HFLM(pretrained=model, tokenizer=tokenizer, accelerator=accelerator, backend="causal",
                         max_length=max_length, **hflm_kwargs)
            model_args = None

        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=task_manage(tasks),
            verbosity="WARNING",
            num_fewshot=num_fewshot,
            batch_size=batch_size if batch_size == "auto" else int(batch_size),
            log_samples=log_samples,
            gen_kwargs=gen_kwargs,
            limit=limit,
        )
    finally:
        if save_path is not None and accelerator.is_main_process:
            shutil.rmtree(save_path, ignore_errors=True)

    if accelerator.is_main_process:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if save and limit is None:
            name = name or model_name
            save_to_csv(name, f"{n_params / 1e9:.2f}", results, save_file)
        if save:
            import json
            import time
            name = name or model_name
            raw_dir = Path(save_file).parent / 'raw' if save_file else Path('results/raw')
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw = {k: v for k, v in results.items() if k != 'samples'}
            stamp = time.strftime('%Y%m%dT%H%M%S')
            def safe(value):
                return str(value).replace('/', '--').replace(' ', '')
            limit_tag = f"__limit{limit}" if limit is not None else ""
            raw_path = raw_dir / f"{safe(name)}__{safe(tasks.replace(',', '+'))[:60]}__n{num_fewshot}{limit_tag}__{stamp}.json"
            raw_path.write_text(json.dumps(raw, indent=1, default=str))
            print(f"raw results saved -> {raw_path}")

    accelerator.wait_for_everyone()
    return


def cli():
    app()


if __name__ == '__main__':
    app()
