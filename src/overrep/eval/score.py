"""Aggregate `overrep-eval` CSVs into the paper tables.

rea: 10-task 0-shot average (acc_norm for hellaswag/arc_challenge/mathqa/piqa/openbookqa, acc for the rest)
gen: coqa F1 (0-shot), gsm8k EM (8-shot), triviaqa EM (5-shot)
ppl: wikitext/c4 word perplexity, not included in the averages
RP(%) = pruned / dense x 100
"""
import sys
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from typer import Option

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _app():
    """Aggregate overrep-eval CSVs into the paper-convention table."""


REA = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'mathqa',
       'mmlu', 'openbookqa', 'piqa', 'race', 'winogrande']
GEN = ['coqa', 'gsm8k', 'triviaqa']
PPL = ['wikitext', 'c4']
TASKS = REA + GEN
ACC_NORM = {'hellaswag', 'arc_challenge', 'mathqa', 'piqa', 'openbookqa'}
N_SHOT = {**{t: 0 for t in REA}, 'coqa': 0, 'gsm8k': 8, 'triviaqa': 5,
          'wikitext': 0, 'c4': 0}


def _warn(msg):
    print(f"[score:warn] {msg}", file=sys.stderr)


def _pick(task, metrics, model=''):
    if task in PPL:
        vals = [v for k, v in metrics.items() if 'word_perplexity' in k]
        return min(vals) if vals else None
    if task == 'coqa':
        if 'f1' not in metrics:
            _warn(f"{model}/{task}: no f1 metric (found {sorted(metrics)}); leaving the cell empty")
        return metrics.get('f1')
    if task in ('gsm8k', 'triviaqa'):
        vals = [v for k, v in metrics.items() if k in ('em', 'exact_match')]
        if not vals:
            _warn(f"{model}/{task}: no EM metric; leaving the cell empty")
        return max(vals) if vals else None
    want = 'acc_norm' if task in ACC_NORM else 'acc'
    if want not in metrics:
        _warn(f"{model}/{task}: metric '{want}' missing (found {sorted(metrics)}); "
              f"leaving the cell empty rather than falling back")
        return None
    return metrics[want]


def wide_scores(df: pd.DataFrame, include_ppl: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df[df['Task'].astype(str).str.strip().str.lower() != 'task']
    df['Task'] = df['Task'].astype(str).str.strip().str.lower()
    df['Metric'] = df['Metric'].astype(str).str.strip()
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['N-shot'] = pd.to_numeric(df['N-shot'], errors='coerce')
    n_bad = df['N-shot'].isna().sum()
    if n_bad:
        _warn(f"ignoring {n_bad} rows whose N-shot is not numeric")
    all_tasks = TASKS + (PPL if include_ppl else [])

    exact, disp = {}, {}
    for model, mg in df.groupby('Model', sort=False):
        e_rec, d_rec = {}, {}
        for task in all_tasks:
            tg = mg[(mg['Task'] == task) & (mg['N-shot'] == N_SHOT[task])]
            if tg.empty:
                continue
            metrics = tg.groupby('Metric')['Value'].max().to_dict()
            v = _pick(task, metrics, model=model)
            if v is not None and pd.notna(v):
                if task in PPL:
                    e_rec[task] = float(v)
                    d_rec[task] = round(float(v), 2)
                else:
                    e_rec[task] = float(v) * 100
                    d_rec[task] = round(float(v) * 100, 1)
        missing = [t for t in TASKS if t not in e_rec]
        if missing and len(missing) < len(TASKS):
            _warn(f"{model}: missing tasks {missing}; averaging over the rest")
        exact[model], disp[model] = e_rec, d_rec

    cols = TASKS + ([t for t in PPL if any(t in r for r in disp.values())] if include_ppl else [])
    out = pd.DataFrame.from_dict(disp, orient='index').reindex(index=list(disp), columns=cols)

    def _avg(model, keys):
        vals = [exact[model][t] for t in keys if t in exact[model]]
        return round(sum(vals) / len(vals), 1) if vals else float('nan')

    out['Rea.Avg'] = [_avg(m, REA) for m in out.index]
    out['Gen.Avg'] = [_avg(m, GEN) for m in out.index]
    return out


def _rich(df, title=''):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta", title=title)
    table.add_column('Model')
    for col in df.columns:
        table.add_column(str(col))
    for idx, row in df.iterrows():
        table.add_row(str(idx), *[('-' if pd.isna(v) else f'{v}') for v in row.values])
    console.print(table)


@app.command(help='Raw eval CSV → paper-convention table (+RP vs dense).')
def report(
        filename: Annotated[str, Option(help="Raw results CSV (overrep-eval --save-file)")] = "results.csv",
        dense: Annotated[str, Option(help="Dense reference Model name for RP(%)")] = None,
        save: Annotated[str, Option(help="Save wide table to CSV")] = None,
        filtering: Annotated[str, Option(help="Model-name substring filter; comma-separated values are unioned")] = None,
):
    df = pd.read_csv(filename)
    wide = wide_scores(df)
    if filtering:
        keys = [k.strip() for k in filtering.split(',') if k.strip()]
        keep = [m for m in wide.index
                if any(k in m for k in keys) or (dense and m == dense)]
        wide = wide.loc[keep]

    if dense:
        if dense not in wide.index:
            raise typer.BadParameter(f"dense model '{dense}' not found in {filename}")
        ref = wide.loc[dense]
        for col in ['Rea.Avg', 'Gen.Avg']:
            wide[f'{col}.RP'] = (wide[col] / ref[col] * 100).round(1)

    _rich(wide, title=f'{filename}  (paper conventions)')
    if save:
        wide.to_csv(save)
        print(f'saved -> {save}')


def cli():
    app()


if __name__ == '__main__':
    app()
