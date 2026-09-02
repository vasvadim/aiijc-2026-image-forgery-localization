#!/usr/bin/env python
"""wandb_mirror.py — зеркалит ран обучения в W&B (онлайн) из файлов run_dir, скачанных с кластера.

Кластер не имеет доступа к api.wandb.ai, а `wandb sync` offline-ранов с этой машины зависает, поэтому
источник правды — обычные файлы, которые пишет train.py: log.jsonl (эпохи), steps.jsonl (шаги),
panel/*.png (картинки), best.json, threshold.json, val-report/metrics.json, control/metrics.json, flops.json.
Скрипт идемпотентен: помнит, что уже отправлено (mirror_state.json), и досылает только новое.
Id рана берётся из wandb_id.txt (тот же, что у offline-рана на кластере), так что ран один.

    python scripts/wandb_mirror.py --dir runs/_wandb_sync/vasilenkova/<run_name> --run-name <run_name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = 'aic2026-detective'


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def epoch_dict(rec: dict, select_by: str, best_epoch: dict | None) -> dict:
    pm, fm = rec['proxy'], rec.get('full')
    d = {'epoch': rec['epoch'], 'train/loss': rec['train_loss'], 'val/loss': rec['val_loss'],
         'val/gap_val_minus_train': rec['val_loss'] - rec['train_loss'], 'lr': rec.get('lr'),
         'train/grad_norm_epoch': rec.get('grad_norm'), 'time/epoch_s': rec.get('time_s'),
         'time/train_s': rec.get('train_time_s'), 'gpu/max_mem_gb': rec.get('gpu_max_mem_gb'),
         'proxy/dice_pos': pm['dice_pos'], 'proxy/fpr_main': pm['fpr_main'], 'proxy/aic': pm['aic'],
         'proxy/empty_pred_on_pos': pm.get('empty_pred_on_pos'), 'proxy/neg_pred_area_median': pm.get('neg_pred_area_median')}
    d.update({f'proxy/fpr_{k}': v for k, v in pm.get('fpr', {}).items()})
    d.update({f'proxy/dice_domain_{k}': v for k, v in pm.get('dice_by_domain', {}).items()})
    d.update({f'proxy/dice_area_{k}': v for k, v in pm.get('dice_by_area', {}).items()})
    if fm:
        d.update({'full/dice_pos': fm['dice_pos'], 'full/fpr_main': fm['fpr_main'], 'full/aic': fm['aic']})
        d.update({f'full/fpr_{k}': v for k, v in fm.get('fpr', {}).items()})
    if best_epoch:
        d['epochs_since_best'] = rec['epoch'] - best_epoch.get(select_by, 0)
    return {k: v for k, v in d.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='локальная копия run_dir (log.jsonl, config.json, ...)')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--project', default=PROJECT)
    args = ap.parse_args()
    d = Path(args.dir)
    epochs = load_jsonl(d / 'log.jsonl')
    steps = load_jsonl(d / 'steps.jsonl')
    if not epochs and not steps:
        print('нечего зеркалить (нет log.jsonl/steps.jsonl)'); return
    config = json.loads((d / 'config.json').read_text(encoding='utf-8')) if (d / 'config.json').exists() else {}
    select_by = config.get('select_by', 'proxy')
    wid = (d / 'wandb_id.txt').read_text().strip() if (d / 'wandb_id.txt').exists() else hashlib.md5(args.run_name.encode()).hexdigest()[:8]
    state_p = d / 'mirror_state.json'
    state = json.loads(state_p.read_text()) if state_p.exists() else {'epochs': 0, 'step': 0}
    best_epoch = None
    # best_epoch по мере эпох (для epochs_since_best) — пересчитываем локально
    import wandb
    run = wandb.init(project=args.project, name=args.run_name, id=wid, resume='allow', config=config, dir=str(d),
                     settings=wandb.Settings(init_timeout=90))
    wandb.define_metric('epoch'); wandb.define_metric('global_step')
    for m in ('train/step_loss', 'train/grad_norm', 'lr_step'):
        wandb.define_metric(m, step_metric='global_step')
    for prefix in ('train/', 'val/', 'proxy/', 'full/', 'time/', 'gpu/', 'images/'):
        wandb.define_metric(prefix + '*', step_metric='epoch')
    wandb.define_metric('lr', step_metric='epoch'); wandb.define_metric('epochs_since_best', step_metric='epoch')
    print(f'run: {run.url}')

    n_steps = 0
    for s in steps:
        if s['global_step'] > state['step']:
            run.log({'train/step_loss': s['loss'], 'train/grad_norm': s['grad_norm'], 'lr_step': s.get('lr'), 'global_step': s['global_step']})
            state['step'] = s['global_step']; n_steps += 1

    best = {'loss': float('inf'), 'proxy': -1.0, 'full': -1.0}; best_epoch = {'loss': 0, 'proxy': 0, 'full': 0}
    n_ep = 0
    for rec in epochs:
        e = rec['epoch']
        if rec['val_loss'] < best['loss']:
            best['loss'] = rec['val_loss']; best_epoch['loss'] = e
        if rec['proxy']['aic'] > best['proxy']:
            best['proxy'] = rec['proxy']['aic']; best_epoch['proxy'] = e
        if rec.get('full') and rec['full']['aic'] > best['full']:
            best['full'] = rec['full']['aic']; best_epoch['full'] = e
        if e <= state['epochs']:
            continue
        dd = epoch_dict(rec, select_by, best_epoch)
        dd['best/proxy_aic'] = best['proxy']; dd['best/epoch_proxy'] = best_epoch['proxy']
        imgs = sorted((d / 'panel').glob(f'ep{e:02d}_*.png')) if (d / 'panel').exists() else []
        if imgs:
            dd['images/proxy_panel'] = [wandb.Image(str(p), caption=p.stem.split('_', 2)[-1]) for p in imgs]
        run.log(dd)
        state['epochs'] = e; n_ep += 1

    summ = {'best': best, 'best_epoch': best_epoch}
    for name, prefix in (('threshold.json', 'threshold'), ('flops.json', 'flops')):
        if (d / name).exists():
            j = json.loads((d / name).read_text(encoding='utf-8'))
            summ.update({f'{prefix}/{k}': v for k, v in j.items() if not isinstance(v, (list, dict))})
    for role in ('val-tune', 'val-report', 'control'):
        mp = d / role / 'metrics.json'
        if mp.exists():
            m = json.loads(mp.read_text(encoding='utf-8')).get('metrics', {}).get('main', {})
            if m:
                summ.update({f'{role}/aic': m.get('aic_main'), f'{role}/dice_pos': m.get('dice_pos'), f'{role}/fpr_main': m.get('fpr_main')})
                summ.update({f'{role}/fpr_{k}': v[0] for k, v in m.get('fpr', {}).items()})
    run.summary.update(summ)
    run.finish()
    state_p.write_text(json.dumps(state))
    print(f'mirrored: +{n_ep} epochs, +{n_steps} step points; total epochs {state["epochs"]}; url {run.url}')


if __name__ == '__main__':
    sys.exit(main())
