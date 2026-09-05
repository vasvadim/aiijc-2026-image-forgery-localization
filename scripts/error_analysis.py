# -*- coding: utf-8 -*-
"""Разбор ошибок чекпоинта на val-tune по probs/: типы ошибок на позитивах, ложные тревоги на негативах,
вклад в дефицит Dice/FPR по доменам/площадям/генераторам + миниатюры примеров (base64 JPEG).
    python error_analysis.py <dir с probs/ и img_probs.csv> <threshold.json> <out_dir>
"""
import sys, os, json, base64
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.expanduser('~/aic2026/code_vadim/src'))
from common import data_dir, load_split

SRC = Path(sys.argv[1]); THR = json.load(open(sys.argv[2])); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
T, AMIN, TAU = float(THR['t']), float(THR.get('a_min', 0)), float(THR.get('tau', 0))
D = data_dir()

def stats(args):
    stem, gt_path, p_img = args
    p = cv2.imread(str(SRC / 'probs' / f'{stem}.png'), 0)
    if p is None: return None
    pred = p >= int(round(T * 255))
    area = float(pred.mean())
    gated = p_img < TAU
    if gated or area < AMIN: pred = np.zeros_like(pred); area_eff = 0.0
    else: area_eff = area
    r = dict(stem=stem, area_raw=area, area=area_eff, gated=bool(gated), h=p.shape[0], w=p.shape[1])
    if isinstance(gt_path, str) and gt_path:
        gt = cv2.imread(str(D / gt_path), 0) > 128
        inter = float(np.logical_and(pred, gt).sum()); ps, gs = float(pred.sum()), float(gt.sum())
        r.update(gt_area=gs / gt.size, dice=(2 * inter / (ps + gs)) if ps + gs > 0 else 1.0,
                 precision=inter / ps if ps > 0 else 0.0, recall=inter / gs if gs > 0 else 0.0,
                 ncomp_gt=int(cv2.connectedComponents(gt.astype(np.uint8))[0] - 1))
    return r

def thumb(stem, img_path, gt_path, maxw=360):
    img = cv2.imread(str(D / img_path)); p = cv2.imread(str(SRC / 'probs' / f'{stem}.png'), 0)
    pred = p >= int(round(T * 255))
    over = img.copy().astype(np.float32)
    red = np.zeros_like(over); red[..., 2] = 255
    over = np.where(pred[..., None], over * 0.55 + red * 0.45, over)
    if isinstance(gt_path, str) and gt_path:
        gt = (cv2.imread(str(D / gt_path), 0) > 128).astype(np.uint8)
        cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 400))
    over = over.astype(np.uint8)
    s = maxw / over.shape[1]; over = cv2.resize(over, (maxw, int(over.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', over, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf.tobytes()).decode()

def main():
    df = load_split('v1', 'val-tune').copy()
    df['stem'] = df.chng_img_path.map(lambda s: Path(s).stem); df = df.drop(columns=['gt_area', 'H', 'W'], errors='ignore')
    ip = pd.read_csv(SRC / 'img_probs.csv')[['stem', 'p_img']]; ip['stem'] = ip.stem.astype(str)
    df = df.merge(ip, on='stem', how='left'); df['p_img'] = df.p_img.fillna(1.0)
    with Pool(32) as pool:
        rows = pool.map(stats, [(s, g if isinstance(g, str) else '', float(pi)) for s, g, pi in zip(df.stem, df.gt_path, df.p_img)], chunksize=64)
    st = pd.DataFrame([r for r in rows if r]); df = df.merge(st, on='stem')
    pos = df[df.is_neg == 0].copy(); neg = df[df.is_neg == 1].copy()
    neg['fa'] = neg.area >= 0.01
    # ---- типы ошибок на позитивах
    def etype(r):
        if r.dice >= 0.5: return 'ok'
        if r.gated: return 'missed_gate'
        if r.area == 0: return 'missed_mask'
        if r.precision >= 0.6 and r.recall < 0.5: return 'under'
        if r.recall >= 0.6 and r.precision < 0.5: return 'over'
        return 'misplaced'
    pos['etype'] = pos.apply(etype, axis=1)
    pos['gt_bin'] = pd.cut(pos.gt_area, [0, 0.01, 0.05, 0.2, 1.0], labels=['<1%', '1–5%', '5–20%', '>20%'])
    pos['loss'] = 1 - pos.dice
    N = len(pos)
    rep = dict(t=T, a_min=AMIN, tau=TAU, n_pos=N, n_neg=len(neg),
               dice_pos=float(pos.dice.mean()),
               fpr={p: [float(g.fa.mean()), int(len(g))] for p, g in neg.groupby('neg_pool')})
    rep['fpr_main'] = float(np.mean([rep['fpr'][p][0] for p in ('neg_train', 'neg_src')]))
    Dv, Fv = rep['dice_pos'], rep['fpr_main']; rep['aic'] = 2 * Dv * (1 - Fv) / (Dv + 1 - Fv)
    # чувствительность AIC: производные по D и по F
    rep['dAIC_dD'] = 2 * (1 - Fv) ** 2 / (Dv + 1 - Fv) ** 2; rep['dAIC_dF'] = -2 * Dv ** 2 / (Dv + 1 - Fv) ** 2
    def budget(g, key):
        out = {}
        for k, s in g.groupby(key, observed=True):
            out[str(k)] = dict(n=int(len(s)), share=len(s) / N, dice=float(s.dice.mean()), loss_contrib=float(s.loss.sum() / N),
                               empty=float((s.area == 0).mean()), gated=float(s.gated.mean()), dice_lt05=float((s.dice < 0.5).mean()))
        return out
    rep['pos_by_etype'] = budget(pos, 'etype'); rep['pos_by_domain'] = budget(pos, 'domain'); rep['pos_by_gtbin'] = budget(pos, 'gt_bin')
    rep['pos_by_generator'] = budget(pos, 'generator')
    rep['pos_by_domain_gtbin'] = {d: budget(g, 'gt_bin') for d, g in pos.groupby('domain')}
    rep['pos_by_ncomp'] = budget(pos.assign(nc=pos.ncomp_gt.clip(upper=4)), 'nc')
    rep['dice_hist'] = np.histogram(pos.dice, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001])[0].tolist()
    rep['pimg_pos_q'] = [float(x) for x in pos.p_img.quantile([0.01, 0.05, 0.1, 0.25, 0.5])]
    rep['pimg_neg_q'] = [float(x) for x in neg.p_img.quantile([0.5, 0.75, 0.9, 0.95, 0.99])]
    rep['pimg_neg_hist'] = np.histogram(neg.p_img, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001])[0].tolist()
    # ---- негативы
    fa = neg[neg.fa]
    rep['neg_by_domain'] = {f'{p}:{d}': dict(n=int(len(g)), fa=float(g.fa.mean()), n_fa=int(g.fa.sum())) for (p, d), g in neg.groupby(['neg_pool', 'domain'])}
    rep['fa_area_hist'] = dict(bins=['1–2%', '2–5%', '5–20%', '>20%'], counts=np.histogram(fa.area, bins=[0.01, 0.02, 0.05, 0.2, 1.0001])[0].tolist())
    rep['neg_area_raw_hist'] = dict(bins=['0', '<0.5%', '0.5–1%', '1–2%', '2–5%', '>5%'], counts=[int((neg.area_raw == 0).sum())] + np.histogram(neg.area_raw[neg.area_raw > 0], bins=[0, 0.005, 0.01, 0.02, 0.05, 1.0001])[0].tolist())
    rep['neg_gated_share'] = float(neg.gated.mean()); rep['neg_mask_fires_share'] = float((neg.area_raw >= 0.01).mean())
    rep['neg_both_fire'] = float(((neg.area_raw >= 0.01) & (~neg.gated)).mean())
    rep['fa_pimg_q'] = [float(x) for x in fa.p_img.quantile([0.1, 0.5, 0.9])] if len(fa) else None
    # что было бы без гейтинга и с a_min
    rep['whatif'] = {}
    for tau in (0.0, 0.5, 0.8, 0.9, 0.95):
        for am in (0.0, 0.01, 0.02):
            pe = pos.copy(); ne = neg.copy()
            pe_empty = (pe.p_img < tau) | (pe.area_raw < am); ne_empty = (ne.p_img < tau) | (ne.area_raw < am)
            d = float(np.where(pe_empty, 0.0, pe.dice.where(~pe.gated, np.nan).fillna(0)).mean()) if tau >= TAU else None
            f = float(np.mean([((~ne_empty) & (ne.area_raw >= 0.01))[ne.neg_pool == p].mean() for p in ('neg_train', 'neg_src')]))
            rep['whatif'][f'tau={tau},a_min={am}'] = dict(fpr=f, dice=d)
    # ---- примеры
    ex = {}
    pick = lambda g, n: g.sort_values('loss', ascending=False).head(n)
    for et in ('missed_gate', 'missed_mask', 'under', 'over', 'misplaced'):
        g = pos[pos.etype == et]
        if len(g) == 0: continue
        sel = g.sample(min(6, len(g)), random_state=0) if et.startswith('missed') else pick(g, 6)
        ex[et] = [dict(stem=r.stem, domain=r.domain, gen=r.generator, dice=round(r.dice, 3), gt_area=round(r.gt_area, 4), area=round(r.area, 4), p_img=round(r.p_img, 3), img=thumb(r.stem, r.chng_img_path, r.gt_path)) for r in sel.itertuples()]
    fa_sorted = fa.sort_values('area', ascending=False)
    ex['fa_big'] = [dict(stem=r.stem, domain=f'{r.neg_pool}:{r.domain}', area=round(r.area, 4), p_img=round(r.p_img, 3), img=thumb(r.stem, r.chng_img_path, '')) for r in fa_sorted.head(8).itertuples()]
    ex['fa_border'] = [dict(stem=r.stem, domain=f'{r.neg_pool}:{r.domain}', area=round(r.area, 4), p_img=round(r.p_img, 3), img=thumb(r.stem, r.chng_img_path, '')) for r in fa_sorted.tail(4).itertuples()]
    good = pos[(pos.dice > 0.95)].sample(4, random_state=1)
    ex['good'] = [dict(stem=r.stem, domain=r.domain, gen=r.generator, dice=round(r.dice, 3), gt_area=round(r.gt_area, 4), p_img=round(r.p_img, 3), img=thumb(r.stem, r.chng_img_path, r.gt_path)) for r in good.itertuples()]
    (OUT / 'analysis.json').write_text(json.dumps(rep, indent=1, default=float))
    (OUT / 'examples.json').write_text(json.dumps(ex))
    pos.drop(columns=['chng_img_path', 'gt_path', 'orgl_img_path'], errors='ignore').to_csv(OUT / 'pos.csv', index=False)
    neg.drop(columns=['chng_img_path', 'gt_path', 'orgl_img_path'], errors='ignore').to_csv(OUT / 'neg.csv', index=False)
    print(json.dumps({k: rep[k] for k in ('dice_pos', 'fpr', 'fpr_main', 'aic', 'dAIC_dD', 'dAIC_dF', 'pos_by_etype', 'neg_gated_share', 'neg_mask_fires_share', 'neg_both_fire')}, indent=1, default=float))

if __name__ == '__main__':
    main()
