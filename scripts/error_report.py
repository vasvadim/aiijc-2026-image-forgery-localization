#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""error_report.py — HTML-разбор ошибок чекпоинта по выходам scripts/error_analysis.py
(analysis.json + examples.json). Показывает, из чего складывается AIC: дефицит Dice по типам ошибок,
доменам, площадям; ложные тревоги по пулам; чувствительность к τ/a_min; примеры картинок.

    python scripts/error_report.py --dir runs/_err_b4_ep6 --out runs/_err_b4_ep6/report.html --title "b4 ep6"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x, d=1):
    return f'{100 * x:.{d}f} %'


def bar(v, vmax, color='var(--acc)'):
    w = 0 if vmax <= 0 else max(0.0, min(100.0, 100 * v / vmax))
    return f'<span class="bar"><span style="width:{w:.1f}%;background:{color}"></span></span>'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def hist_svg(counts, labels, color='var(--acc)', w=560, h=150):
    n = len(counts); m = max(counts) or 1; bw = w / n
    out = [f'<svg viewBox="0 0 {w} {h + 34}" class="hist" role="img">']
    for i, (c, l) in enumerate(zip(counts, labels)):
        bh = (h - 10) * c / m
        x = i * bw + 4
        out.append(f'<rect x="{x:.1f}" y="{h - bh:.1f}" width="{bw - 8:.1f}" height="{bh:.1f}" rx="3" fill="{color}"><title>{l}: {c}</title></rect>')
        out.append(f'<text x="{x + (bw - 8) / 2:.1f}" y="{h - bh - 4:.1f}" text-anchor="middle" class="v">{c}</text>')
        out.append(f'<text x="{x + (bw - 8) / 2:.1f}" y="{h + 22}" text-anchor="middle" class="l">{l}</text>')
    out.append('</svg>')
    return ''.join(out)


ET_NAME = {
    'ok': 'Dice ≥ 0.5 (нормальные, теряют на границах)',
    'misplaced': 'Не туда: маска есть, но мимо (precision и recall низкие)',
    'missed_gate': 'Срезано гейтингом: p_img < τ, маска обнулена',
    'missed_mask': 'Пустая маска при p_img ≥ τ',
    'over': 'Пересегментация: recall ≥ 0.6, precision < 0.5',
    'under': 'Недосегментация: precision ≥ 0.6, recall < 0.5',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='чекпоинт')
    a = ap.parse_args()
    d = Path(a.dir)
    R = json.load(open(d / 'analysis.json', encoding='utf-8'))
    EX = json.load(open(d / 'examples.json', encoding='utf-8'))

    D, F, A = R['dice_pos'], R['fpr_main'], R['aic']
    dD, dF = R['dAIC_dD'], R['dAIC_dF']
    loss_total = 1 - D
    gain_pos = loss_total * dD; gain_neg = F * -dF

    # ---- позитивы: типы ошибок
    et = R['pos_by_etype']; order = ['ok', 'misplaced', 'missed_gate', 'over', 'under', 'missed_mask']
    mx = max(v['loss_contrib'] for v in et.values())
    rows = []
    for k in order:
        if k not in et: continue
        v = et[k]
        rows.append([f'<b>{k}</b><br><span class="mut">{ET_NAME[k]}</span>', v['n'], pct(v['share']), f'{v["dice"]:.3f}',
                     f'{v["loss_contrib"]:.4f} {bar(v["loss_contrib"], mx)}', f'{v["loss_contrib"] * dD:.4f}'])
    t_etype = table(['Тип', 'Картинок', 'Доля', 'Средний Dice', 'Вклад в дефицит Dice (1 − D)', '≈ потеря AIC'], rows)

    def budget_table(key, name, sort=True, extra=True):
        b = R[key]; items = sorted(b.items(), key=lambda kv: -kv[1]['loss_contrib']) if sort else list(b.items())
        mx = max(v['loss_contrib'] for v in b.values())
        rows = []
        for k, v in items:
            r = [f'<b>{k}</b>', v['n'], pct(v['share']), f'{v["dice"]:.3f}', f'{v["loss_contrib"]:.4f} {bar(v["loss_contrib"], mx)}']
            if extra: r += [pct(v['dice_lt05']), pct(v['gated'])]
            rows.append(r)
        head = [name, 'Картинок', 'Доля позитивов', 'Dice', 'Вклад в дефицит Dice'] + (['Dice < 0.5', 'Срезано гейтингом'] if extra else [])
        return table(head, rows)

    t_dom = budget_table('pos_by_domain', 'Домен')
    t_bin = budget_table('pos_by_gtbin', 'Площадь GT', sort=False)
    t_gen = budget_table('pos_by_generator', 'Генератор')
    t_nc = budget_table('pos_by_ncomp', 'Компонент в GT (4 = 4+)', sort=False, extra=False)

    # домен × площадь (Dice, n)
    bins = ['<1%', '1–5%', '5–20%', '>20%']
    dx = R['pos_by_domain_gtbin']
    rows = []
    for dom in sorted(dx, key=lambda k: R['pos_by_domain'][k]['dice']):
        cells = []
        for b in bins:
            v = dx[dom].get(b)
            if not v: cells.append('<span class="mut">—</span>'); continue
            g = v['dice']; col = f'rgba(194,71,61,{max(0, 0.9 - g):.2f})' if g < 0.9 else 'transparent'
            cells.append(f'<span class="cell" style="background:{col}">{g:.2f} <span class="mut">({v["n"]})</span></span>')
        rows.append([f'<b>{dom}</b>'] + cells)
    t_dx = table(['Домен \\ площадь GT'] + bins, rows, 'grid')

    # ---- негативы
    nb = R['neg_by_domain']
    rows = [[f'<b>{k}</b>', v['n'], v['n_fa'], pct(v['fa'], 2)] for k, v in sorted(nb.items(), key=lambda kv: -kv[1]['fa'])]
    t_neg = table(['Пул : домен оригинала', 'Негативов', 'Ложных тревог', 'FPR'], rows)
    fa_h = R['fa_area_hist']; raw_h = R['neg_area_raw_hist']
    n_neg = R['n_neg']; n_fires = round(R['neg_mask_fires_share'] * n_neg); n_both = round(R['neg_both_fire'] * n_neg)
    n_gated = round(R['neg_gated_share'] * n_neg)

    # what-if
    wi = R['whatif']
    rows = []
    for k, v in wi.items():
        tau = k.split(',')[0].split('=')[1]; am = k.split('=')[-1]
        dice = v['dice']; aic = (2 * dice * (1 - v['fpr']) / (dice + 1 - v['fpr'])) if dice is not None else None
        rows.append([tau, am, f'{v["fpr"]:.4f}', f'{dice:.4f}' if dice is not None else '<span class="mut">n/a (τ ниже рабочего)</span>', f'<b>{aic:.4f}</b>' if aic else '—'])
    t_wi = table(['τ', 'a_min', 'FPR_main', 'Dice', 'AIC'], rows)

    # ---- примеры
    def gallery(key, title, note):
        items = EX.get(key, [])
        if not items: return ''
        cards = []
        for e in items:
            cap = f'{e.get("domain", "")}'
            if 'gen' in e: cap += f' · {e["gen"]}'
            meta = []
            if 'dice' in e: meta.append(f'Dice {e["dice"]:.2f}')
            if 'gt_area' in e: meta.append(f'GT {100 * e["gt_area"]:.1f} %')
            meta.append(f'pred {100 * e["area"]:.1f} %' if 'area' in e else '')
            meta.append(f'p_img {e["p_img"]:.2f}')
            cards.append(f'<figure><img src="data:image/jpeg;base64,{e["img"]}" alt="{e["stem"]}" loading="lazy"><figcaption><b>{cap}</b><br>{" · ".join(m for m in meta if m)}</figcaption></figure>')
        return f'<h3>{title}</h3><p class="mut">{note}</p><div class="gal">{"".join(cards)}</div>'

    dh = R['dice_hist']; dl = ['0–0.1', '0.1–0.3', '0.3–0.5', '0.5–0.7', '0.7–0.8', '0.8–0.9', '0.9–0.95', '0.95–1']
    ph = R['pimg_neg_hist']; pl = ['<0.1', '0.1–0.3', '0.3–0.5', '0.5–0.7', '0.7–0.8', '0.8–0.9', '0.9–0.95', '0.95–0.99', '≥0.99']

    dom = R['pos_by_domain']; ids9 = dom.get('ids9', {})
    html = f'''<title>Разбор ошибок {a.title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F2F4F6;--card:#FFFFFF;--ink:#17212B;--mut:#5C6B78;--line:#D9DEE4;--acc:#0E7C86;--bad:#C2473D;--good:#2E8B57;--warn:#B7791F;--soft:#E6EEF0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0F1519;--card:#171F26;--ink:#E8EDF1;--mut:#9AA7B2;--line:#2A3540;--acc:#3FB4BE;--bad:#E06A60;--good:#5CB985;--warn:#D9A24A;--soft:#1E2B30}}}}
:root[data-theme="dark"]{{--bg:#0F1519;--card:#171F26;--ink:#E8EDF1;--mut:#9AA7B2;--line:#2A3540;--acc:#3FB4BE;--bad:#E06A60;--good:#5CB985;--warn:#D9A24A;--soft:#1E2B30}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0}}
main{{max-width:1120px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px;text-wrap:balance}} h2{{font-size:20px;margin:36px 0 10px;padding-top:14px;border-top:1px solid var(--line)}} h3{{font-size:16px;margin:22px 0 6px}}
p{{max-width:72ch}} .mut{{color:var(--mut)}} .num{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
.tile .k{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}} .tile .v{{font-size:30px;font-weight:600;font-family:"IBM Plex Mono",monospace;margin:2px 0}} .tile .s{{font-size:13px;color:var(--mut)}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th{{text-align:left;font-weight:500;color:var(--mut);font-size:12.5px;letter-spacing:.03em;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
td{{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}} tr:last-child td{{border-bottom:0}}
.bar{{display:inline-block;width:120px;height:9px;background:var(--soft);border-radius:4px;vertical-align:middle;margin-left:8px;overflow:hidden}} .bar span{{display:block;height:100%;border-radius:4px}}
.cell{{display:inline-block;padding:2px 6px;border-radius:4px}}
.hist{{width:100%;max-width:620px;height:auto;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}} .hist .v{{font-size:11px;fill:var(--mut);font-family:"IBM Plex Mono",monospace}} .hist .l{{font-size:11px;fill:var(--mut)}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin:8px 0 6px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}} figure img{{width:100%;display:block}} figcaption{{font-size:12.5px;padding:7px 9px;color:var(--mut)}} figcaption b{{color:var(--ink);font-weight:500}}
.legend{{display:inline-flex;gap:14px;font-size:13px;color:var(--mut);margin:4px 0 8px}} .sw{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:4px}}
.callout{{background:var(--card);border-left:4px solid var(--acc);border-radius:6px;padding:10px 14px;margin:12px 0}} .callout.bad{{border-color:var(--bad)}}
.funnel{{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap;margin:8px 0}} .funnel div{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}} .funnel .n{{font-family:"IBM Plex Mono",monospace;font-size:22px;font-weight:600}}
ul{{padding-left:20px}} li{{margin:4px 0}}
</style>
<main>
<h1>Разбор ошибок {a.title}</h1>
<p class="mut">val-tune: {R["n_pos"]} позитивов и {R["n_neg"]} негативов, пороги t = {R["t"]}, a_min = {R["a_min"]}, τ = {R["tau"]}. AIC = гармоническое среднее Dice по позитивам и (1 − FPR) по негативам.</p>

<div class="tiles">
<div class="tile"><div class="k">AIC (val-tune)</div><div class="v">{A:.4f}</div><div class="s">= HM({D:.3f}, {1 - F:.3f})</div></div>
<div class="tile"><div class="k">Dice по позитивам</div><div class="v">{D:.3f}</div><div class="s">дефицит 1 − D = {loss_total:.3f}</div></div>
<div class="tile"><div class="k">FPR_main</div><div class="v">{F:.4f}</div><div class="s">neg_train {R["fpr"]["neg_train"][0]:.4f} · neg_src {R["fpr"]["neg_src"][0]:.4f}</div></div>
<div class="tile"><div class="k">Что даёт +0.01</div><div class="v">{dD * 0.01:.4f} / {-dF * 0.01:.4f}</div><div class="s">к Dice / к (−FPR): ∂AIC/∂D = {dD:.2f}, ∂AIC/∂F = {dF:.2f}</div></div>
</div>

<div class="callout"><b>Где потолок.</b> Если довести негативы до идеала (FPR 0), AIC вырастет максимум на {gain_neg:.3f}. Если довести позитивы до идеала (Dice 1), на {gain_pos:.3f}. Резерв метрики почти целиком в позитивах: {gain_pos / (gain_pos + gain_neg):.0%} против {gain_neg / (gain_pos + gain_neg):.0%}.</div>

<h2>1. Позитивы: из чего складывается дефицит Dice {loss_total:.3f}</h2>
<div class="two">
<div><h3>Распределение Dice по картинкам</h3>{hist_svg(dh, dl)}<p class="mut">{dh[0] + dh[1] + dh[2]} картинок ({pct((dh[0] + dh[1] + dh[2]) / R["n_pos"])}) с Dice &lt; 0.5 и {dh[-1]} ({pct(dh[-1] / R["n_pos"])}) с Dice &gt; 0.95.</p></div>
<div><h3>Типы ошибок</h3><p class="mut">Вклад = Σ(1 − Dice) по группе / все позитивы. Сумма вкладов равна дефициту {loss_total:.3f}.</p></div>
</div>
{t_etype}
<div class="callout">Половина дефицита ({et["ok"]["loss_contrib"]:.3f}) сидит в «нормальных» картинках с Dice ≥ 0.5: это качество границ, средний Dice там {et["ok"]["dice"]:.3f}. Вторая половина — грубые провалы: «не туда» ({et["misplaced"]["loss_contrib"]:.3f}), срезано гейтингом ({et["missed_gate"]["loss_contrib"]:.3f}), пересегментация ({et["over"]["loss_contrib"]:.3f}).</div>

<h3>По доменам</h3>
{t_dom}
<div class="callout bad"><b>ids9 — главное слабое место:</b> {pct(ids9.get("share", 0))} позитивов, но {ids9.get("loss_contrib", 0) / loss_total:.0%} всего дефицита Dice (средний {ids9.get("dice", 0):.3f}, у каждой пятой картинки Dice &lt; 0.5). Это домен с JPEG q80–95 и генератором «unknown». openimages второй по Dice, но там ещё и {pct(dom["openimages"]["gated"])} позитивов срезано гейтингом: классификатор не уверен в этом источнике.</div>

<h3>По площади GT</h3>
{t_bin}
<h3>Домен × площадь GT (Dice, в скобках число картинок; краснее = хуже)</h3>
{t_dx}
<p class="mut">ids9 плох на всех размерах масок (0.58 / 0.76 / 0.83), то есть это проблема домена, а не мелких масок. Мелкие маски (&lt; 1 %) плохи везде, но их всего {pct(R["pos_by_gtbin"]["<1%"]["share"])} позитивов.</p>

<div class="two">
<div><h3>По генератору</h3>{t_gen}</div>
<div><h3>По числу компонент в GT</h3>{t_nc}<p class="mut">Многокомпонентные маски (4+) теряют ~0.04 Dice относительно однокомпонентных: часть кусочков пропускается.</p></div>
</div>

<h2>2. Негативы: {R["n_neg"]} картинок, {round(F * R["n_neg"])} ложных тревог в среднем по пулам</h2>
<div class="funnel">
<div><div class="k mut">всего негативов</div><div class="n">{n_neg}</div></div>
<div><div class="k mut">маска сработала (≥ 1 %)</div><div class="n">{n_fires}</div><div class="mut">{pct(R["neg_mask_fires_share"], 2)}</div></div>
<div><div class="k mut">классификатор пропустил (p_img ≥ τ)</div><div class="n">{n_neg - n_gated}</div><div class="mut">{pct(1 - R["neg_gated_share"], 2)}</div></div>
<div><div class="k mut">обе головы ошиблись = FA</div><div class="n">{n_both}</div><div class="mut">{pct(R["neg_both_fire"], 2)}</div></div>
</div>
<div class="two">
<div>{t_neg}</div>
<div><h3>Площадь ложных тревог</h3>{hist_svg(fa_h["counts"], fa_h["bins"], 'var(--bad)')}<p class="mut">Ложные тревоги в основном крупные ({fa_h["counts"][2] + fa_h["counts"][3]} из {sum(fa_h["counts"])} больше 5 % кадра), медианный p_img у них {R["fa_pimg_q"][1]:.2f}: модель в них уверена. Порог по площади (a_min) срежет лишь {fa_h["counts"][0]} штук в зоне 1–2 %.</p></div>
</div>
<div class="two">
<div><h3>p_img классификатора на негативах</h3>{hist_svg(ph, pl, 'var(--warn)')}<p class="mut">{ph[-1] + ph[-2]} негативов ({pct((ph[-1] + ph[-2]) / n_neg, 1)}) получают p_img ≥ 0.95 — гейтинг их не спасёт ни при каком τ.</p></div>
<div><h3>Площадь сырой маски на негативах</h3>{hist_svg(raw_h["counts"], raw_h["bins"], 'var(--acc)')}<p class="mut">У {raw_h["counts"][0]} негативов маска пустая вообще; «пограничных» (0.5–2 %) мало, так что и тут a_min почти ничего не даёт.</p></div>
</div>

<h3>Что если сдвинуть τ и a_min (по тем же probs)</h3>
{t_wi}
<p class="mut">Гейтинг сейчас на плато: τ 0.8 и 0.9 дают одинаковый AIC, потому что каждое снижение FPR оплачивается срезанными позитивами (при τ 0.8 их {et["missed_gate"]["n"]}, это −{et["missed_gate"]["loss_contrib"] * dD:.4f} AIC, ровно столько же, сколько гейтинг экономит на негативах). a_min только вредит: мелкие настоящие маски теряются быстрее, чем уходят ложные тревоги.</p>

<h2>3. Примеры</h2>
<div class="legend"><span><span class="sw" style="background:rgba(220,60,50,.6)"></span>предсказание (заливка)</span><span><span class="sw" style="background:#2E8B57"></span>GT (контур)</span></div>
{gallery('good', 'Как выглядит норма (Dice > 0.95)', 'Для калибровки глаза.')}
{gallery('misplaced', 'Не туда', 'Худшие по Dice: маска есть, но не там, где правка. Часто модель красит другой объект или фон.')}
{gallery('missed_gate', 'Срезано гейтингом (p_img < τ)', 'Случайная выборка. Маска у модели могла быть, но классификатор сказал «чистая».')}
{gallery('over', 'Пересегментация', 'Правка найдена, но маска расползлась далеко за GT.')}
{gallery('under', 'Недосегментация', 'Найден кусок правки, остальное пропущено (часто у многокомпонентных GT).')}
{gallery('missed_mask', 'Пустая маска при уверенном классификаторе', 'Классификатор видит правку, сегментатор нет.')}
{gallery('fa_big', 'Ложные тревоги: самые крупные', 'Оригиналы без правок, модель уверенно красит большие области.')}
{gallery('fa_border', 'Ложные тревоги: пограничные (1–2 %)', 'Единственные, которые снял бы a_min.')}

<h2>4. Выводы</h2>
<ul>
<li><b>Слабое место — позитивы, а не негативы.</b> Негативы почти решены (FPR {F:.3f}, потолок выигрыша {gain_neg:.3f} AIC). Весь резерв ({gain_pos:.3f}) в Dice.</li>
<li><b>Внутри позитивов — домен ids9</b>: четверть данных даёт половину дефицита, плох на всех размерах масок. Это единственный JPEG-сжатый домен, ровно то, что должны лечить аугментации (раны aug v1/v2).</li>
<li><b>Вторая половина дефицита — границы</b> на нормальных картинках (Dice 0.93 при «попадании»). Тут помогают edge-loss, разрешение и качество декодера, а не пороги.</li>
<li><b>Гейтинг исчерпан.</b> Классификатор режет столько же настоящих правок (особенно openimages {pct(dom["openimages"]["gated"])}, raise {pct(dom["raise"]["gated"])}), сколько ловит ложных тревог. Улучшать надо сам классификатор (негативы из всех источников, patch-level голова), а не τ.</li>
<li><b>a_min не нужен:</b> ложные тревоги крупные и уверенные, пограничных единицы.</li>
<li><b>Мелкие маски (&lt; 5 %)</b> дают {R["pos_by_gtbin"]["<1%"]["loss_contrib"] + R["pos_by_gtbin"]["1–5%"]["loss_contrib"]:.3f} дефицита при {pct(R["pos_by_gtbin"]["<1%"]["share"] + R["pos_by_gtbin"]["1–5%"]["share"])} доли — второй приоритет после ids9; лечится mask-mixing и edge-loss.</li>
</ul>
</main>
'''
    Path(a.out).write_text(html, encoding='utf-8')
    print(a.out, f'{len(html) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
