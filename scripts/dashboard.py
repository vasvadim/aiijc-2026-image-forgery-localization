#!/usr/bin/env python
"""dashboard.py — HTML-дашборд обучения из log.jsonl (замена W&B, когда тот недоступен).

    python scripts/dashboard.py --run runs/_wandb_sync/vasilenkova/<run1> --run .../<run2> ... \
        --out runs/_dashboard/train_dashboard.html --title "Детектив: обучение"

На странице есть выбор рана: графики конкретного рана (loss, классификатор, разрезы по площади/доменам,
пустые прогнозы, градиент, время, таблица эпох, плитки) переключаются, сравнительные графики
(AIC, Dice, FPR по всем ранам) остаются общими, выбранный ран в них подсвечен. Первый --run выбран по
умолчанию; выбор запоминается в браузере. Старые флаги --main/--ref принимаются как синонимы --run.
Все графики — небольшие SVG с одной осью и ховер-подсказкой.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load(d: Path) -> dict:
    ep = [json.loads(l) for l in (d / 'log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    cfg = json.loads((d / 'config.json').read_text(encoding='utf-8')) if (d / 'config.json').exists() else {}
    thr = json.loads((d / 'threshold.json').read_text(encoding='utf-8')) if (d / 'threshold.json').exists() else None
    return dict(name=d.name, epochs=ep, config=cfg, threshold=thr, bestfile=(d / 'best.json').exists())


TEMPLATE = r"""<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--bg:#f6f6f3;--surface:#ffffff;--ink:#1c1d1f;--ink-2:#5a5b58;--ink-3:#8d8e89;--line:#e4e4df;--grid:#ececE7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--good:#1a7f37;--warn:#b26a00;--bad:#c62828;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#151617;--surface:#1e1f21;--ink:#ececea;--ink-2:#b9b9b3;--ink-3:#84857f;--line:#2e2f31;--grid:#292a2c;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#3da53d;--s7:#9085e9;--good:#4caf50;--warn:#e0a030;--bad:#ef6b6b;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#151617;--surface:#1e1f21;--ink:#ececea;--ink-2:#b9b9b3;--ink-3:#84857f;--line:#2e2f31;--grid:#292a2c;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#3da53d;--s7:#9085e9;--good:#4caf50;--warn:#e0a030;--bad:#ef6b6b;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:24px 20px 48px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:18px}
header h1{font-size:20px;font-weight:600;margin:0;text-wrap:balance}
header .sub{color:var(--ink-2);flex-basis:100%}
select{font:inherit;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);max-width:100%}
select:focus{outline:2px solid var(--accent);outline-offset:1px}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;letter-spacing:.02em;font-weight:500}
.pill.run{background:color-mix(in oklab,var(--good) 15%,transparent);color:var(--good)}
.pill.done{background:color-mix(in oklab,var(--ink-3) 18%,transparent);color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3);margin-bottom:6px}
.tile .v{font-size:24px;font-weight:500;line-height:1.1}
.tile .d{font-size:12px;color:var(--ink-2);margin-top:4px}
h2{font-size:15px;font-weight:600;margin:26px 0 10px}
h2 .hint{font-weight:400;color:var(--ink-3);font-size:12px;margin-left:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px 8px;position:relative}
.card h3{margin:0 0 2px;font-size:13px;font-weight:600}
.card .note{font-size:12px;color:var(--ink-3);margin-bottom:6px;min-height:16px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink-2);margin:4px 0 2px}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.legend i.dash{border-top-style:dashed}
.legend b{font-weight:600;color:var(--ink)}
svg{width:100%;height:auto;display:block;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 8px;border-radius:4px;font-size:12px;line-height:1.35;white-space:nowrap;display:none;z-index:2}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink-3);font-weight:500;text-transform:uppercase;font-size:11px;letter-spacing:.05em;position:sticky;top:0;background:var(--surface)}
td:first-child,th:first-child{text-align:left}
tr.best td{font-weight:600;color:var(--accent)}
.foot{color:var(--ink-3);font-size:12px;margin-top:22px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
<header>
  <h1 id="title"></h1>
  <label for="runsel" class="mono" style="font-size:12px;color:var(--ink-3)">ран</label>
  <select id="runsel" aria-label="Выбор рана"></select>
  <span id="status" class="pill"></span>
  <span class="sub" id="subtitle"></span>
</header>
<div class="tiles" id="tiles"></div>
<h2>Сравнение ранов<span class="hint">все раны; выбранный — толстой линией</span></h2>
<div class="grid" id="charts_cmp"></div>
<h2>Выбранный ран<span class="hint" id="runname"></span></h2>
<div class="grid" id="charts_run"></div>
<h2>Эпохи выбранного рана</h2>
<div class="tbl"><table id="tbl"></table></div>
<p class="foot" id="foot"></p>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const RUNS = DATA.runs.filter(r => r.epochs.length);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f4 = x => (x==null||Number.isNaN(x)) ? '—' : (+x).toFixed(4);
const f3 = x => (x==null||Number.isNaN(x)) ? '—' : (+x).toFixed(3);
const get = (r, path) => path.split('.').reduce((o,k)=> (o==null?undefined:o[k]), r);
const short = n => n.replace(/^vad_/,'').replace(/^\d{8}_/,'');
document.getElementById('title').textContent = DATA.title;
if (!RUNS.length) { document.getElementById('subtitle').textContent = 'ни у одного рана нет завершённых эпох — обновлено ' + DATA.generated; throw new Error('no epochs'); }
const S1=css('--s1'), S2=css('--s2'), S3=css('--s3'), S4=css('--s4'), S5=css('--s5'), S6=css('--s6'), S7=css('--s7');
const RUNCOL = [S1,S2,S4,S7,S5,S6,S3];
RUNS.forEach((r,i)=>{ r.color = RUNCOL[i % RUNCOL.length]; r.best = r.epochs.reduce((a,b)=> b.proxy.aic>a.proxy.aic?b:a, r.epochs[0]); r.last = r.epochs[r.epochs.length-1]; });
const pts = (arr, fn) => arr.map(r=>[r.epoch, fn(r)]);

// ---- generic line chart
function chart(el, title, note, series, opts={}) {
  const W=560, H=220, pl=44, pr=12, pt=10, pb=26;
  const all = series.flatMap(s=>s.pts).filter(p=>p[1]!=null && !Number.isNaN(p[1]));
  if (!all.length) { el.innerHTML = `<h3>${title}</h3><div class="note">нет данных</div>`; return; }
  const xs = all.map(p=>p[0]), ys = all.map(p=>p[1]);
  const xmin = 1, xmax = Math.max(2, ...xs);
  let ymin = opts.ymin ?? Math.min(...ys), ymax = opts.ymax ?? Math.max(...ys);
  if (opts.ymin==null||opts.ymax==null){ const pad=(ymax-ymin||0.1)*0.08; if(opts.ymin==null) ymin-=pad; if(opts.ymax==null) ymax+=pad; }
  const X = x => pl + (x-xmin)/(xmax-xmin||1)*(W-pl-pr);
  const Y = y => pt + (1-(y-ymin)/(ymax-ymin||1))*(H-pt-pb);
  const ticksY = 4, ty = Array.from({length:ticksY+1},(_,i)=>ymin+(ymax-ymin)*i/ticksY);
  const txStep = xmax>12 ? Math.ceil(xmax/10) : 1;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">`;
  ty.forEach(v=>{ g+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${pl-6}" y="${Y(v)+4}" text-anchor="end" font-size="10" fill="var(--ink-3)" font-family="IBM Plex Mono,monospace">${v.toFixed(opts.dec??2)}</text>`; });
  for (let x=xmin;x<=xmax;x+=txStep) g+=`<text x="${X(x)}" y="${H-8}" text-anchor="middle" font-size="10" fill="var(--ink-3)" font-family="IBM Plex Mono,monospace">${x}</text>`;
  g+=`<line x1="${pl}" x2="${W-pr}" y1="${H-pb}" y2="${H-pb}" stroke="var(--line)"/>`;
  if (opts.hline!=null && opts.hline>=ymin && opts.hline<=ymax) g+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(opts.hline)}" y2="${Y(opts.hline)}" stroke="var(--ink-3)" stroke-dasharray="2 4"/><text x="${W-pr}" y="${Y(opts.hline)-4}" text-anchor="end" font-size="10" fill="var(--ink-3)">${opts.hlabel||''}</text>`;
  series.forEach(s=>{ const p=s.pts.filter(q=>q[1]!=null&&!Number.isNaN(q[1])); if(!p.length) return;
    const sw = s.bold ? 3.5 : 2;
    g+=`<polyline fill="none" stroke="${s.color}" stroke-width="${sw}" ${s.dash?'stroke-dasharray="5 4"':''} ${s.faint?'opacity="0.55"':''} stroke-linejoin="round" points="${p.map(q=>X(q[0])+','+Y(q[1])).join(' ')}"/>`;
    const e=p[p.length-1]; g+=`<circle cx="${X(e[0])}" cy="${Y(e[1])}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
    if (s.best){ const b=p.find(q=>q[0]===s.best); if(b) g+=`<circle cx="${X(b[0])}" cy="${Y(b[1])}" r="5.5" fill="none" stroke="${s.color}" stroke-width="2"/>`; }
  });
  g+=`<line class="cross" x1="0" x2="0" y1="${pt}" y2="${H-pb}" stroke="var(--ink-3)" stroke-width="1" style="display:none"/></svg>`;
  const legend = series.length>1 ? `<div class="legend">${series.map(s=>`<span><i class="${s.dash?'dash':''}" style="border-color:${s.color}"></i>${s.bold?'<b>'+s.name+'</b>':s.name}</span>`).join('')}</div>` : '';
  el.innerHTML = `<h3>${title}</h3><div class="note">${note||''}</div>${g}${legend}<div class="tip"></div>`;
  const svg = el.querySelector('svg'), tip = el.querySelector('.tip'), cross = el.querySelector('.cross');
  svg.addEventListener('mousemove', ev=>{ const r=svg.getBoundingClientRect(); const x = (ev.clientX-r.left)/r.width*W; const ep = Math.round(xmin+(x-pl)/(W-pl-pr)*(xmax-xmin)); if(ep<xmin||ep>xmax){tip.style.display='none';cross.style.display='none';return;}
    cross.setAttribute('x1',X(ep)); cross.setAttribute('x2',X(ep)); cross.style.display='';
    tip.innerHTML = `<b>эпоха ${ep}</b><br>` + series.map(s=>{const q=s.pts.find(p=>p[0]===ep); return q&&q[1]!=null?`${s.name}: ${(+q[1]).toFixed(4)}`:null;}).filter(Boolean).join('<br>');
    tip.style.display='block'; const cr=el.getBoundingClientRect(); tip.style.left=Math.min(ev.clientX-cr.left+12, cr.width-tip.offsetWidth-8)+'px'; tip.style.top=(ev.clientY-cr.top-10)+'px'; });
  svg.addEventListener('mouseleave', ()=>{tip.style.display='none';cross.style.display='none';});
}
const addTo = (holder, title, note, series, opts) => { const d=document.createElement('div'); d.className='card'; holder.appendChild(d); chart(d,title,note,series,opts); };

// ---- run selector
const sel = document.getElementById('runsel');
RUNS.forEach((r,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=`${short(r.name)} · ${r.epochs.length} эп. · best ${f4(r.best.proxy.aic)}`; sel.appendChild(o); });
let selIdx = 0;
try { const saved = localStorage.getItem('aic_dash_run'); const j = RUNS.findIndex(r=>r.name===saved); if (j>=0) selIdx=j; } catch(e) {}
sel.value = selIdx;
sel.addEventListener('change', ()=>{ selIdx = +sel.value; try { localStorage.setItem('aic_dash_run', RUNS[selIdx].name); } catch(e) {} render(); });

// ---- comparison charts (all runs, selected bold)
function renderCmp() {
  const holder = document.getElementById('charts_cmp'); holder.innerHTML = '';
  const runSeries = fn => RUNS.map((r,i)=>({name:short(r.name), color:r.color, bold:i===selIdx, faint:i!==selIdx, pts:pts(r.epochs, fn), best: i===selIdx ? r.best.epoch : null}));
  addTo(holder, 'Proxy AIC', 'Dice per image + FPR_main на proxy-подмножестве, t = 0.5; у ранов с головой — с гейтингом τ = 0.5; кольцо — лучшая эпоха выбранного', runSeries(r=>r.proxy.aic), {dec:3});
  addTo(holder, 'Dice_pos', 'среднее по позитивам proxy', runSeries(r=>r.proxy.dice_pos), {dec:3});
  addTo(holder, 'FPR_main', 'среднее FPR по пулам neg_train и neg_src при t = 0.5 (с гейтингом, если есть голова)', runSeries(r=>r.proxy.fpr_main), {ymin:0, dec:2});
  addTo(holder, 'val loss', 'сравнимо только при одинаковом loss и одинаковом наборе негативов', runSeries(r=>r.val_loss), {dec:3});
  if (RUNS.some(r=>r.epochs.some(e=>e.proxy.dice_by_area))) {
    addTo(holder, 'Dice на мелких масках (< 1 % кадра)', 'самый чувствительный к разрешению разрез', runSeries(r=>get(r,'proxy.dice_by_area.lt1')), {ymin:0, ymax:1, dec:2});
    addTo(holder, 'Dice на масках 1–5 % кадра', '', runSeries(r=>get(r,'proxy.dice_by_area.1-5')), {ymin:0, ymax:1, dec:2});
  }
}

// ---- per-run charts
function renderRun() {
  const M = RUNS[selIdx], E = M.epochs, c = M.config, last = M.last, best = M.best;
  document.getElementById('runname').textContent = M.name;
  const st = document.getElementById('status'); st.textContent = M.bestfile ? 'завершён' : 'обучается'; st.className = 'pill ' + (M.bestfile?'done':'run');
  const sz = Array.isArray(c.img_size) ? c.img_size.join('×') : (c.img_size + '²');
  document.getElementById('subtitle').textContent = `${c.encoder||''} · ${sz} · декодер ${c.decoder_channels ? c.decoder_channels.join(',') : 'smp'} · ${c.loss} · lr ${c.lr}${c.scheduler&&c.scheduler!=='none'?' '+c.scheduler:''} · batch ${c.batch_size}×${c.world_size||1} GPU · aux ${c.aux_head?'да':'нет'} · neg-src ${c.neg_src??0} · patience ${c.patience??'—'} · валидация ${c.version||'v1'}`;
  const since = last.epoch - best.epoch;
  const tiles = [
    ['эпоха', `${last.epoch} / ${c.epochs}`, `${Math.round(last.time_s/60)} мин/эпоху`],
    ['лучший proxy AIC', f4(best.proxy.aic), `эпоха ${best.epoch}` + (best.full?` · full ${f4(best.full.aic)}`:'') + (best.proxy.aic_ungated!=null?` · ungated ${f4(best.proxy.aic_ungated)}`:'')],
    ['без улучшения', `${since} эп.`, `стоп при ${c.patience??'—'}`],
    ['Dice_pos (лучшая эп.)', f4(best.proxy.dice_pos), 'proxy, t = 0.5'],
    ['FPR_main (лучшая эп.)', f4(best.proxy.fpr_main), `train ${f3(best.proxy.fpr.neg_train)} · src ${f3(best.proxy.fpr.neg_src)}`],
    ['val loss (посл.)', f4(last.val_loss), `train ${f4(last.train_loss)} · разрыв ${f3(last.val_loss-last.train_loss)}`],
  ];
  if (last.proxy.cls) tiles.push(['классификатор (посл.)', f3(last.proxy.cls.acc), `tpr ${f3(last.proxy.cls.tpr)} · tnr ${f3(last.proxy.cls.tnr)}`]);
  if (M.threshold) tiles.push(['подбор порога', `${M.threshold.t} / ${M.threshold.a_min}` + (M.threshold.tau!=null?` / τ ${M.threshold.tau}`:''), `val-tune AIC ${f4(M.threshold.aic)}`]);
  document.getElementById('tiles').innerHTML = tiles.map(t=>`<div class="tile"><div class="k">${t[0]}</div><div class="v mono">${t[1]}</div><div class="d">${t[2]}</div></div>`).join('');

  const holder = document.getElementById('charts_run'); holder.innerHTML = '';
  const col = M.color;
  addTo(holder, 'Proxy AIC и full', 'сплошная — proxy на тензорах; пунктир — официальная метрика на PNG (t = 0.5)', [
    {name:'proxy', color:col, pts:pts(E,r=>r.proxy.aic), best:best.epoch}, {name:'full', color:col, dash:true, pts:pts(E,r=>r.full?r.full.aic:null)},
    ...(E.some(r=>r.proxy.aic_ungated!=null)?[{name:'ungated', color:S3, dash:true, pts:pts(E,r=>r.proxy.aic_ungated)}]:[])], {dec:3});
  addTo(holder, 'Loss', 'train vs val (' + (c.loss||'') + '); рост val при падающем train — переобучение', [
    {name:'train', color:S3, pts:pts(E,r=>r.train_loss)}, {name:'val-tune', color:col, pts:pts(E,r=>r.val_loss)}], {dec:3});
  if (E.some(r=>r.proxy.cls)) addTo(holder, 'Image-level голова', 'accuracy при p ≥ 0.5; TPR/TNR при τ = 0.5', [
    {name:'acc', color:col, pts:pts(E,r=>r.proxy.cls?r.proxy.cls.acc:null)}, {name:'tpr', color:S3, pts:pts(E,r=>r.proxy.cls?r.proxy.cls.tpr:null)},
    {name:'tnr', color:S4, pts:pts(E,r=>r.proxy.cls?r.proxy.cls.tnr:null)}], {ymin:0, ymax:1, dec:2});
  addTo(holder, 'FPR по пулам', 'доля чистых кадров с маской ≥ 1 % при t = 0.5', [
    {name:'neg_train', color:col, pts:pts(E,r=>r.proxy.fpr.neg_train)}, {name:'neg_src', color:S3, pts:pts(E,r=>r.proxy.fpr.neg_src)}], {ymin:0, dec:2});
  if (E.some(r=>r.proxy.dice_by_area)) addTo(holder, 'Dice по площади GT', 'доля кадра, занятая манипуляцией', [
    {name:'< 1 %', color:col, pts:pts(E,r=>get(r,'proxy.dice_by_area.lt1'))}, {name:'1–5 %', color:S3, pts:pts(E,r=>get(r,'proxy.dice_by_area.1-5'))},
    {name:'5–20 %', color:S4, pts:pts(E,r=>get(r,'proxy.dice_by_area.5-20'))}, {name:'> 20 %', color:S7, pts:pts(E,r=>get(r,'proxy.dice_by_area.gt20'))}], {ymin:0, ymax:1, dec:2});
  if (E.some(r=>r.proxy.dice_by_domain)) { const doms = Object.keys(last.proxy.dice_by_domain||{}); const cols=[col,S3,S4,S5,S6,S7,S2];
    addTo(holder, 'Dice по доменам', 'источник картинки (из имени файла)', doms.map((d,i)=>({name:d, color:cols[i%cols.length], pts:pts(E,r=>get(r,'proxy.dice_by_domain.'+d))})), {ymin:0, ymax:1, dec:2}); }
  if (E.some(r=>r.proxy.empty_pred_on_pos!=null)) addTo(holder, 'Пустые прогнозы на позитивах и площадь на негативах', 'доля позитивов с пустой маской; медиана площади маски на чистых кадрах (порог тревоги 1 %)', [
    {name:'пустых на позитивах', color:col, pts:pts(E,r=>r.proxy.empty_pred_on_pos)}, {name:'медиана площади на негативах', color:S3, pts:pts(E,r=>r.proxy.neg_pred_area_median)}], {ymin:0, hline:0.01, hlabel:'1 %', dec:3});
  if (E.some(r=>r.grad_norm!=null)) addTo(holder, 'Норма градиента и lr', 'средняя по эпохе L2-норма градиента; lr — пунктиром (×1e4)', [
    {name:'grad norm', color:col, pts:pts(E,r=>r.grad_norm)}, {name:'lr ×1e4', color:S4, dash:true, pts:pts(E,r=>r.lr!=null?r.lr*1e4:null)}], {dec:2});
  addTo(holder, 'Время эпохи, с', 'полный цикл: train + валидация + full-метрика', [{name:'сек', color:col, pts:pts(E,r=>r.time_s)}], {ymin:0, dec:0});

  const cols = [['эпоха',r=>r.epoch,0],['train',r=>r.train_loss,4],['val',r=>r.val_loss,4],['Dice',r=>r.proxy.dice_pos,4],['FPR train',r=>r.proxy.fpr.neg_train,3],['FPR src',r=>r.proxy.fpr.neg_src,3],['FPR main',r=>r.proxy.fpr_main,4],['proxy AIC',r=>r.proxy.aic,4],['ungated',r=>r.proxy.aic_ungated,4],['full AIC',r=>r.full?r.full.aic:null,4],['cls acc',r=>r.proxy.cls?r.proxy.cls.acc:null,3],['пустых',r=>r.proxy.empty_pred_on_pos,3],['grad',r=>r.grad_norm,2],['lr',r=>r.lr,6],['мем GB',r=>r.gpu_max_mem_gb,1],['с',r=>r.time_s,0]];
  document.getElementById('tbl').innerHTML = '<thead><tr>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr></thead><tbody>'+E.map(r=>`<tr class="${r.epoch===best.epoch?'best':''}">`+cols.map(c=>{const v=c[1](r); return `<td class="mono">${v==null?'—':(+v).toFixed(c[2])}</td>`;}).join('')+'</tr>').join('')+'</tbody>';
  document.getElementById('foot').textContent = `Обновлено ${DATA.generated}. Источник: log.jsonl из ~/aic2026/runs/vasilenkova/<run>/ на кластере. proxy = метрика на тензорах в рабочем разрешении на фиксированном подмножестве val-tune; full = metric.py на PNG в исходном размере.`;
}
function render(){ renderCmp(); renderRun(); }
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', default=[], help='папка рана с log.jsonl; можно несколько, первый выбран по умолчанию')
    ap.add_argument('--main', default=None, help='синоним первого --run')
    ap.add_argument('--ref', action='append', default=[], help='синоним --run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Детектив: обучение')
    args = ap.parse_args()
    paths = ([args.main] if args.main else []) + args.run + args.ref
    runs = [load(Path(p)) for p in paths if (Path(p) / 'log.jsonl').exists()]
    data = dict(runs=runs, title=args.title, generated=datetime.now().strftime('%Y-%m-%d %H:%M'))
    html = TEMPLATE.replace('__TITLE__', args.title).replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f'{out} ({len(runs)} runs: ' + ', '.join(f"{r['name']}={len(r['epochs'])}ep" for r in runs) + ')')


if __name__ == '__main__':
    main()
