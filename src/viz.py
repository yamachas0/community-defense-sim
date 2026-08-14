"""ラン結果を1枚の自己完結HTMLに書き出す（所有マップの時系列＋KPI折れ線＋発話）。

外部CDN・外部データに依存しない（フォントのみ Google Fonts、無くても崩れない）。
simulations/<run>/<run>.html として保存され、そのファイル単体で配布・提出できる。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

ACCENT = "#10B981"


def _payload(sim) -> Dict[str, Any]:
    parcels = [{"pid": p.pid, "x": p.x, "y": p.y, "block": p.block, "use": p.use,
                "value": p.assessed_value} for p in sim.ledger.parcels.values()]
    agents = [{"id": a.agent_id, "name": a.name, "role": a.role} for a in sim.agents]
    def _kind(e: Dict[str, Any]) -> str:
        o = e.get("outcome")
        if isinstance(o, dict):
            return str(o.get("kind", ""))
        return str(o or "")

    events = [{"step": e["step"], "id": e["agent_id"], "name": e.get("name", ""),
               "role": e["role"], "action": e.get("action_type", ""),
               "target": e.get("target", ""), "amount": e.get("amount", 0),
               "ok": not _kind(e).endswith(("_rejected", "invalid_action", "parse_fail")),
               "outcome": _kind(e),
               "utterance": e.get("utterance", ""),
               "reasoning": e.get("reasoning", "")} for e in sim.events]
    utter = getattr(sim, "classified", None) or sim.all_utterances
    return {
        "meta": sim.summary,
        "parcels": sorted(parcels, key=lambda q: (q["y"], q["x"])),
        "frames": sim.owner_frames,
        "kpi": sim.kpi_rows,
        "cognition": getattr(sim, "cognition", []),
        "utterances": utter,
        "events": events,
        "agents": agents,
        "acquirers": sim.acquirer_ids,
        "municipality": sim.municipality_id,
        "ordinances": sim.ledger.ordinances,
        "publications": sim.ledger.publications,
    }


def render_report(sim, path: str, folder: str) -> str:
    data = _payload(sim)
    html = _TEMPLATE.replace("__TITLE__", folder).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>静かな占領 — __TITLE__</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
:root{--black:#0A0A0A;--accent:#10B981;--accent-bg:#ECFDF5;--g50:#F9FAFB;--g100:#F3F4F6;
 --g200:#E5E7EB;--g300:#D1D5DB;--g500:#6B7280;--g600:#4B5563;--g700:#374151;--g900:#111827;
 --warn:#B45309;--warn-bg:#FFFBEB;--font:'Noto Sans JP','Hiragino Kaku Gothic ProN','Meiryo',system-ui,sans-serif;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--g900);font-family:var(--font);font-size:15px;line-height:1.75;}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 80px;}
header{background:var(--black);color:#fff;padding:28px 0 24px;margin-bottom:28px;}
header .wrap{padding-bottom:0;}
.eyebrow{font-size:11px;letter-spacing:.18em;color:var(--accent);font-weight:700;margin:0 0 8px;}
h1{font-size:23px;margin:0 0 10px;font-weight:900;line-height:1.4;}
.meta{font-size:12.5px;color:var(--g300);margin:0;}
.meta b{color:#fff;font-weight:500;}
h2{font-size:18px;font-weight:900;margin:44px 0 16px;padding:8px 0 8px 13px;
 border-left:5px solid var(--accent);background:var(--g50);}
h3{font-size:15px;font-weight:700;margin:24px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--black);}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:0 0 18px;}
.card{border:1px solid var(--g200);border-left:4px solid var(--accent);padding:11px 13px;}
.card .k{font-size:11.5px;color:var(--g500);font-weight:700;letter-spacing:.04em;}
.card .v{font-size:23px;font-weight:900;line-height:1.25;}
.card .s{font-size:11.5px;color:var(--g600);}
.panel{border:1px solid var(--g200);padding:14px;margin:0 0 18px;background:#fff;}
.ctrl{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 12px;}
.ctrl input[type=range]{flex:1;min-width:200px;accent-color:var(--accent);}
button{font-family:var(--font);font-weight:700;font-size:13px;border:2px solid var(--black);
 background:#fff;padding:6px 14px;cursor:pointer;}
button:hover{background:var(--black);color:#fff;}
.stepnum{font-weight:900;font-size:16px;min-width:112px;}
.mapwrap{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;}
canvas{border:1px solid var(--g200);max-width:100%;}
.legend{font-size:12px;line-height:2;color:var(--g700);}
.sw{display:inline-block;width:13px;height:13px;border:1px solid var(--g300);
 vertical-align:-2px;margin-right:6px;}
table{width:100%;border-collapse:collapse;font-size:12.5px;line-height:1.6;}
th{background:var(--black);color:#fff;text-align:left;padding:7px 8px;font-weight:700;white-space:nowrap;}
td{border-bottom:1px solid var(--g200);padding:6px 8px;vertical-align:top;}
tr:nth-child(even) td{background:var(--g50);}
.tag{display:inline-block;font-size:10.5px;font-weight:900;padding:1px 6px;border-radius:2px;white-space:nowrap;}
.t-ok{background:var(--accent-bg);color:#059669;border:1px solid var(--accent);}
.t-ng{background:var(--warn-bg);color:var(--warn);border:1px solid var(--warn);}
.utt{border-left:3px solid var(--g300);padding:4px 0 4px 10px;margin:0 0 8px;font-size:13px;}
.utt b{font-size:12px;color:var(--g600);}
.utt.f-their{border-left-color:var(--warn);}
.utt.f-our{border-left-color:var(--accent);}
.small{font-size:12px;color:var(--g500);}
.scroll{max-height:340px;overflow:auto;border:1px solid var(--g200);}
.note{background:var(--warn-bg);border-left:4px solid var(--warn);padding:11px 13px;
 margin:0 0 16px;font-size:13px;color:var(--g700);}
.chartbox{border:1px solid var(--g200);padding:8px;margin:0 0 14px;}
@media(max-width:640px){.stepnum{min-width:88px;}h1{font-size:19px;}}
</style></head><body>
<header><div class="wrap">
<p class="eyebrow">QUIET ACQUISITION — SIMULATION RESULT</p>
<h1>静かな占領 ── AIが土地を買う街</h1>
<p class="meta" id="hdrmeta"></p>
</div></header>
<div class="wrap">
<div class="cards" id="cards"></div>
<div class="note" id="verdict"></div>

<h2>1. 所有マップの時系列</h2>
<div class="panel">
 <div class="ctrl">
  <button id="play">▶ 再生</button>
  <span class="stepnum" id="stepnum"></span>
  <input type="range" id="slider" min="0" value="0">
 </div>
 <div class="mapwrap">
  <canvas id="map" width="560" height="420"></canvas>
  <div class="legend" id="legend"></div>
 </div>
 <div id="stepinfo" class="small" style="margin-top:10px;"></div>
</div>

<h2>2. KPI の推移</h2>
<div class="chartbox"><svg id="ch1" viewBox="0 0 900 260" style="width:100%;height:auto;"></svg></div>
<div class="chartbox"><svg id="ch2" viewBox="0 0 900 260" style="width:100%;height:auto;"></svg></div>

<h2>3. その月の声</h2>
<div class="panel"><div id="utts" class="scroll" style="padding:10px;"></div></div>

<h2>4. その月の行動ログ</h2>
<div class="panel"><div class="scroll"><table id="evt"><thead><tr>
<th>主体</th><th>行動</th><th>対象</th><th>金額</th><th>記帳</th><th>理由</th></tr></thead>
<tbody></tbody></table></div>
<p class="small">記帳が <span class="tag t-ng">不成立</span> の行は、エージェントが選んだが世界の帳簿上は成立しなかった行動（存在しない区画IDなど）。<b>コードは補正していない</b>。</p></div>

<h2>5. 出来事</h2>
<div class="panel" id="events"></div>
</div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s==null?"":s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const nameOf = {}; D.agents.forEach(a=>nameOf[a.id]=a.name);
const roleOf = {}; D.agents.forEach(a=>roleOf[a.id]=a.role);
const ACQ = new Set(D.acquirers);
const nSteps = D.meta.steps;
const pct = v => (v==null?"—":(v*100).toFixed(1)+"%");

/* ---------- header + cards ---------- */
const m = D.meta;
$("#hdrmeta").innerHTML = `<b>${esc(m.run_name)}</b> ／ ${esc(m.provider)} · ${esc(m.model)} ／ `
 + `${m.steps}か月 ／ 区画${m.parcels} ／ エージェント`
 + Object.entries(m.agents).map(([k,v])=>`${esc(k)}${v}`).join(" ")
 + ` ／ 所要${m.elapsed_sec}s`;

const k = m.kpi;
const cards = [
 ["① 所有集中度（買い手シェア）", pct(k.final_acquirer_share), "HHI "+(k.final_hhi??0).toFixed(3)],
 ["② 認知転相率（累積）", pct(k.cognition_shift_final), "「あの主体の街」と語る発話の比率"],
 ["③ 売却カスケード", (k.cascade&&k.cascade.induced!=null?k.cascade.induced:"—")+"件",
  "最大連鎖 "+((k.cascade&&k.cascade.max_chain)||0)],
 ["④ 手遅れ度（規制発動時シェア）", k.late_index?pct(k.late_index.share_at_enactment):"規制なし",
  k.late_index?`第${k.late_index.step}月「${esc(k.late_index.title)}」`:"最後まで規制は発動しなかった"],
 ["⑤ 検知ラグ（初報道時シェア）", k.detection_lag?pct(k.detection_lag.share_at_first_report):"報道なし",
  k.detection_lag?`第${k.detection_lag.step}月`:"買収は報じられなかった"],
 ["⑥ 生活KPI", pct(k.business_survival), "事業者残存率／住民転出率 "+pct(k.resident_outflow)],
];
$("#cards").innerHTML = cards.map(c=>`<div class="card"><div class="k">${c[0]}</div>
 <div class="v">${c[1]}</div><div class="s">${c[2]}</div></div>`).join("");

const lag = k.detection_lag ? k.detection_lag.share_at_first_report : null;
const late = k.late_index ? k.late_index.share_at_enactment : null;
$("#verdict").innerHTML = "<b>読み方</b>：④手遅れ度と⑤検知ラグが高いほど「気づいた時にはもう遅い」が起きている。"
 + (lag!=null?` このランでは最初に買収が報じられた時点で既に <b>${pct(lag)}</b> が取得済み。`:" このランでは買収は最後まで報じられなかった。")
 + (late!=null?` 規制が動いた時点では <b>${pct(late)}</b>。`:" 規制は最後まで発動しなかった。");

/* ---------- map ---------- */
const cv = $("#map"), ctx = cv.getContext("2d");
const cols = Math.max(...D.parcels.map(p=>p.x))+1, rows = Math.max(...D.parcels.map(p=>p.y))+1;
const CW = Math.floor(cv.width/cols), CH = Math.floor(cv.height/rows);
const COL = {acq:"#10B981", pub:"#93C5FD", hh:"#E5E7EB", biz:"#FCD34D", other:"#C4B5FD"};
function ownerClass(id){ if(ACQ.has(id)) return "acq"; if(id===D.municipality) return "pub";
 const r = roleOf[id]; if(r==="household") return "hh"; if(r==="business") return "biz"; return "other"; }
function draw(step){
 const fr = D.frames[step]; ctx.clearRect(0,0,cv.width,cv.height);
 D.parcels.forEach(p=>{
  const own = fr.owner[p.pid], cls = ownerClass(own);
  ctx.fillStyle = COL[cls]; ctx.fillRect(p.x*CW, p.y*CH, CW-2, CH-2);
  ctx.strokeStyle="#fff"; ctx.lineWidth=2; ctx.strokeRect(p.x*CW,p.y*CH,CW-2,CH-2);
  const use = fr.use[p.pid];
  ctx.fillStyle = cls==="acq" ? "#04351F" : "#374151";
  ctx.font = "700 10px sans-serif";
  ctx.fillText(p.pid, p.x*CW+4, p.y*CH+13);
  ctx.font = "700 15px sans-serif";
  const gl = use==="shop"?"商":use==="vacant"?"空":use==="public"?"公":"住";
  ctx.fillText(gl, p.x*CW+CW/2-8, p.y*CH+CH/2+11);
  if(use==="shop" && !fr.tenant[p.pid]){ ctx.strokeStyle="#B45309"; ctx.lineWidth=2;
   ctx.beginPath(); ctx.moveTo(p.x*CW+4,p.y*CH+CH-8); ctx.lineTo(p.x*CW+CW-8,p.y*CH+CH-8); ctx.stroke(); }
 });
}
$("#legend").innerHTML =
 `<div><span class="sw" style="background:${COL.acq}"></span>買い手AIが所有</div>
  <div><span class="sw" style="background:${COL.hh}"></span>住民世帯が所有</div>
  <div><span class="sw" style="background:${COL.biz}"></span>地元事業者が所有</div>
  <div><span class="sw" style="background:${COL.pub}"></span>公有地</div>
  <div style="margin-top:8px;">住＝住宅／商＝商店／空＝空地／公＝公有地</div>
  <div><span style="color:#B45309;font-weight:900;">──</span> 下線＝空き店舗</div>`;

/* ---------- charts ---------- */
function lineChart(svg, series, opts){
 const W=900,H=260,L=54,R=168,T=18,B=34;
 const xs = i => L + (W-L-R) * (nSteps<=1?0:i/(nSteps));
 const ymax = opts.ymax!=null?opts.ymax:Math.max(0.0001,...series.flatMap(s=>s.data.filter(v=>v!=null)));
 const ys = v => H-B - (H-T-B) * (v/ymax);
 let g = `<rect x="0" y="0" width="${W}" height="${H}" fill="#fff"/>`;
 for(let t=0;t<=4;t++){ const v=ymax*t/4, y=ys(v);
  g += `<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#E5E7EB"/>`
     + `<text x="${L-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#6B7280">${opts.fmt(v)}</text>`; }
 for(let s=0;s<=nSteps;s+=Math.max(1,Math.round(nSteps/12))){
  g += `<text x="${xs(s)}" y="${H-12}" text-anchor="middle" font-size="11" fill="#6B7280">${s}</text>`; }
 (opts.marks||[]).forEach(mk=>{ const x=xs(mk.step);
  g += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H-B}" stroke="${mk.color}" stroke-dasharray="4 3"/>`
     + `<text x="${x+4}" y="${T+12}" font-size="11" font-weight="700" fill="${mk.color}">${esc(mk.label)}</text>`; });
 series.forEach((s,si)=>{
  let d="",started=false;
  s.data.forEach((v,i)=>{ if(v==null){return;} const x=xs(i+1),y=ys(v);
   d += (started?"L":"M")+x+" "+y+" "; started=true; });
  g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.5"/>`;
  g += `<text x="${W-R+8}" y="${T+16+si*18}" font-size="12" font-weight="700" fill="${s.color}">${esc(s.label)}</text>`;
 });
 g += `<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#111827"/>`;
 g += `<text x="${W-R}" y="${H-2}" text-anchor="end" font-size="11" fill="#6B7280">経過月</text>`;
 svg.innerHTML = g;
}
const marks = [];
if(k.detection_lag) marks.push({step:k.detection_lag.step,label:"初報道",color:"#B45309"});
if(k.late_index) marks.push({step:k.late_index.step,label:"規制発動",color:"#0A0A0A"});
lineChart($("#ch1"), [
 {label:"① 買い手シェア",color:"#10B981",data:D.kpi.map(r=>r.acquirer_share)},
 {label:"HHI 集中指数",color:"#111827",data:D.kpi.map(r=>r.hhi)},
 {label:"② 認知転相率",color:"#B45309",data:D.cognition.map(r=>r.shift_rate_cum)},
], {ymax:1, fmt:v=>(v*100).toFixed(0)+"%", marks});
lineChart($("#ch2"), [
 {label:"③ 誘発売却(累積)",color:"#059669",data:D.kpi.map(r=>r.cascade_induced)},
 {label:"営業中の店舗数",color:"#2563EB",data:D.kpi.map(r=>r.shops_occupied)},
 {label:"平均賃料(万/月)",color:"#B45309",data:D.kpi.map(r=>r.mean_shop_rent)},
], {fmt:v=>v.toFixed(0), marks});

/* ---------- step-linked panels ---------- */
const slider = $("#slider"); slider.max = nSteps; slider.value = nSteps;
function utterHtml(step){
 const rows = D.utterances.filter(u=>u.step===step);
 if(!rows.length) return '<p class="small">この月、街のSNS・立ち話に記録された発話はない。</p>';
 return rows.map(u=>{
  const f = u.frame==="their_town"?"f-their":u.frame==="our_town"?"f-our":"";
  const lab = u.frame==="their_town"?"あの主体の街":u.frame==="our_town"?"私たちの街":"";
  return `<div class="utt ${f}"><b>${esc(u.name||nameOf[u.from]||u.from)}（${esc(u.role)}）`
   + (lab?` · ${lab}`:"")+`</b><br>${esc(u.text)}</div>`;}).join("");
}
function evtHtml(step){
 const rows = D.events.filter(e=>e.step===step);
 return rows.map(e=>`<tr><td>${esc(e.name||e.id)}<br><span class="small">${esc(e.role)}</span></td>
  <td>${esc(e.action)}</td><td>${esc(e.target)}</td><td>${e.amount||""}</td>
  <td><span class="tag ${e.ok?"t-ok":"t-ng"}">${e.ok?"成立":"不成立"}</span><br>
  <span class="small">${esc(e.outcome)}</span></td>
  <td class="small">${esc(e.reasoning).slice(0,140)}</td></tr>`).join("");
}
function stepInfo(step){
 const r = D.kpi[step-1];
 if(!r) return "第0月（初期状態）";
 return `第${step}月 — 買い手シェア <b>${pct(r.acquirer_share)}</b> ／ HHI ${r.hhi.toFixed(3)}`
  + ` ／ 累計成約 ${r.transfers_cum} ／ 営業中の店 ${r.shops_occupied}/${r.shops_total}`
  + ` ／ 転出率 ${pct(r.resident_outflow)}`;
}
function update(){
 const s = +slider.value;
 $("#stepnum").textContent = s===0?"第0月":"第"+s+"月";
 draw(s); $("#stepinfo").innerHTML = stepInfo(s);
 $("#utts").innerHTML = utterHtml(s);
 $("#evt").querySelector("tbody").innerHTML = evtHtml(s);
}
slider.addEventListener("input", update);
let timer=null;
$("#play").addEventListener("click", ()=>{
 if(timer){clearInterval(timer);timer=null;$("#play").textContent="▶ 再生";return;}
 if(+slider.value>=nSteps) slider.value=0;
 $("#play").textContent="⏸ 停止";
 timer=setInterval(()=>{ if(+slider.value>=nSteps){clearInterval(timer);timer=null;
  $("#play").textContent="▶ 再生";return;} slider.value=+slider.value+1; update(); }, 420);
});

/* ---------- events summary ---------- */
let ev = "";
if(D.publications.length){ ev += "<h3>報道</h3>" + D.publications.map(p=>
 `<div class="utt"><b>第${p.step}月</b><br>${esc(p.headline)}</div>`).join(""); }
else ev += "<h3>報道</h3><p class='small'>期間中、この街のことは記事にならなかった。</p>";
if(D.ordinances.length){ ev += "<h3>規制</h3>" + D.ordinances.map(o=>
 `<div class="utt"><b>第${o.step}月『${esc(o.title)}』</b><br>${esc(o.body)}</div>`).join(""); }
else ev += "<h3>規制</h3><p class='small'>期間中、土地取引の規制は発動しなかった。</p>";
$("#events").innerHTML = ev;

update();
</script></body></html>
"""
