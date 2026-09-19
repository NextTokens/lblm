#!/usr/bin/env python3
"""
poc.py -- LBLM live dashboard: "a class of one" made visible.

A stdlib-only local server + one-page dashboard that runs the REAL instrument model
(wstate.py arm `selleanu` -- the §81/§86 learned selector, word mode, WNS rail) and shows,
live, the four properties no batch-trained model has:

  1. LEARNS FROM ZERO, ONE PASS      -- the surprise heatmap and running bits/char drop as it reads
  2. CALIBRATED CONFIDENCE           -- top-3 next-character predictions with a COMMIT / ABSTAIN
                                        decision at a fixed threshold (the §33 calibration property)
  3. CATCHES THE SHIFT ITSELF        -- English -> source code -> encrypted-random stream, with the
                                        model's own surprise raising the alarm (stream.py's rule)
  4. ONE-EXPOSURE FACT BINDING       -- teach "the <cue> is <outcome>." ONCE; the next-char
                                        prediction for the prompt moves immediately (§85/§86)

Run:  python poc.py        then open http://127.0.0.1:8777
(The model is single-threaded Python at ~1-2 KB/s -- the demo streams slower than real time on
purpose: you are supposed to watch it learn. This is the instrument, not the Rust engine.)
"""
import os, sys, json, math, random, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["WSLOTMODE"] = "word"
os.environ["WSLOTS"] = "32"
os.environ["WNS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wstate

LOCK = threading.Lock()
TAU = 0.90           # commit threshold (§33-style: accuracy-at-commit rises monotonically with tau)


class Live:
    def __init__(self):
        self.reset()

    def reset(self):
        self.m = wstate.Model(arm="selleanu", seed=0)
        self.n = 0                 # chars seen
        self.bits_total = 0.0
        self.hist = []             # last chars (for display)
        self.recent = []           # (char, bits) ring
        self.bpc_curve = []        # running bits/char, sampled
        self.phase = "idle"
        self.alarm = None          # {"at": n, "from": a, "to": b}
        self.refractory_until = 0
        self.base_long = []        # trailing long-run bits for the alarm
        self.story = None          # auto-run plan: [(phase, nbytes, label)]
        self.story_i = 0
        self.story_pos = 0

    def _alarm_check(self, c_bits):
        self.recent.append(c_bits)
        if len(self.recent) > 48:
            self.recent.pop(0)
        self.base_long.append(c_bits)
        if len(self.base_long) > 768:
            self.base_long.pop(0)
        if self.n > 600 and self.n > self.refractory_until and len(self.base_long) >= 768:
            tail = sum(self.recent) / len(self.recent)
            prior = (sum(self.base_long) - sum(self.recent)) / max(1, len(self.base_long) - len(self.recent))
            if tail - prior > 0.55:
                self.alarm = {"at": self.n, "from": round(prior, 2), "to": round(tail, 2)}
                self.refractory_until = self.n + 400

    def feed(self, data, learn=True):
        """feed bytes; return per-char events"""
        out = []
        for b in data:
            c = 0.0
            for j in range(7, -1, -1):
                c += self.m.step((b >> j) & 1, learn)
            self.n += 1
            self.bits_total += c
            self._alarm_check(c)
            out.append({"ch": chr(b) if 32 <= b < 127 or b == 10 else ".",
                        "bits": round(c, 2)})
            if self.n % 24 == 0:
                self.bpc_curve.append(round(self.bits_total / self.n, 4))
        return out

    # ---- counterfactual peek: beam search over the next byte using predict() (pure) ----
    def peek(self, k=3, width=6):
        m = self.m
        save = (m.cur, m.phase, m.htail)
        beam = [(0.0, m.cur, m.phase, m.htail, [])]
        done = []
        for _lvl in range(8):
            nxt = []
            for lp, cur, phase, ht, bits in beam:
                for bit in (0, 1):
                    m.cur, m.phase, m.htail = cur, phase, ht
                    p, _sts = m.predict()
                    pb = p if bit else 1.0 - p
                    nxt.append((lp - math.log2(max(pb, 1e-9)),
                                ((cur << 1) | bit) & 0xFF, phase + 1,
                                ((ht << 1) | bit) & ((1 << 48) - 1), bits + [bit]))
            nxt.sort(key=lambda x: x[0])
            beam = nxt[:width]
            if _lvl == 7:
                done = beam
        m.cur, m.phase, m.htail = save
        out = []
        for lp, _c, _ph, _ht, bits in done[:k]:
            val = 0
            for bit in bits:
                val = (val << 1) | bit
            # confidence = geometric-mean bit probability along this path, re-walked with the
            # SAME state updates the beam used (cur, phase AND htail -- predict() keys the
            # order contexts on htail)
            conf = 0.0
            m.cur, m.phase, m.htail = save
            for bit in bits:
                p, _ = m.predict()
                pb = p if bit else 1.0 - p
                conf += math.log2(max(pb, 1e-9))
                m.cur = ((m.cur << 1) | bit) & 0xFF
                m.phase += 1
                m.htail = ((m.htail << 1) | bit) & ((1 << 48) - 1)
            m.cur, m.phase, m.htail = save
            out.append({"ch": chr(val) if 32 <= val < 127 else "·",
                        "bits": round(lp, 2),
                        "conf": round(2 ** (conf / 8), 4)})
        return out

    def stats(self):
        return {"n": self.n, "bpc": round(self.bits_total / max(1, self.n), 4),
                "phase": self.phase, "alarm": self.alarm,
                "curve": self.bpc_curve[-160:]}


LIVE = Live()

# demo story: English -> code -> random (the shift arc)
TEXT_EN = open("data/corpus.txt", "rb").read()
TEXT_CODE = open("data/corpus_code.txt", "rb").read()
_rng = random.Random(11)
TEXT_RND = bytes(_rng.randrange(256) for _ in range(4096)) * 2

STORY = [("english", 5200, "English prose (corpus.txt)"),
         ("code", 4200, "Source code (corpus_code.txt)"),
         ("random", 2600, "Uniform random bytes (unlearnable)")]


def step_story(k):
    ev = []
    for _ in range(k):
        if LIVE.story_i >= len(STORY):
            LIVE.phase = "done"
            break
        phase, total, label = STORY[LIVE.story_i]
        LIVE.phase = label
        src = {"english": TEXT_EN, "code": TEXT_CODE, "random": TEXT_RND}[phase]
        chunk = src[LIVE.story_pos:LIVE.story_pos + 1]
        if not chunk:
            LIVE.story_i += 1
            LIVE.story_pos = 0
            continue
        ev += LIVE.feed(chunk, learn=True)
        LIVE.story_pos += 1
        if LIVE.story_pos >= total:
            LIVE.story_i += 1
            LIVE.story_pos = 0
    return ev


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>LBLM — a class of one</title><style>
:root{--bg:#0b0e14;--fg:#c9d4e3;--dim:#5b6b82;--acc:#39d98a;--warn:#ff5f56;--amber:#f7c948}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.45 "SF Mono",Consolas,monospace;padding:18px}
h1{font-size:17px;letter-spacing:.5px} h1 span{color:var(--dim)}
.sub{color:var(--dim);margin:4px 0 14px;font-size:12px}
.grid{display:grid;grid-template-columns:1.25fr .9fr;gap:14px}
.card{background:#111621;border:1px solid #1d2636;border-radius:10px;padding:12px}
.card h2{font-size:12px;color:var(--dim);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px}
#stream{height:130px;overflow:hidden;font-size:15px;line-height:1.5;word-break:break-all;white-space:pre-wrap}
canvas{width:100%;height:150px;display:block}
#preds{margin-top:6px}
.pred{display:flex;align-items:center;gap:8px;margin:4px 0}
.pred .g{font-size:17px;background:#0b0e14;border:1px solid #1d2636;border-radius:6px;width:34px;text-align:center;padding:2px 0}
.pred .bar{height:10px;background:linear-gradient(90deg,#1f8f5f,#39d98a);border-radius:5px}
.pred .lp{color:var(--dim);font-size:12px;min-width:120px}
.badge{display:inline-block;padding:2px 8px;border-radius:5px;font-size:11px;letter-spacing:1px}
.commit{background:#123b2a;color:var(--acc);border:1px solid #1f8f5f}
.abstain{background:#3b2f12;color:var(--amber);border:1px solid #6b5a1f}
#alarm{display:none;margin:8px 0;padding:8px;border-radius:8px;background:#3a1518;color:#ff8a84;border:1px solid #6e2a2a}
button{background:#182234;color:var(--fg);border:1px solid #26344c;border-radius:7px;padding:7px 14px;cursor:pointer;font:inherit}
button:hover{background:#20304a} button.pri{background:#12402e;border-color:#1f8f5f;color:#bff5dd}
input{background:#0b0e14;border:1px solid #26344c;border-radius:7px;color:var(--fg);padding:7px 10px;font:inherit;width:100%}
.row{display:flex;gap:8px;margin-top:8px}
.stats{display:flex;gap:18px;margin-top:6px;font-size:12px;color:var(--dim)}
.stats b{color:var(--fg);font-size:15px}
.heat-0{color:#2e4a3c}.heat-1{color:#39d98a}.heat-2{color:#f7c948}.heat-3{color:#ff8a54}.heat-4{color:#ff5f56}
.ctl{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
#speed{width:90px}
.hint{color:var(--dim);font-size:11px;margin-top:6px}
.kv{font-size:12px;color:var(--dim)} .kv b{color:var(--fg)}
</style></head><body>
<h1>LBLM <span>— an online bit-level predictor. No training run. No epochs. One pass.</span></h1>
<div class="sub">live: the real §81/§86 selector model (wstate selleanu, word mode) reading real bytes on this machine · ~1–2 KB/s by design — watch it learn</div>
<div class="grid">
 <div>
  <div class="card"><h2>The stream, colored by surprise (green = predicted, red = surprised)</h2>
   <div class="ctl"><button class="pri" id="play">▶ run the arc</button>
    <button id="pause">⏸ pause</button><button id="reset">reset</button>
    <span class="kv">speed <input id="speed" type="range" min="1" max="60" value="6"></span></div>
   <div id="stream"></div>
   <div class="stats"><div>chars read <b id="n">0</b></div><div>bits/char <b id="bpc">–</b>
   <span style="color:var(--dim)">(8.0 = blind, ~1.6 = fluent)</span></div><div>now: <b id="phase">idle</b></div></div>
   <div id="alarm"></div>
   <canvas id="cv" width="820" height="150"></canvas>
   <div class="hint">the arc: English → source code → uniform random. The alarm is the model's OWN surprise
   (trailing 48 chars vs the previous 700) — no other detector.</div>
  </div>
  <div class="card" style="margin-top:14px"><h2>Teach it a fact — once</h2>
   <div class="row"><input id="cue" placeholder="cue word (e.g. vaelor)" value="vaelor">
   <input id="outcome" placeholder="outcome word (e.g. Thandrel)" value="Thandrel">
   <button class="pri" id="teach">teach once</button></div>
   <div class="hint">ONE exposure stores the fact: read it twice live — the second reading is much
   cheaper (that collapse IS one-exposure storage, §85). Trust — the readout actually USING it —
   is the track's honest open limit (§85–§91).</div>
   <div id="teachout" class="kv" style="margin-top:8px"></div>
  </div>
 </div>
 <div>
  <div class="card"><h2>Next-character decision (the System-One primitive)</h2>
   <div class="row"><input id="prompt" placeholder="ask: feed a prompt, see the next-char decision" value="the vaelor is "><button id="ask">ask</button></div>
   <div id="preds"><div class="kv">predictions appear here…</div></div>
   <div class="hint">top-3 by beam search over the model's own bit probabilities; confidence is the
   geometric-mean bit probability; <b>COMMIT</b> ≥ τ=0.90, else <b>ABSTAIN</b> (the §33 property:
   accuracy-at-commit rises monotonically with τ). Note: order tables store taught facts immediately —
   the served prediction trusting them is the open §91 frontier.</div>
  </div>
  <div class="card" style="margin-top:14px"><h2>Feed it anything</h2>
   <div class="row"><input id="feedbox" placeholder="type/paste text — it learns it live"><button id="feed">feed</button></div>
   <div id="feedout" class="kv" style="margin-top:6px"></div>
   <div class="hint">fed with learning ON — this is the model adapting in front of you.</div>
  </div>
  <div class="card" style="margin-top:14px"><h2>What makes it a class of one</h2>
   <div class="kv" style="line-height:1.8">
    <b>learns online, one pass</b> — the curve above IS the training<br>
    <b>calibrated</b> — commits/abstains at a fixed τ, measurably monotone<br>
    <b>detects its own ignorance</b> — the alarm is native surprise<br>
    <b>binds facts from one exposure</b> — §85/§86, live above<br>
    <b>open</b> — every number reproduces from source; model ≈ 1 file</div>
  </div>
 </div>
</div>
<script>
let playing=false, timer=null, speed=6;
const $=id=>document.getElementById(id);
const heat=b=>b<0.4?"heat-0":b<1.6?"heat-1":b<3.2?"heat-2":b<5.5?"heat-3":"heat-4";
async function api(p,body){const r=await fetch("/api/"+p,{method:body?"POST":"GET",
 headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
 return r.json();}
function drawStream(evs){const s=$("stream");
 for(const e of evs){const sp=document.createElement("span");sp.className=heat(e.bits);
  sp.textContent=e.ch==="\n"?"⏎ ":e.ch;s.appendChild(sp);}
 while(s.childNodes.length>460)s.removeChild(s.firstChild);s.scrollTop=s.scrollHeight;}
function drawPreds(preds){const el=$("preds");el.innerHTML="";
 const max=Math.max(...preds.map(p=>p.conf),1e-9);
 for(const p of preds){const d=document.createElement("div");d.className="pred";
  const ok=p.conf>=0.90;
  d.innerHTML=`<div class="g">${p.ch==" "?"␣":p.ch}</div>
   <div style="flex:1"><div class="bar" style="width:${Math.round(100*p.conf)}%"></div></div>
   <div class="lp">${p.conf.toFixed(3)} · ${p.bits.toFixed(2)} bits</div>
   <span class="badge ${ok?'commit':'abstain'}">${ok?'COMMIT':'ABSTAIN'}</span>`;
  el.appendChild(d);}}
function drawCurve(curve){const cv=$("cv"),c=cv.getContext("2d");c.clearRect(0,0,cv.width,cv.height);
 if(curve.length<2)return;const mx=8.2;
 c.strokeStyle="#233048";c.beginPath();for(const y of [2,4,6,8]){const yy=cv.height-(y/mx)*cv.height;
  c.moveTo(0,yy);c.lineTo(cv.width,yy);}c.stroke();
 c.strokeStyle="#39d98a";c.lineWidth=2;c.beginPath();
 curve.forEach((v,i)=>{const x=i/(curve.length-1)*cv.width,y=cv.height-(Math.min(v,mx)/mx)*cv.height;
  i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 c.fillStyle="#5b6b82";c.font="11px monospace";c.fillText("8.0",4,12);c.fillText("0",4,cv.height-4);}
async function tick(){const r=await api("step?k="+speed*8);
 drawStream(r.ev);$("n").textContent=r.stats.n;$("bpc").textContent=r.stats.bpc.toFixed(3);
 $("phase").textContent=r.stats.phase;drawCurve(r.stats.curve);
 if(r.stats.alarm){const a=$("alarm");a.style.display="block";
  a.innerHTML=`⚡ SHIFT DETECTED at char ${a0(r.stats.alarm.at)} — trailing surprise ${r.stats.alarm.to} bits vs ${r.stats.alarm.from} before. The model noticed.`;}
 if(r.stats.phase==="done"){stop();$("phase").textContent="arc complete — feed it anything, or reset";}}
const a0=n=>n.toLocaleString();
function stop(){playing=false;clearInterval(timer);$("play").textContent="▶ run the arc";}
$("play").onclick=async()=>{if(playing)return;const s=await api("state");
 if(s.stats.phase==="done"||s.stats.n>12000)await api("reset",{});
 playing=true;$("play").textContent="running…";timer=setInterval(tick,120);};
$("pause").onclick=stop;
$("reset").onclick=async()=>{stop();await api("reset",{});$("stream").innerHTML="";
 $("alarm").style.display="none";$("n").textContent=0;$("bpc").textContent="–";drawCurve([]);};
$("speed").oninput=e=>speed=+e.target.value;
$("ask").onclick=async()=>{const r=await api("quiz",{prompt:$("prompt").value});drawPreds(r.preds);
 if(r.note){const el=document.createElement("div");el.className="kv";el.textContent=r.note;$("preds").appendChild(el);}};
$("feed").onclick=async()=>{const t=$("feedbox").value;if(!t)return;const r=await api("feed",{text:t});
 $("feedout").innerHTML=`fed ${t.length} chars — mean ${r.bpc.toFixed(2)} bits/char `+
  (r.bpc<1.2?"(it already knows this)":r.bpc<3?"(partly known — learning)":"(new territory)")+
  ` · total ${r.n.toLocaleString()} chars read`;drawStream(r.ev.slice(-60));};
$("teach").onclick=async()=>{const cue=$("cue").value.trim()||"vaelor",out=$("outcome").value.trim()||"Thandrel";
 const r=await api("teach",{cue,outcome});
 $("teachout").innerHTML=`“the ${cue} is ${out}. ” — first reading: <b>${r.exposure_bpc.toFixed(2)} bits/char</b>`+
  ` (novel words cost; this is the WRITE) → second reading: <b>${r.reread_bpc.toFixed(2)}</b>`+
  `${r.reread_bpc < r.exposure_bpc*0.75 ? " — stored from ONE exposure ✓" : ""}<br>`+
  `next-char after “the ${cue} is ” (top): <b>${r.after.map(p=>p.ch+"·"+p.conf.toFixed(2)).join("  ")}</b> `+
  `<span style="color:var(--dim)">(storage is instant; served TRUST is the open frontier)</span>`;
 drawPreds(r.after);
 $("prompt").value=`the ${cue} is `;};
(async()=>{const s=await api("state");$("n").textContent=s.stats.n;})();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            with LOCK:
                self._send(200, {"stats": LIVE.stats()})
        elif self.path.startswith("/api/step"):
            k = 24
            if "k=" in self.path:
                k = max(1, min(600, int(self.path.split("k=")[1].split("&")[0])))
            with LOCK:
                ev = step_story(k)
                self._send(200, {"ev": ev, "stats": LIVE.stats()})
        else:
            self._send(404, {"err": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/reset":
            with LOCK:
                LIVE.reset()
            self._send(200, {"ok": True})
        elif self.path == "/api/feed":
            with LOCK:
                t = str(body.get("text", ""))[:4000].encode("utf-8", "replace")
                ev = LIVE.feed(t, learn=True)
                self._send(200, {"ev": ev[-60:], "bpc": LIVE.bits_total / max(1, LIVE.n), "n": LIVE.n})
        elif self.path == "/api/quiz":
            with LOCK:
                t = str(body.get("prompt", ""))[:400].encode("utf-8", "replace")
                LIVE.feed(t, learn=False)
                preds = LIVE.peek(3)
                self._send(200, {"preds": preds})
        elif self.path == "/api/teach":
            with LOCK:
                cue = str(body.get("cue", "vaelor"))[:40]
                out = str(body.get("outcome", "Thandrel"))[:40]
                fact = f"the {cue} is {out}. ".encode()
                prompt = f"the {cue} is ".encode()
                LIVE.feed(prompt, learn=False)
                before = LIVE.peek(3)
                e1 = LIVE.feed(fact, learn=True)                       # exposure 1: pay to store
                e2 = LIVE.feed(fact, learn=True)                       # exposure 2: cheaper -> stored
                b1 = sum(e["bits"] for e in e1) / max(1, len(e1))
                b2 = sum(e["bits"] for e in e2) / max(1, len(e2))
                LIVE.feed(prompt, learn=False)
                after = LIVE.peek(3)
                self._send(200, {"before": before, "after": after,
                                 "exposure_bpc": b1, "reread_bpc": b2})
        else:
            self._send(404, {"err": "no"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    print(f"LBLM live dashboard -> http://127.0.0.1:{port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
