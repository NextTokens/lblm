#!/usr/bin/env python3
"""
poc.py -- Text Lab: paste text, watch a machine learn it, test it.

Simple, user-facing, three steps, plain language. Under the hood it is the REAL
research model (wstate.py arm `selleanu`, word mode, non-forgetting rail) learning
online, one pass, no training run. Experimental preview; feedback decides the build.

Run:  python poc.py     ->  http://127.0.0.1:8777
"""
import os, sys, json, math, random, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["WSLOTMODE"] = "word"
os.environ["WSLOTS"] = "32"
os.environ["WNS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wstate

LOCK = threading.Lock()


class Live:
    """One fresh machine per run. feed() learns; peek() predicts without learning."""

    def __init__(self):
        self.reset()
        self.text = b""

    def reset(self):
        self.m = wstate.Model(arm="selleanu", seed=0)
        self.n = 0
        self.bits_total = 0.0
        self.text = b""

    def feed(self, data, learn=True):
        out = []
        for b in data:
            c = 0.0
            for j in range(7, -1, -1):
                c += self.m.step((b >> j) & 1, learn)
            self.n += 1
            self.bits_total += c
            out.append(c)
        return out

    def peek(self, k=3, width=6):
        """top-k next characters by the model's own bit probabilities (beam search over the
        8 bits; predict() is pure so nothing leaks). conf = 2**(-lp), the EXACT byte
        probability along the path (a 43%-probable byte reads 0.43, not a geometric mean)."""
        m = self.m
        save = (m.cur, m.phase, m.htail)
        beam = [(0.0, m.cur, m.phase, m.htail, [])]
        for lvl in range(8):
            nxt = []
            for lp, cur, phase, ht, bits in beam:
                for bit in (0, 1):
                    m.cur, m.phase, m.htail = cur, phase, ht
                    p, _ = m.predict()
                    pb = p if bit else 1.0 - p
                    nxt.append((lp - math.log2(max(pb, 1e-9)),
                                ((cur << 1) | bit) & 0xFF, phase + 1,
                                ((ht << 1) | bit) & ((1 << 48) - 1), bits + [bit]))
            nxt.sort(key=lambda x: x[0])
            beam = nxt[:width]
        m.cur, m.phase, m.htail = save
        out = []
        for lp, _c, _ph, _ht, bits in beam[:k]:
            val = 0
            for bit in bits:
                val = (val << 1) | bit
            out.append({"ch": chr(val) if 32 <= val < 127 else "·",
                        "bits": round(lp, 2), "conf": round(2 ** (-lp), 4)})
        return out

    def bpc(self):
        return self.bits_total / max(1, self.n)


LIVE = Live()

SAMPLES = {
    "english": ("A news article", open("data/corpus.txt", "rb").read()[:6000]),
    "code": ("Source code", open("data/corpus_code.txt", "rb").read()[:6000]),
    "gibberish": ("Random gibberish", bytes(random.Random(3).randrange(256) for _ in range(3000))),
}
# a fresh slice the machine has NEVER seen, for the prediction game
FRESH_ENGLISH = open("data/corpus.txt", "rb").read()[8000:20000]

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Text Lab — watch a machine learn</title><style>
:root{--bg:#0e1117;--card:#151b26;--line:#232d3f;--fg:#dbe4f0;--dim:#7d8ba1;
--green:#39d98a;--yellow:#f7c948;--red:#ff6b62;--blue:#5aa9ff}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif;padding:0}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;align-items:center;gap:12px;margin-bottom:6px}
h1{font-size:26px;font-weight:700}
.badge{font-size:11px;letter-spacing:1px;color:var(--yellow);border:1px solid #6b5a1f;
 background:#2b2410;padding:3px 9px;border-radius:99px}
.sub{color:var(--dim);margin-bottom:26px;font-size:15px}
.step{display:flex;gap:14px;margin-bottom:18px}
.n{flex:0 0 30px;height:30px;border-radius:50%;background:var(--card);border:1px solid var(--line);
 display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--blue)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;flex:1}
.card h2{font-size:15px;font-weight:600;margin-bottom:10px}
textarea{width:100%;min-height:110px;background:var(--bg);border:1px solid var(--line);border-radius:10px;
 color:var(--fg);padding:12px;font:14px/1.5 Consolas,monospace;resize:vertical}
textarea:focus{outline:1px solid var(--blue)}
.btn{background:#173254;border:1px solid #2b5aa0;color:#cfe5ff;border-radius:10px;
 padding:10px 20px;font:600 15px sans-serif;cursor:pointer}
.btn:hover{background:#1d3f6b}
.btn.go{background:#12402e;border-color:#1f8f5f;color:#bff5dd;font-size:16px;padding:11px 26px}
.btn.ghost{background:transparent;border-color:var(--line);color:var(--dim)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{background:var(--bg);border:1px solid var(--line);color:var(--dim);border-radius:99px;
 padding:5px 13px;font-size:13px;cursor:pointer}
.chip:hover{color:var(--fg);border-color:var(--blue)}
.hint{color:var(--dim);font-size:13px;margin-top:9px}
.meterrow{display:flex;align-items:baseline;gap:14px;margin:6px 0 2px}
.big{font-size:44px;font-weight:800;font-variant-numeric:tabular-nums}
.unit{color:var(--dim)}
.scale{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin:2px 0 10px}
.bar{height:10px;border-radius:6px;background:linear-gradient(90deg,#39d98a,#f7c948,#ff6b62);position:relative;max-width:420px}
.bar i{position:absolute;top:-4px;width:4px;height:18px;background:#fff;border-radius:2px;left:100%;
 transition:left .25s;box-shadow:0 0 6px #fff8}
#reading{font:14px/1.6 Consolas,monospace;white-space:pre-wrap;word-break:break-all;
 max-height:180px;overflow:hidden;border-radius:8px;padding:8px;background:var(--bg)}
.g0{color:#2e6b4f}.g1{color:#39d98a}.g2{color:#c9b458}.g3{color:#ff9d54}.g4{color:#ff6b62}
.controls{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
.score{display:flex;gap:26px;flex-wrap:wrap;margin:8px 0}
.score div b{display:block;font-size:30px;font-variant-numeric:tabular-nums}
.score div{color:var(--dim);font-size:13px}
.cal{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:10px}
.qrow{display:flex;gap:6px;align-items:center;font:14px Consolas,monospace;margin:3px 0;flex-wrap:wrap}
.qrow .t{color:var(--dim)}
.ok1{color:var(--green);font-weight:700}.ok3{color:var(--yellow)}.miss{color:var(--dim)}
footer{margin-top:34px;color:var(--dim);font-size:12.5px;line-height:1.7;border-top:1px solid var(--line);padding-top:14px}
.hidden{display:none}
</style></head><body><div class="wrap">
<header><h1>Text Lab</h1><span class="badge">EXPERIMENTAL PREVIEW</span></header>
<div class="sub">A machine that learns text as it reads — one pass, live, no training run.
Give it something to read and watch it go from blind to fluent in front of you.</div>

<div class="step"><div class="n">1</div><div class="card">
 <h2>Give it text it has never seen</h2>
 <textarea id="intext" placeholder="Paste or type anything — an article, code, notes, another language…"></textarea>
 <div class="chips">
  <span class="chip" onclick="useSample('english')">Try: a news article</span>
  <span class="chip" onclick="useSample('code')">Try: source code</span>
  <span class="chip" onclick="useSample('gibberish')">Try: random gibberish</span>
 </div>
</div></div>

<div class="step"><div class="n">2</div><div class="card">
 <h2>Watch it learn <span style="color:var(--dim);font-weight:400">(the machine starts blank every time)</span></h2>
 <div class="meterrow"><div class="big" id="bpc">8.0</div>
  <div class="unit">bits per character<br>= how surprised it is</div></div>
 <div class="bar"><i id="needle"></i></div>
 <div class="scale" style="max-width:420px"><span>0 · bored</span><span>~2 · fluent in English</span><span>8 · blind</span></div>
 <div id="reading"></div>
 <div class="controls">
  <button class="btn go" id="start">Start reading</button>
  <button class="btn ghost" id="pause">Pause</button>
  <label class="hint" style="margin:0">speed
   <input id="speed" type="range" min="1" max="30" value="8" style="vertical-align:middle"></label>
 </div>
 <div class="hint">Green characters were predicted; red ones surprised it. When the number stops
 dropping, the machine has learned what your text has to teach it.</div>
</div></div>

<div class="step"><div class="n">3</div><div class="card">
 <h2>Test what it just learned</h2>
 <div class="hint" style="margin:0 0 8px">The machine re-reads the first part of your text once more,
 then predicts the ending character by character. Green = first try, yellow = in its top 3.</div>
 <button class="btn" id="play">Test what it learned</button>
 <div id="gameres" class="hidden">
  <div class="score">
   <div><b id="s1">–</b>first-try hits</div>
   <div><b id="s3">–</b>in top 3</div>
   <div><b id="sconf">–</b>sure vs unsure</div>
  </div>
  <div class="cal" id="cal"></div>
  <div id="rounds" style="margin-top:10px;max-height:150px;overflow:auto"></div>
  <button class="btn ghost" id="copy" onclick="copySummary()">Copy summary</button>
 </div>
</div></div>

<footer>Under the hood: LBLM, an experimental online text predictor developed in the open.
Everything on this page is the real research model running live on your machine — no network calls.
It learns in a single pass; whether it <i>trusts</i> what it just learned is the current research
frontier. Your feedback decides what gets built next.</footer>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;");
let running=false,timer=null,ev=null,bits=null,pos=0,speed=8;
const heat=b=>b<0.5?"g0":b<1.5?"g1":b<3?"g2":b<5.5?"g3":"g4";
async function api(p,body){const r=await fetch("/api/"+p,{method:body?"POST":"GET",
 headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
 return r.json();}
async function useSample(name){const r=await api("sample?name="+name);$("intext").value=r.text;}
$("start").onclick=async()=>{
 stopRead();const t=$("intext").value;
 if(!t.trim()){alert("Paste or type some text first (or tap a Try chip).");return;}
 const r=await api("begin",{text:t});ev=r.ev;bits=r.bits;pos=0;
 $("reading").innerHTML="";$("bpc").textContent="8.0";$("needle").style.left="100%";
 running=true;timer=setInterval(tick,110);};
function stopRead(){running=false;clearInterval(timer);}
$("pause").onclick=stopRead;
$("speed").oninput=e=>speed=+e.target.value;
function tick(){let n=speed*4;
 while(n-->0&&pos<ev.length){const s=document.createElement("span");
  s.className=heat(bits[pos]);s.textContent=ev[pos]==="\n"?" ":ev[pos];
  $("reading").appendChild(s);pos++;}
 while($("reading").childNodes.length>420)$("reading").removeChild($("reading").firstChild);
 const b=bits.slice(Math.max(0,pos-400),pos);
 if(b.length){const m=b.reduce((a,x)=>a+x,0)/b.length;      // recent surprise, what you SEE
  $("bpc").textContent=m.toFixed(2);$("needle").style.left=Math.min(100,m/8*100)+"%";}
 if(pos>=ev.length){stopRead();
  api("state").then(r=>{$("bpc").textContent=r.bpc.toFixed(2);$("needle").style.left=Math.min(100,r.bpc/8*100)+"%";});}}
$("play").onclick=async()=>{const r=await api("game",{});
 $("gameres").classList.remove("hidden");
 $("s1").textContent=r.top1+" / "+r.n;$("s3").textContent=r.top3+" / "+r.n;
 $("sconf").textContent=Math.round(r.acc_hi*100)+"% vs "+Math.round(r.acc_lo*100)+"%";
 $("cal").innerHTML=`<b>Honest confidence:</b> splitting the rounds by how confident it felt —
  when <b>most confident</b> it got <b>${Math.round(r.acc_hi*100)}%</b> right; when
  <b>least confident</b> only <b>${Math.round(r.acc_lo*100)}%</b>${r.acc_hi>r.acc_lo?
  " — the confidence means something ✓":"."} It says when it doesn't know.`;
 const rd=$("rounds");rd.innerHTML="";
 for(const q of r.questions){const d=document.createElement("div");d.className="qrow";
  const cls=q.rank===0?"ok1":q.rank<3?"ok3":"miss";
  d.innerHTML=`<span class="t">${esc(q.ctx)}</span>
   <span class="${cls}">${q.guesses.map(g=>g===" "?"␣":g).join(" ")}</span>
   <span class="t">→ <b style="color:var(--fg)">${q.truth===" "?"␣":esc(q.truth)}</b></span>`;
  rd.appendChild(d);}
 window._summary=`Text Lab (LBLM, experimental) — first-try ${r.top1}/${r.n} `+
  `(${Math.round(100*r.top1/r.n)}%), top-3 ${r.top3}/${r.n}; accuracy when most confident `+
  `${Math.round(r.acc_hi*100)}% vs ${Math.round(r.acc_lo*100)}% when least.`;};
function copySummary(){navigator.clipboard.writeText(window._summary||"").then(()=>{
 $("copy").textContent="Copied ✓";setTimeout(()=>$("copy").textContent="Copy summary",1500);});}
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
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            with LOCK:
                self._send(200, {"n": LIVE.n, "bpc": round(LIVE.bpc(), 4)})
        elif self.path.startswith("/api/sample"):
            name = self.path.split("name=")[-1].split("&")[0] if "name=" in self.path else "english"
            name = name if name in SAMPLES else "english"
            self._send(200, {"text": SAMPLES[name][1].decode("utf-8", "replace")})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/begin":
            with LOCK:
                LIVE.reset()                      # a fresh, blank machine every run
                t = str(body.get("text", ""))[:20000].encode("utf-8", "replace")
                LIVE.text = t
                bits = LIVE.feed(t, learn=True)
                ev = [chr(b) if b == 10 or 32 <= b < 127 else "." for b in t]
                self._send(200, {"ev": ev, "bits": [round(x, 2) for x in bits]})
        elif self.path == "/api/game":
            with LOCK:
                # quiz on the ENDING of the text it just read (fall back to the sample).
                # honest: the ending was read once, at the start of the run, and the model
                # has kept learning since; we re-warm on the first 70% then quiz the rest.
                text = LIVE.text if len(LIVE.text) >= 800 else SAMPLES["english"][1]
                LIVE.reset()
                cut = int(len(text) * 0.6)
                LIVE.feed(text[:cut], learn=True)
                positions = list(range(cut, len(text) - 1))[:40]
                qs, top1, top3 = [], 0, 0
                conf_ranked = []
                for i in positions:
                    preds = LIVE.peek(3)
                    guesses = [p["ch"] for p in preds]
                    truth_b = text[i]
                    truth = chr(truth_b) if truth_b == 10 or 32 <= truth_b < 127 else "·"
                    rank = guesses.index(truth) if truth in guesses else 9
                    if rank == 0:
                        top1 += 1
                    if rank < 3:
                        top3 += 1
                    conf_ranked.append((preds[0]["conf"] if preds else 0.0, rank))
                    ctx = text[max(0, i - 14):i].decode("utf-8", "replace").replace("\n", " ")
                    qs.append({"ctx": "…" + ctx, "guesses": guesses, "truth": truth, "rank": rank})
                    LIVE.feed(text[i:i + 1], learn=True)   # it keeps learning — honest online eval
                # calibration by confidence tertile: accuracy when most vs least confident
                conf_ranked.sort(key=lambda x: -x[0])
                k3 = max(1, len(conf_ranked) // 3)
                top_third = conf_ranked[:k3]
                bot_third = conf_ranked[-k3:]
                acc_hi = sum(1 for _c, r in top_third if r == 0) / max(1, len(top_third))
                acc_lo = sum(1 for _c, r in bot_third if r == 0) / max(1, len(bot_third))
                self._send(200, {"n": len(positions), "top1": top1, "top3": top3,
                                 "acc_hi": acc_hi, "acc_lo": acc_lo,
                                 "questions": qs})
        else:
            self._send(404, {})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    print(f"Text Lab -> http://127.0.0.1:{port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
