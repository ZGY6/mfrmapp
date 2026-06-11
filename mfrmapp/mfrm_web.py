"""
MFRMSight Web Server — 纯 HTTP 版本 (v0.8.0)
不需要 Gradio/Streamlit，手机上直接打开
运行: python mfrm_web.py
"""
import http.server
import json
import tempfile, os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

# 导入引擎 (从同目录的 mfrm_app.py)
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from mfrm_app import parse_minifac_txt, parse_xlsx_interactive, Engine

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MFRMSight v0.8.0</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f7fb;color:#1a1a2e;padding:12px;max-width:720px;margin:0 auto}
h1{font-size:1.5rem;color:#1a56db;text-align:center;margin:12px 0 4px}
.sub{color:#6b7280;font-size:.8rem;text-align:center;margin-bottom:12px}
.card{background:#fff;border-radius:10px;padding:14px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card h2{font-size:1rem;margin-bottom:8px}
input[type=file]{width:100%;padding:30px 16px;border:2px dashed #d1d5db;border-radius:8px;text-align:center;cursor:pointer;font-size:.9rem}
input[type=file]:hover{border-color:#1a56db}
.metrics{display:flex;flex-wrap:wrap;gap:6px}
.m{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-radius:8px;padding:10px;flex:1;min-width:80px;text-align:center}
.m .v{font-size:1.3rem;font-weight:700}
.m .l{font-size:.7rem;opacity:.9}
table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:6px}
th{background:#e8edf5;padding:6px;font-weight:600}
td{padding:5px 6px;text-align:center;border-bottom:1px solid #f0f0f0}
tr:hover td{background:#f8fafc}
select,input[type=text]{padding:8px;border:1px solid #d1d5db;border-radius:8px;font-size:.85rem;margin-top:6px;width:100%}
.btn{background:#1a56db;color:#fff;border:none;padding:10px;border-radius:8px;font-size:.95rem;cursor:pointer;width:100%;margin-top:10px}
.btn:hover{background:#1e40af}
#status{text-align:center;padding:8px;color:#6b7280;font-size:.85rem}
#results{display:none}
.sinfo{font-size:.8rem;color:#4b5563;margin:6px 0}
footer{text-align:center;color:#9ca3af;font-size:.75rem;padding:16px}
</style>
</head>
<body>
<h1>📊 MFRMSight v0.8.0</h1>
<div class="sub">多面Rasch模型分析 · 手机版</div>

<div class="card">
  <h2>📂 上传数据</h2>
  <input type="file" id="f" accept=".txt,.xlsx,.xls">
  <select id="mode">
    <option value="auto">自动检测</option>
    <option value="items">Items模式 (1号-4号 = 题目)</option>
    <option value="student">Student模式 (1号-4号 = 学生)</option>
    <option value="single">单学生模式</option>
  </select>
  <input type="text" id="sname" placeholder="学生名称 (单学生模式)" style="display:none">
  <button class="btn" onclick="go()">🚀 开始分析</button>
  <div id="status"></div>
</div>

<div id="results">
  <div class="card"><div id="sum"></div></div>
  <div class="card"><div id="t1"></div></div>
  <div class="card"><div id="t2"></div></div>
  <div class="card"><div id="t3"></div></div>
  <div class="card"><div id="t4"></div></div>
</div>

<footer>MFRMSight v0.8.0 · Andrich Rating Scale · Fisher-scoring JMLE</footer>

<script>
document.getElementById('mode').onchange=function(){
  document.getElementById('sname').style.display=this.value==='single'?'block':'none'
}

function mc(l,v){return'<div class="m"><div class="v">'+v+'</div><div class="l">'+l+'</div></div>'}

function bt(title,d){
  if(!d||!d.r||d.r.length===0)return''
  var h='<h2>'+title+' <span style="font-weight:400;font-size:.8rem;color:#888">Sep='+d.sp+' Rel='+d.rl+'</span></h2>'
  h+='<table><thead><tr><th>名称</th><th>总分</th><th>ObsAvg</th><th>Meas</th><th>SE</th><th>Infit</th><th>Outfit</th></tr></thead><tbody>'
  d.r.forEach(function(r){
    h+='<tr><td>'+r.l+'</td><td>'+Math.round(r.t)+'</td><td>'+r.oa.toFixed(2)+'</td><td>'+r.m.toFixed(3)+'</td><td>'+r.se.toFixed(3)+'</td><td>'+r.inf.toFixed(3)+'</td><td>'+r.otf.toFixed(3)+'</td></tr>'
  })
  h+='</tbody></table>'
  return h
}

function go(){
  var file=document.getElementById('f').files[0]
  if(!file){document.getElementById('status').innerHTML='请先选择文件';return}
  document.getElementById('status').innerHTML='⏳ 分析中...'
  document.getElementById('results').style.display='none'
  var fd=new FormData()
  fd.append('file',file)
  fd.append('mode',document.getElementById('mode').value)
  fd.append('single',document.getElementById('sname').value)
  fetch('/go',{method:'POST',body:fd})
    .then(function(r){return r.json()})
    .then(function(d){
      if(d.error){document.getElementById('status').innerHTML='❌ '+d.error;return}
      var s=d.summary
      document.getElementById('status').innerHTML='✅ 完成 ('+s.N+'条, Var='+s.var_exp+'%)'
      var h='<div class="metrics">'
      h+=mc('反应数',s.N)+mc('面向',s.n_s+'x'+s.n_r+'x'+s.n_c+(s.n_i>1?'x'+s.n_i:''))
      h+=mc('分数',s.score_range)+mc('方差解释',s.var_exp+'%')+mc('残差SD',s.resid_sd)
      h+='</div><p class="sinfo">ObsMean='+s.obs_mean.toFixed(2)+' | ExpMean='+s.exp_mean.toFixed(2)+' | StResSD='+s.stres_sd.toFixed(4)+'</p>'
      document.getElementById('sum').innerHTML=h
      document.getElementById('t1').innerHTML=bt('🎓 学生面向',d.students)
      document.getElementById('t2').innerHTML=bt('👤 评分者面向',d.raters)
      document.getElementById('t3').innerHTML=bt('📋 标准面向',d.criteria)
      document.getElementById('t4').innerHTML=bt('📝 题目面向',d.items)
      document.getElementById('results').style.display='block'
    })
    .catch(function(e){document.getElementById('status').innerHTML='❌ 网络错误: '+e.message})
}
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        if self.path != "/go":
            self.send_error(404); return
        ct = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ct:
            self.send_error(400); return
        boundary = ct.split("boundary=")[1].encode()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        parts = body.split(b"--" + boundary)
        file_data, mode, single = None, "auto", ""
        for part in parts:
            if b"Content-Disposition" not in part: continue
            hdr, content = part.split(b"\r\n\r\n", 1)
            content = content.rsplit(b"\r\n", 1)[0]
            if b'filename=' in part:
                file_data = content
            elif b'name="mode"' in part:
                mode = content.decode()
            elif b'name="single"' in part:
                single = content.decode()
        if not file_data:
            self._json({"error": "未收到文件"}); return
        sfx = ".txt" if b".txt" in hdr else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as f:
            f.write(file_data); tmp = f.name
        try:
            if sfx == ".txt":
                d = parse_minifac_txt(tmp)
            elif mode == "items":
                d = parse_xlsx_interactive(tmp, 3)
            elif mode == "single" and single:
                d = parse_xlsx_interactive(tmp, 3)
            else:
                d = parse_xlsx_interactive(tmp, 3)
            e = Engine(d).fit()
            r = e.report()
            self._json({"summary": r["summary"], "students": r["facets"].get("students"),
                        "raters": r["facets"].get("raters"), "criteria": r["facets"].get("criteria"),
                        "items": r["facets"].get("items")})
        except Exception as ex:
            import traceback
            self._json({"error": str(ex) + "\n" + traceback.format_exc()})
        finally:
            os.unlink(tmp)

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def log_message(self, *a): pass


if __name__ == "__main__":
    # BUG-005/BUG-006 修复: 自动选择可用端口 + PID 追踪
    import socket
    PORT = 8080
    for _ in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", PORT))
            s.close()
            break
        except OSError:
            PORT += 1
    print(f"\n{'='*50}")
    print(f"  MFRMSight v0.8.0 Web 服务 (PID={os.getpid()})")
    print(f"  打开浏览器: http://localhost:{PORT}")
    print(f"  手机访问:   http://YOUR_IP:{PORT}")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*50}\n")
    http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
