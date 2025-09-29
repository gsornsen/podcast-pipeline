from __future__ import annotations
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
import json

def launch(out_root: Path):
    app = Flask(__name__)
    T = """
    <!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>Dataset Reviewer</title>
    <style>body{font-family:system-ui;margin:24px}.card{border:1px solid #ddd;border-radius:12px;padding:12px 16px;margin-bottom:12px}.row{display:grid;grid-template-columns:240px 1fr auto;gap:12px;align-items:center}textarea{width:100%;min-height:60px}.badge{font-size:12px;padding:2px 6px;border-radius:8px;background:#eee}</style>
    </head><body>
    <div><button onclick='save()'>Save changes</button> <span id='status'></span></div>
    <div id='content'></div>
    <script>
    let data=[];
    async function load(){
      const r = await fetch('/api/list'); data = await r.json();
      const root = document.getElementById('content'); root.innerHTML='';
      data.items.forEach((it,idx)=>{
        const d=document.createElement('div'); d.className='card';
        d.innerHTML = `<div class='row'>
          <audio controls src='/audio/${it.audio}'></audio>
          <textarea id='t_${idx}'>${it.text||''}</textarea>
          <label><input id='r_${idx}' type='checkbox' ${it.reject?'checked':''}/> Reject</label>
        </div>
        <div class='row'><div>
          <span class='badge'>${it.split}</span>
          ${it.qc? `<span class='badge'>SNR ${it.qc.snr_db?.toFixed?.(1)??it.qc.snr_db}</span>`:''}
          ${it.qc? `<span class='badge'>RMS ${it.qc.rms_db?.toFixed?.(1)??it.qc.rms_db}</span>`:''}
          ${it.qc? `<span class='badge'>Hum ${it.qc.hum_score?.toFixed?.(2)??''}</span>`:''}
          ${it.qc? `<span class='badge'>Plosive ${it.qc.plosive_ratio?.toFixed?.(2)??''}</span>`:''}
        </div><small>${it.audio}</small><div></div></div>`;
        root.appendChild(d);
      });
    }
    async function save(){
      const payload = data.items.map((it,idx)=>({audio:it.audio, text:document.getElementById('t_'+idx).value, reject:document.getElementById('r_'+idx).checked}));
      const r = await fetch('/api/save',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({items:payload})});
      const j = await r.json(); document.getElementById('status').innerText=j.msg; load();
    }
    load();
    </script></body></html>
    """
    def read_split(split):
        p = out_root/split/'metadata.jsonl'
        items = []
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                j = json.loads(line); j['split']=split; j['reject']=False; items.append(j)
        return items
    @app.get('/')
    def index(): return render_template_string(T)
    @app.get('/api/list')
    def api_list():
        items = read_split('train')+read_split('val')+read_split('test')
        return jsonify({"items": items})
    @app.post('/api/save')
    def api_save():
        payload = request.get_json(force=True)
        quarantine = out_root/'quarantine'; quarantine.mkdir(parents=True, exist_ok=True)
        metas = {}
        for s in ['train','val','test']:
            p = out_root/s/'metadata.jsonl'
            metas[s] = []
            if p.exists(): metas[s] = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
        for it in payload['items']:
            rel = it['audio']; split = rel.split('/')[0]; relp = '/'.join(rel.split('/')[1:])
            arr = metas.get(split,[])
            for j in arr:
                if j['audio'].endswith(relp):
                    j['text'] = it.get('text', j.get('text',''))
                    if it.get('reject'):
                        src = out_root/split/Path(relp)
                        dest_dir = quarantine/split; dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir/src.name
                        if src.exists(): src.replace(dest)
                        j['audio'] = str(Path('quarantine')/split/dest.name)
                    break
        for s in metas:
            p = out_root/s/'metadata.jsonl'
            with p.open('w', encoding='utf-8') as f:
                for j in metas[s]: f.write(json.dumps(j, ensure_ascii=False)+"\n")
        return jsonify({"ok":True, "msg":"Saved. Rejected moved to quarantine/."})
    @app.get('/audio/<path:rel>')
    def audio(rel):
        full = out_root/rel
        return send_from_directory(full.parent, full.name)
    app.run(host='127.0.0.1', port=7860, debug=False)

def main():
    """Main entry point for the dataset reviewer CLI."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True, help='Output directory containing dataset to review')
    args = ap.parse_args()
    launch(Path(args.out_dir))


if __name__ == '__main__':
    main()
