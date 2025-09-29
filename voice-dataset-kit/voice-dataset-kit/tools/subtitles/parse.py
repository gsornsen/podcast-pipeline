from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

def _parse_srt(path: Path):
    import pysrt
    out = []
    subs = pysrt.open(str(path), encoding='utf-8')
    for it in subs:
        st = it.start.hours*3600 + it.start.minutes*60 + it.start.seconds + it.start.milliseconds/1000
        et = it.end.hours*3600 + it.end.minutes*60 + it.end.seconds + it.end.milliseconds/1000
        out.append((float(st), float(et), it.text.replace('\n',' ').strip()))
    return out

def _parse_vtt(path: Path):
    import webvtt
    def p(ts: str):
        h, m, s = ts.split(':')
        return int(h)*3600 + int(m)*60 + float(s)
    out = []
    for c in webvtt.read(str(path)):
        out.append((p(c.start), p(c.end), c.text.replace('\n',' ').strip()))
    return out

def load_for_audio(audio_path: Path) -> List[Tuple[float,float,str]]:
    srt = audio_path.with_suffix('.srt')
    vtt = audio_path.with_suffix('.vtt')
    subs = []
    try:
        if srt.exists(): subs = _parse_srt(srt)
        elif vtt.exists(): subs = _parse_vtt(vtt)
    except Exception as e:
        print(f"[subs] parse error {audio_path.name}: {e}")
    # Fix non‑monotonic cues: sort, drop negatives, merge overlaps/tiny gaps
    subs = [(s,e,t) for (s,e,t) in subs if e > s + 1e-3]
    subs.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s,e,t in subs:
        if not merged: merged.append([s,e,t]); continue
        ps,pe,pt = merged[-1]
        if s <= pe + 0.05:
            merged[-1][1] = max(pe, e)
            merged[-1][2] = (pt + ' ' + t).strip()
        else:
            merged.append([s,e,t])
    return [(float(s),float(e),t) for s,e,t in merged]

def windows_from_subs(subs, min_sec, max_sec):
    out = []
    if not subs: return out
    st,et,txt = subs[0]
    for s,e,t in subs[1:]:
        if (e - st) <= max_sec:
            et = e; txt = (txt + ' ' + t).strip()
        else:
            if (et - st) >= min_sec: out.append((st,et,txt))
            st,et,txt = s,e,t
    if (et - st) >= min_sec: out.append((st,et,txt))
    return out
