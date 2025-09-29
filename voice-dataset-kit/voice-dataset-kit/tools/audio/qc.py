from __future__ import annotations
import numpy as np
import scipy.signal as ss
import librosa

def rms_db(y):
    eps = 1e-12
    return float(20*np.log10(np.sqrt(np.mean(y*y)+eps)+eps))

def snr_db(y):
    if len(y) < 1024: return -100.0
    sig = float(np.mean(y**2))
    noise = float(np.percentile(np.abs(y), 5)**2) + 1e-12
    return float(10*np.log10(sig/(noise)+1e-12))

def has_clipping(y, thresh=0.999):
    return bool(np.any(np.abs(y) >= thresh))

def detect_hum(y, sr, mains_hz: int|None=None):
    n = min(len(y), 5*sr)
    if n < sr//2: return {"hum_score": 0.0}
    seg = y[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    cands = [50.0, 60.0] if mains_hz is None else [float(mains_hz)]
    total = np.mean(mag) + 1e-9
    best = 0.0
    for f0 in cands:
        s = 0.0
        for k in range(1, 6):
            fk = f0*k
            if fk > sr/2 - 10: break
            band = (freqs >= fk-1.5) & (freqs <= fk+1.5)
            if np.any(band): s += float(np.max(mag[band]))
        best = max(best, s)
    return {"hum_score": float(np.tanh(best/(10*total)))}

def auto_notch(y, sr, mains_hz: int|None=None, q=35.0, strength=0.8):
    cands = [50.0, 60.0] if mains_hz is None else [float(mains_hz)]
    def energy_near(f0):
        w = int(sr*2)
        seg = y[:min(len(y), w)]
        if len(seg)<sr//2: return 0.0
        f, Pxx = ss.welch(seg, fs=sr, nperseg=2048)
        m = (f>f0-2)&(f<f0+2)
        return float(Pxx[m].mean()) if np.any(m) else 0.0
    if mains_hz is None:
        base = 50.0 if energy_near(50.0) >= energy_near(60.0) else 60.0
    else:
        base = float(mains_hz)
    y_f = y.copy()
    depth = max(0.1, min(0.95, strength))
    for k in range(1, 5):
        f0 = base*k
        if f0>sr/2-50: break
        b, a = ss.iirnotch(w0=f0/(sr/2), Q=q)
        y_w = ss.lfilter(b, a, y_f)
        y_f = (1-depth)*y_f + depth*y_w
    return y_f

def detect_plosives(y, sr):
    win = int(0.02*sr); hop = int(0.01*sr)
    if len(y)<win: return {"plosive_ratio": 0.0}
    b, a = ss.butter(4, [20/(sr/2), 150/(sr/2)], btype='band')
    lf = ss.lfilter(b, a, y)
    thr = np.percentile(np.abs(lf), 95)*0.8
    frames=flags=0; i=0
    while i+win<=len(lf):
        f = lf[i:i+win]
        zcr = float(((f[:-1]*f[1:])<0).mean()) if len(f)>1 else 0.0
        if np.max(np.abs(f))>thr and zcr<0.05: flags+=1
        frames+=1; i+=hop
    return {"plosive_ratio": float(flags/max(frames,1))}

def trim_with_margin(y, sr, margin_ms=150, top_db=40.0):
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals.size==0: return y[:0]
    s = int(max(0, intervals[0,0] - margin_ms*sr/1000))
    e = int(min(len(y), intervals[-1,1] + margin_ms*sr/1000))
    return y[s:e]
