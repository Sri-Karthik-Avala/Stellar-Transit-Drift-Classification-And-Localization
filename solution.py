# made by - Karthik
import os, numpy as np, pandas as pd

N = 1600
SCAN_LEN = 40 * 40
LGRID = np.geomspace(300, 1600, 55)
SPOT_BAND = np.geomspace(120, 260, 20)
KM = 22
WHALF = 16

def find_file(name):
    cands = ["dataset/public", ".", "public", "input", "dataset", "data",
             "../input", "../dataset/public", "/kaggle/input"]
    for d in cands:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)

def gauss_kernel(sig):
    r = int(round(4 * sig))
    x = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (x / sig) ** 2)
    return g / g.sum()

G = gauss_kernel(2.5)

def rolling_median(s, win):
    r = win // 2
    p = np.pad(s, r, mode="reflect")
    sw = np.lib.stride_tricks.sliding_window_view(p, win)
    return np.median(sw, axis=1)

def gsmooth(x, sig):
    r = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="reflect"), k, "valid")

def matched(res):
    return -np.convolve(res, G, "same")

def fold_phase(mf, P, nb=400):
    x = np.arange(N)
    ph = ((x % P) / P * nb).astype(int) % nb
    prof = np.bincount(ph, weights=mf, minlength=nb)
    cnt = np.bincount(ph, minlength=nb)
    prof = prof / np.maximum(cnt, 1)
    k = np.ones(5) / 5
    profs = np.convolve(np.r_[prof[-2:], prof, prof[:2]], k, "same")[2:-2]
    i = int(profs.argmax())
    y0, y1, y2 = profs[(i - 1) % nb], profs[i], profs[(i + 1) % nb]
    den = y0 - 2 * y1 + y2
    di = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    c0 = ((i + np.clip(di, -1, 1)) + 0.5) / nb * P
    peak = profs.max()
    base = np.median(profs)
    spread = profs.std() + 1e-9
    return c0 % P, (peak - base) / spread

def detrend_iter(s, P, sig_bg=6, mw=5, niter=2):
    bg = rolling_median(s, 15)
    res = s - bg
    mf = matched(res)
    c0, fs = fold_phase(mf, P)
    for _ in range(niter):
        mask = np.ones(N, bool)
        for k in range(int((N - c0) / P) + 1):
            cen = c0 + k * P
            mask[max(0, int(round(cen - mw))):min(N, int(round(cen + mw + 1)))] = False
        xf = np.arange(N)
        bgi = np.interp(xf, xf[mask], s[mask])
        bg = gsmooth(bgi, sig_bg)
        res = s - bg
        mf = matched(res)
        c0, fs = fold_phase(mf, P)
    return res, mf, c0, fs, bg

def subpix(mf, jp):
    if 0 < jp < N - 1:
        y0, y1, y2 = mf[jp - 1], mf[jp], mf[jp + 1]
        den = y0 - 2 * y1 + y2
        return (np.clip(0.5 * (y0 - y2) / den, -1, 1) if abs(den) > 1e-12 else 0.0), mf[jp]
    return 0.0, mf[jp]

def localize_win(mf, centers, half):
    xs, ws, kk = [], [], []
    for k, cen in centers:
        lo = int(max(0, round(cen - half)))
        hi = int(min(N, round(cen + half + 1)))
        if hi - lo < 3:
            continue
        seg = mf[lo:hi]
        j = int(seg.argmax())
        jp = lo + j
        dx, h = subpix(mf, jp)
        xs.append(jp + dx)
        ws.append(max(h, 0.0))
        kk.append(k)
    return np.array(kk, float), np.array(xs), np.array(ws)

def robust_c(kk, xs, ws, P):
    r = xs - kk * P
    w = np.clip(ws, 0, None)
    c = np.sum(w * r) / (w.sum() + 1e-9)
    mask = w > 0
    for _ in range(6):
        d = r - c
        med = np.median(d[mask]) if mask.any() else 0.0
        mad = np.median(np.abs(d[mask] - med)) if mask.any() else 0.0
        thr = max(3 * 1.4826 * mad, 3.0)
        mask = (np.abs(d - med) < thr) & (ws > 0)
        if mask.sum() < 2:
            break
        c = np.sum(w[mask] * r[mask]) / (w[mask].sum() + 1e-9)
    return c, mask

def fit_sin(kk, xs, ws, mask, P, lam=0.0, lmed=560.0):
    k = kk[mask]; x = xs[mask]; w = np.clip(ws[mask], 0, None)
    pos = k * P; base = x - pos; sw = np.sqrt(w); one = np.ones_like(pos)
    swsum = sw.sum() + 1e-9
    def pen(L):
        return lam * (np.log(L) - np.log(lmed)) ** 2 if lam > 0 else 0.0
    cands = []
    for L in LGRID:
        f = 2 * np.pi / L
        M = np.stack([one, np.cos(f * pos), np.sin(f * pos)], 1) * sw[:, None]
        coef, _, _, _ = np.linalg.lstsq(M, base * sw, rcond=None)
        rss = np.sum((base * sw - M @ coef) ** 2) / swsum + pen(L)
        r = base - (coef[0] + coef[1] * np.cos(f * pos) + coef[2] * np.sin(f * pos))
        cands.append((rss, L, coef, r.std()))
    cands.sort(key=lambda t: t[0])
    pick = cands[0]
    for rss, L, coef, rstd in cands[:10]:
        if np.hypot(coef[1], coef[2]) < 2.2 * max(rstd, 1.4) + 2.0:
            pick = (rss, L, coef, rstd)
            break
    br, L, coef, _ = pick
    for Lf in np.linspace(L * 0.85, L * 1.18, 12):
        f = 2 * np.pi / Lf
        M = np.stack([one, np.cos(f * pos), np.sin(f * pos)], 1) * sw[:, None]
        c2, _, _, _ = np.linalg.lstsq(M, base * sw, rcond=None)
        rss = np.sum((base * sw - M @ c2) ** 2) / swsum + pen(Lf)
        if rss < br:
            br = rss; L = Lf; coef = c2
    f = 2 * np.pi / L
    amp = np.hypot(coef[1], coef[2])
    rss_const = np.sum(((base - np.sum(w * base) / w.sum()) * sw) ** 2)
    resid = base - (coef[0] + coef[1] * np.cos(f * pos) + coef[2] * np.sin(f * pos))
    return dict(L=L, coef=coef, f=f, amp=amp, rss_const=rss_const, rss_sin=br,
                residstd=resid.std() if mask.sum() > 3 else 0.0, resid=resid)

def build_stack(mf, res, c0, P):
    sd = mf.std() + 1e-9
    rsd = res.std() + 1e-9
    img = np.zeros((2, KM, 2 * WHALF), np.float32)
    msk = np.zeros(KM, np.float32)
    ks = list(range(0, int((N - c0) / P) + 1))
    for i, k in enumerate(ks[:KM]):
        cen = c0 + k * P
        j = int(round(cen)); lo = j - WHALF; hi = j + WHALF
        a = max(0, lo); b = min(N, hi)
        img[0, i, a - lo:b - lo] = mf[a:b] / sd
        img[1, i, a - lo:b - lo] = -res[a:b] / rsd
        msk[i] = 1
    return img, msk

FEAT_NAMES = ["amp", "amp_deb", "rawresid_std", "sigeps", "fit_improve", "fstat", "ls_power",
              "smooth", "nin", "K", "meandepth", "foldsnr", "spot", "raw_std", "P", "L",
              "amp_snr", "log_rrs", "log_ampdeb", "rrs_over_sig", "amp_spotband", "amp_ratio",
              "ndip_ratio", "depth_cv", "spot_corr"]

def process(scan, P):
    raw_std = scan.std()
    res, mf, c0, foldsnr, bg = detrend_iter(scan, P)
    stack, smask = build_stack(mf, res, c0, P)
    ks = list(range(0, int((N - c0) / P) + 1))
    kk, xs, ws = localize_win(mf, [(k, c0 + k * P) for k in ks], P / 2)
    if len(kk) < 5:
        return None
    c, mask = robust_c(kk, xs, ws, P)
    if mask.sum() >= 6:
        J0 = fit_sin(kk, xs, ws, mask, P)
        f0 = J0["f"]; co0 = J0["coef"]
        pred = [(k, co0[0] + k * P + co0[1] * np.cos(f0 * k * P) + co0[2] * np.sin(f0 * k * P)) for k in ks]
    else:
        pred = [(k, c + k * P) for k in ks]
    kk2, xs2, ws2 = localize_win(mf, pred, 11)
    if len(kk2) < 5:
        kk2, xs2, ws2 = kk, xs, ws
    kk, xs, ws = kk2, xs2, ws2
    K = len(kk)
    c, mask = robust_c(kk, xs, ws, P)
    nin = int(mask.sum())
    use = mask if mask.sum() >= 5 else (ws > 0)
    J = fit_sin(kk, xs, ws, use, P)
    sigeps = J["residstd"]
    nfit = max(nin, 5)
    amp_floor = sigeps * np.sqrt(2.0 / nfit)
    amp_deb = np.sqrt(max(J["amp"] ** 2 - 2 * amp_floor ** 2, 0.0))
    fit_improve = (J["rss_const"] - J["rss_sin"]) / (J["rss_const"] + 1e-9)
    nfit2 = max(nin, 6)
    fstat = ((J["rss_const"] - J["rss_sin"]) / 2.0) / ((J["rss_sin"] / max(nfit2 - 3, 1)) + 1e-12)
    fstat = np.log1p(max(fstat, 0.0))
    ls_power = fit_improve
    rr = J["resid"]
    if len(rr) > 3:
        d2 = np.diff(rr, 2)
        smooth = 1.0 - min(np.var(d2) / (2 * np.var(rr) + 1e-9), 2.0) / 2.0
    else:
        smooth = 0.0
    roff = (xs - kk * P - c)[mask] if mask.sum() > 2 else (xs - kk * P - c)
    rawresid_std = roff.std() if len(roff) > 2 else 0.0
    meandepth = ws[mask].mean() if mask.sum() > 0 else ws.mean()
    F = np.abs(np.fft.rfft(res - res.mean()))
    fr = np.fft.rfftfreq(N)
    spot = F[(fr > 1 / 400) & (fr < 1 / 60)].mean() / (F[(fr > 1 / 30) & (fr < 1 / 8)].mean() + 1e-9)
    latt = c + np.arange(0, 40) * P
    knext = int(np.where(latt > 1599)[0][0])
    next_lat = latt[knext]
    f = J["f"]; co = J["coef"]; ka = np.arange(0, 40)
    xpred = co[0] + ka * P + co[1] * np.cos(f * ka * P) + co[2] * np.sin(f * ka * P)
    nz = np.where(xpred > 1599)[0]
    next_drift = xpred[nz[0]] if len(nz) else next_lat
    amp_snr = amp_deb / (sigeps / np.sqrt(nfit) + 1e-9)
    kI = kk[mask] if mask.sum() >= 3 else kk
    xI = xs[mask] if mask.sum() >= 3 else xs
    wI = np.clip((ws[mask] if mask.sum() >= 3 else ws), 0, None)
    posI = kI * P; baseI = xI - posI; swI = np.sqrt(wI); oneI = np.ones_like(posI)
    amp_spotband = 0.0
    if len(posI) > 4:
        bb = 1e18
        for L in SPOT_BAND:
            ff = 2 * np.pi / L
            M = np.stack([oneI, np.cos(ff * posI), np.sin(ff * posI)], 1) * swI[:, None]
            cco, _, _, _ = np.linalg.lstsq(M, baseI * swI, rcond=None)
            rr2 = np.sum((baseI * swI - M @ cco) ** 2)
            if rr2 < bb:
                bb = rr2; amp_spotband = np.hypot(cco[1], cco[2])
    amp_ratio = amp_deb / (amp_spotband + 0.5)
    ndip_ratio = nin / max(K, 1)
    depths = ws[mask] if mask.sum() > 0 else ws
    depth_cv = depths.std() / (depths.mean() + 1e-9) if len(depths) > 1 else 0.0
    gb = np.gradient(bg)
    sgv = np.array([gb[int(round(np.clip(pp, 0, N - 1)))] for pp in posI])
    residI = xI - (c + kI * P)
    spot_corr = abs(np.corrcoef(residI, sgv)[0, 1]) if (len(residI) > 3 and sgv.std() > 1e-12 and residI.std() > 1e-12) else 0.0
    feats = [J["amp"], amp_deb, rawresid_std, sigeps, fit_improve, fstat, ls_power, smooth,
             nin, K, meandepth, foldsnr, spot, raw_std, P, J["L"], amp_snr,
             np.log1p(rawresid_std), np.log1p(amp_deb), rawresid_std / (sigeps + 1e-9),
             amp_spotband, amp_ratio, ndip_ratio, depth_cv, spot_corr]
    dips = (kk, xs, ws, mask, c, P)
    return feats, next_lat, next_drift, stack, smask, dips

def extract(df, id2idx, imgs):
    X, NL, ND, ST, SM, DP = [], [], [], [], [], []
    for _, row in df.iterrows():
        P = float(row.P)
        idx = id2idx.get(row.sample_id)
        out = None
        if idx is not None:
            scan = imgs[idx].reshape(-1).astype(np.float64)
            if scan.size == SCAN_LEN:
                out = process(scan, P)
        if out is None:
            X.append([0.0] * len(FEAT_NAMES)); NL.append(1600.0 + P / 2); ND.append(1600.0 + P / 2)
            ST.append(np.zeros((2, KM, 2 * WHALF), np.float32)); SM.append(np.zeros(KM, np.float32)); DP.append(None)
        else:
            f, nl, nd, st, sm, dp = out
            X.append(f); NL.append(nl); ND.append(nd); ST.append(st); SM.append(sm); DP.append(dp)
    return (np.nan_to_num(np.array(X, float), nan=0.0, posinf=0.0, neginf=0.0),
            np.array(NL, float), np.array(ND, float),
            np.array(ST, np.float32), np.array(SM, np.float32), DP)

def recompute_drift(NL, DP, lam, lmed):
    ND = NL.copy()
    for i, dp in enumerate(DP):
        if dp is None:
            continue
        kk, xs, ws, mask, c, P = dp
        if mask.sum() < 6:
            continue
        J = fit_sin(kk, xs, ws, mask, P, lam=lam, lmed=lmed)
        f = J["f"]; co = J["coef"]; ka = np.arange(40)
        xp = co[0] + ka * P + co[1] * np.cos(f * ka * P) + co[2] * np.sin(f * ka * P)
        nz = np.where(xp > 1599)[0]
        if len(nz):
            ND[i] = xp[nz[0]]
    return ND

def loc_score(pred, true):
    return float(np.mean(np.exp(-np.abs(pred - true) / 1.5)))

def tree_models():
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    return [RandomForestClassifier(700, n_jobs=-1, random_state=0, min_samples_leaf=2),
            ExtraTreesClassifier(800, n_jobs=-1, random_state=1, min_samples_leaf=2)]

def cnn_available():
    try:
        import torch
        _ = torch.__version__
        return True
    except Exception:
        return False

def make_cnn():
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.c = nn.Sequential(
                nn.Conv2d(2, 40, (3, 5), padding=(1, 2)), nn.BatchNorm2d(40), nn.SiLU(),
                nn.Conv2d(40, 64, (3, 5), padding=(1, 2)), nn.BatchNorm2d(64), nn.SiLU(),
                nn.Conv2d(64, 80, (3, 3), padding=1), nn.BatchNorm2d(80), nn.SiLU(),
                nn.Conv2d(80, 80, (3, 3), padding=1), nn.BatchNorm2d(80), nn.SiLU())
            s.head = nn.Sequential(nn.Linear(80 * 2 + 1, 160), nn.BatchNorm1d(160), nn.SiLU(),
                                   nn.Dropout(.45), nn.Linear(160, 64), nn.BatchNorm1d(64),
                                   nn.SiLU(), nn.Dropout(.3), nn.Linear(64, 3))
        def forward(s, x, m, p):
            h = s.c(x); rm = m.unsqueeze(1).unsqueeze(-1); h = h * rm
            avg = h.sum((2, 3)) / (rm.sum((2, 3)) * h.shape[3] + 1e-6)
            mx = h.amax((2, 3))
            return s.head(__import__("torch").cat([avg, mx, p[:, None]], 1))
    return Net()

def cnn_fit_predict(Atr, Mtr, Ptr, ytr, Ava, Mva, Pva, seeds, epochs):
    import torch, torch.nn as nn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pmu, psd = Ptr.mean(), Ptr.std() + 1e-6
    T = lambda a: torch.tensor(a, device=dev)
    Ai = T(Atr); Mi = T(Mtr); Pi = T((Ptr - pmu) / psd); yi = torch.tensor(ytr, dtype=torch.long, device=dev)
    Av = T(Ava); Mv = T(Mva); Pv = T((Pva - pmu) / psd)
    out = np.zeros((len(Ava), 3))
    n = len(Atr)
    for sd in range(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        net = make_cnn().to(dev)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lf = nn.CrossEntropyLoss(label_smoothing=0.05)
        for e in range(epochs):
            net.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 128):
                ix = perm[i:i + 128]; opt.zero_grad()
                xb = Ai[ix] + torch.randn_like(Ai[ix]) * 0.06
                if np.random.rand() < 0.5:
                    xb = torch.flip(xb, [2])
                lf(net(xb, Mi[ix], Pi[ix]), yi[ix]).backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            out += torch.softmax(net(Av, Mv, Pv), 1).cpu().numpy()
    return out / seeds

def tune_class_weights(oof, y):
    from sklearn.metrics import f1_score
    w = np.ones(3); best = f1_score(y, oof.argmax(1), average="macro")
    for _ in range(50):
        improved = False
        for c in range(3):
            for dl in [0.85, 0.92, 0.97, 1.03, 1.08, 1.15]:
                w2 = w.copy(); w2[c] *= dl
                f = f1_score(y, (oof * w2).argmax(1), average="macro")
                if f > best:
                    best = f; w = w2; improved = True
        if not improved:
            break
    return w

def tune_gating(pred, NL, ND, sig, npt):
    dw = np.zeros(3)
    for c in range(3):
        sel = pred == c
        if sel.sum() == 0:
            dw[c] = 1.0 if c == 2 else (0.0 if c == 0 else 0.85); continue
        bw, bl = 0.0, -1.0
        for w in np.linspace(0, 1, 41):
            l = loc_score(NL[sel] + w * (ND[sel] - NL[sel]), npt[sel])
            if l > bl:
                bl, bw = l, w
        dw[c] = bw
    ba, bb, bl = 3.0, 5.0, -1.0
    for a in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for b in [3.0, 4.0, 5.0, 6.0, 8.0]:
            rel = np.clip(1 - (sig - a) / b, 0, 1)
            l = loc_score(NL + dw[pred] * rel * (ND - NL), npt)
            if l > bl:
                bl, ba, bb = l, a, b
    return dw, ba, bb

def main():
    np.random.seed(0)
    npz = np.load(find_file("images.npz"), allow_pickle=True)
    ids = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in npz["ids"]])
    imgs = npz["images"]
    id2idx = {s: i for i, s in enumerate(ids)}
    tr = pd.read_csv(find_file("train.csv"))
    te = pd.read_csv(find_file("test.csv"))

    Xtr, NLtr, NDtr, STtr, SMtr, DPtr = extract(tr, id2idx, imgs)
    Xte, NLte, NDte, STte, SMte, DPte = extract(te, id2idx, imgs)
    Ptr = tr.P.values.astype(np.float32); Pte = te.P.values.astype(np.float32)
    sig_tr = Xtr[:, FEAT_NAMES.index("sigeps")]
    sig_te = Xte[:, FEAT_NAMES.index("sigeps")]

    _ = (DPtr, DPte)

    lab = {"none": 0, "weak": 1, "strong": 2}
    inv = {v: k for k, v in lab.items()}
    y = np.array([lab[v] for v in tr.regime.values])
    npt = tr.next_pos.values.astype(float)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    tree_oof = np.zeros((len(y), 3))
    for tri, vai in skf.split(Xtr, y):
        probs = np.zeros((len(vai), 3))
        for m in tree_models():
            m.fit(Xtr[tri], y[tri]); probs += m.predict_proba(Xtr[vai])
        tree_oof[vai] = probs / 2.0

    use_cnn = cnn_available()
    if use_cnn:
        cnn_oof = np.zeros((len(y), 3))
        for tri, vai in skf.split(Xtr, y):
            cnn_oof[vai] = cnn_fit_predict(STtr[tri], SMtr[tri], Ptr[tri], y[tri],
                                           STtr[vai], SMtr[vai], Ptr[vai], seeds=3, epochs=75)
        bw, bf = 0.45, -1.0
        for w in np.linspace(0.2, 0.6, 17):
            f = f1_score(y, ((1 - w) * tree_oof + w * cnn_oof).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        oof = (1 - bw) * tree_oof + bw * cnn_oof
    else:
        bw = 0.0; oof = tree_oof

    cw = tune_class_weights(oof, y)
    oof_pred = (oof * cw).argmax(1)
    f1 = f1_score(y, oof_pred, average="macro")
    dw, ga, gb = tune_gating(oof_pred, NLtr, NDtr, sig_tr, npt)
    rel_tr = np.clip(1 - (sig_tr - ga) / gb, 0, 1)
    cv_loc = loc_score(NLtr + dw[oof_pred] * rel_tr * (NDtr - NLtr), npt)
    print("CV macroF1=%.4f loc=%.4f composite=%.2f  (cnn=%s w=%.2f)" %
          (f1, cv_loc, 50 * (f1 + cv_loc), use_cnn, bw))

    tree_te = np.zeros((len(Xte), 3))
    for m in tree_models():
        m.fit(Xtr, y); tree_te += m.predict_proba(Xte)
    tree_te /= 2.0
    if use_cnn:
        cnn_te = cnn_fit_predict(STtr, SMtr, Ptr, y, STte, SMte, Pte, seeds=4, epochs=90)
        te_probs = (1 - bw) * tree_te + bw * cnn_te
    else:
        te_probs = tree_te
    te_pred = (te_probs * cw).argmax(1)

    rel_te = np.clip(1 - (sig_te - ga) / gb, 0, 1)
    next_pos = NLte + dw[te_pred] * rel_te * (NDte - NLte)
    bad = ~np.isfinite(next_pos); next_pos[bad] = NLte[bad]
    bad2 = ~np.isfinite(next_pos); next_pos[bad2] = 1600.0 + te.P.values[bad2] / 2.0

    sub = pd.DataFrame({"sample_id": te.sample_id.values,
                        "regime": [inv[p] for p in te_pred],
                        "next_pos": next_pos})
    os.makedirs("working", exist_ok=True)
    out = os.path.join("working", "submission.csv")
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape, sub.regime.value_counts().to_dict())

if __name__ == "__main__":
    main()
