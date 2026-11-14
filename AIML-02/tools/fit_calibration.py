"""Fit per-label calibration and thresholds on the NIH sample and check them out-of-fold.

Out-of-fold check of flip-TTA + per-label Platt calibration + F1 thresholds, then the final fit.
Labels with fewer than MIN_POS positives in the fitting data keep their raw score and get no threshold."""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from data_audit import NIH_LABELS
# usage: python tools/fit_calibration.py <manifest.csv> <logits.npz> <out_dir>
MANIFEST, LOGITS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_POS = 20
sig = lambda x: 1/(1+np.exp(-x))
m = pd.read_csv(MANIFEST)
idx = {l:i for i,l in enumerate(NIH_LABELS)}
y = np.zeros((len(m), 14))
for r,s in enumerate(m.finding_labels):
    for l in str(s).split("|"):
        if l in idx: y[r, idx[l]] = 1
z = np.load(LOGITS)
base, tta = z["logits"], (z["logits"]+z["logits_flip"])/2
g = m.patient_id.to_numpy()

def fit(yc, lc):
    lr = LogisticRegression(C=1e4, max_iter=500).fit(lc[:,None], yc)
    a, b = float(lr.coef_[0,0]), float(lr.intercept_[0])
    p = sig(a*lc+b)
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 150)))
    t = max(cand, key=lambda t: f1_score(yc, p>=t, zero_division=0))
    return a, b, float(t)

def ece(y_, p_, bins=10):
    y_, p_ = y_.ravel(), p_.ravel(); e=0; ed=np.linspace(0,1,bins+1)
    for lo,hi in zip(ed[:-1],ed[1:]):
        k=(p_>=lo)&((p_<hi) if hi<1 else (p_<=hi))
        if k.any(): e+=k.mean()*abs(y_[k].mean()-p_[k].mean())
    return e

def oof(seed):
    ug = np.unique(g); perm = np.random.default_rng(seed).permutation(len(ug))
    gid = np.array([perm[np.searchsorted(ug, v)] for v in g])
    P = sig(tta).copy(); T = np.full(y.shape, np.nan)
    for tr, te in GroupKFold(5).split(tta, groups=gid):
        for c in range(14):
            if y[tr,c].sum() < MIN_POS: continue
            a,b,t = fit(y[tr,c], tta[tr,c])
            P[te,c] = sig(a*tta[te,c]+b); T[te,c] = t
    return P, T

def summarize(name, P, flag):
    ok = [c for c in range(14) if 0<y[:,c].sum()<len(y)]
    return dict(method=name,
        macro_auroc=np.mean([roc_auc_score(y[:,c],P[:,c]) for c in ok]),
        macro_auprc=np.mean([average_precision_score(y[:,c],P[:,c]) for c in ok]),
        brier=float(np.mean((P-y)**2)), ece=float(ece(y,P)),
        micro_precision=precision_score(y,flag,average="micro",zero_division=0),
        micro_recall=recall_score(y,flag,average="micro",zero_division=0),
        micro_f1=f1_score(y,flag,average="micro",zero_division=0))

rows=[summarize("served model, threshold 0.5", sig(base), sig(base)>=0.5),
      summarize("flip-TTA, threshold 0.5", sig(tta), sig(tta)>=0.5)]
acc=[]
for sd in range(5):
    P,T = oof(sd)
    flag = np.where(np.isnan(T), P>=0.5, P>=T)
    acc.append(summarize("flip-TTA + calibration + thresholds (out-of-fold)", P, flag))
rows.append({k:(acc[0][k] if k=="method" else float(np.mean([a[k] for a in acc]))) for k in acc[0]})
df = pd.DataFrame(rows); print(df.round(4).to_string(index=False)); df.to_csv(f"{OUT}/calibration_study.csv", index=False)

# final fit on all 999
cal = {}
for c,l in enumerate(NIH_LABELS):
    n = int(y[:,c].sum())
    if n < MIN_POS: cal[l] = {"calibrated": False, "positives": n}
    else:
        a,b,t = fit(y[:,c], tta[:,c]); cal[l] = {"calibrated": True, "a": a, "b": b, "threshold": t, "positives": n}
json.dump({"method": "per-label Platt scaling on flip-TTA logits, F1-optimal threshold on calibrated probability",
           "fit_on": "NIH ChestX-ray14 official-test sample, 999 images from 549 patients", "min_positives": MIN_POS,
           "labels": cal}, open(f"{OUT}/calibration.json","w"), indent=2)
print({l:(v["positives"], v["calibrated"]) for l,v in cal.items()})
