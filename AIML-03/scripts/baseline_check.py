import sys, json, numpy as np
sys.path.insert(0, "src")
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, evaluate
r = load_parts(sys.argv[1]); tr, va, te = split_by_time(r)
es = build_eval_set(tr, va)
pos_counts = np.bincount(tr.user[tr.positive], minlength=int(r.user.max()) + 1)
n_hist = pos_counts[es.users]
print("val eval users", len(es.users), "| with >=1 train positive:", int((n_hist >= 1).sum()), ">=5:", int((n_hist >= 5).sum()), ">=20:", int((n_hist >= 20).sum()), "| no train positive:", int((n_hist == 0).sum()))
pop = np.bincount(tr.item[tr.positive]); order = np.argsort(-pop, kind="stable")
order = order[pop[order] > 0]
def popularity(u, seen, n):
    head = order[:n + len(seen)]
    return head[~np.isin(head, seen)][:n]
for name, sel in [("all", np.ones(len(es.users), bool)), ("warm>=5", n_hist >= 5), ("cold(0)", n_hist == 0)]:
    print(name, evaluate(popularity, es, es.users[sel], ks=(10, 100)))
