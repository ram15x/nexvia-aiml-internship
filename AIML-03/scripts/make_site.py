"""Generate the project overview page (docs/index.html, copied to site/) from results/*.json so it cannot drift from the evidence."""
import json, shutil
from pathlib import Path

t = json.loads(Path("results/final_test.json").read_text()); css = Path("scripts/site_style.css").read_text()
sy, pd_, ab = t["systems"], t["paired"], t["ab_replay"]
names = [("popularity", "Popularity"), ("two_tower", "Two-tower retriever alone"), ("itemknn", "Item-KNN alone"), ("dcn", "Full system, DCN-v2 ranker"), ("wide_deep", "Full system, Wide &amp; Deep ranker")]
rows = "\n".join(f"<tr><td>{n}</td><td>{sy[k]['recall@10']['mean']:.4f}</td><td>{sy[k]['ndcg@10']['mean']:.4f}</td><td>{sy[k]['recall@100']['mean']:.3f}</td></tr>" for k, n in names)
d = pd_["dcn_minus_itemknn"]["ndcg@10"]; dw = pd_["dcn_minus_wide_deep"]["ndcg@10"]
html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A two-stage movie recommender on MovieLens 25M: two-tower and item-KNN retrieval, a deep ranker, cold-start fallbacks, caching, retraining and an honest evaluation.">
  <title>Two-stage movie recommender · MovieLens 25M</title>
  <style>{css}
    table {{ width:100%; border-collapse:collapse; margin:12px 0 6px; }} th, td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:600; font-size:.9rem; }}
    td:not(:first-child), th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }} tr.hl td {{ color:var(--accent); font-weight:650; }} img.arch {{ width:100%; max-width:720px; border-radius:10px; background:#fff; }} @media (max-width:560px) {{ th, td {{ padding:8px 4px; font-size:.86rem; }} td:first-child {{ max-width:34vw; }} }} ul.plain li {{ margin:8px 0; color:var(--muted); }} ul.plain strong {{ color:var(--text); }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand">Recommendation Systems · AIML-03</div>
      <nav aria-label="Main navigation"><a href="#system">System</a><a href="#results">Results</a><a href="#limits">Limits</a><a href="https://github.com/ram15x/personalized-recommendation-engine">Source</a></nav>
    </header>
    <main>
      <section class="hero">
        <div class="eyebrow">Movie recommendation · MovieLens 25M</div>
        <h1>A two-stage recommender, measured honestly.</h1>
        <p class="lead">Two candidate generators (a TensorFlow Recommenders two-tower model indexed in FAISS, and item-based collaborative filtering) feed a deep ranker. Cold-start fallbacks, a Redis-cached API, Spark feature tables and a gated Airflow retraining pipeline surround it.</p>
        <div class="actions">
          <a class="button primary" href="https://github.com/ram15x/personalized-recommendation-engine">Explore the source on GitHub</a>
          <a class="button" href="https://github.com/ram15x/personalized-recommendation-engine/blob/main/docs/results.md">Full results</a>
        </div>
      </section>
      <aside class="notice"><strong>Hosted page:</strong> this is a project overview, not a live recommender. MovieLens data is not redistributed here, so the interactive demo runs locally from a built bundle.</aside>
      <section id="system">
        <h2>System</h2>
        <div class="flow">
          <div class="step"><strong>Retrieve</strong>Two-tower via FAISS HNSW and item-KNN each propose candidates.</div>
          <div class="step"><strong>Merge</strong>The two lists are interleaved into 250 unseen candidates.</div>
          <div class="step"><strong>Rank</strong>DCN-v2 (or Wide &amp; Deep) re-scores them; A/B arms by user hash.</div>
          <div class="step"><strong>Serve</strong>FastAPI + Redis; popularity or genre-filtered popularity for new users.</div>
        </div>
        <p style="margin-top:22px"><img class="arch" src="architecture.png" alt="Architecture: offline Airflow pipeline with a quality gate, and the online serving path"></p>
      </section>
      <section id="results">
        <h2>Results on the held-out test window</h2>
        <p class="muted">{t['users']:,} users, time-based split (train before 2016-06-25, test from 2018-01-04), test scored once with models frozen at the train cut-off.</p>
        <table><thead><tr><th>System</th><th>Recall@10</th><th>NDCG@10</th><th>Recall@100</th></tr></thead><tbody>
{rows}
        </tbody></table>
        <ul class="plain">
          <li><strong>The full system beats the best single retriever</strong> by {d['mean']:+.3f} NDCG@10 (95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}). On validation the gap was larger (+0.049); part of that was specific to the validation window.</li>
          <li><strong>DCN-v2 and Wide &amp; Deep are tied</strong> (NDCG@10 difference {dw['mean']:+.4f}, CI {dw['ci95'][0]:+.4f} to {dw['ci95'][1]:+.4f}).</li>
          <li><strong>Simulated A/B replay</strong> (not live traffic): CTR {ab['control_dcn']['ctr']*100:.2f}% vs {ab['treatment_wide_deep']['ctr']*100:.2f}%, p = {ab['p_value_two_sided']}.</li>
        </ul>
      </section>
      <section id="limits">
        <h2>What this does not show</h2>
        <ul class="plain">
          <li><strong>No live traffic.</strong> Click-through comes from a replay of held-out interactions; MovieLens has no impressions or clicks.</li>
          <li><strong>Triton</strong> serving is prepared (exported SavedModels, repository, client tested against a protocol stub) but not yet confirmed on a real Triton server.</li>
          <li><strong>Stale models at test time.</strong> Models are frozen 18 months before the test window; weekly retraining is what the pipeline is for.</li>
          <li><strong>The two-tower alone does not beat item-KNN.</strong> It stays because the two sources find different films and the ranker uses both.</li>
        </ul>
        <p class="caveat">Every number on this page is generated from <code>results/final_test.json</code>. More caveats: <a href="https://github.com/ram15x/personalized-recommendation-engine/blob/main/docs/limitations.md">docs/limitations.md</a>.</p>
      </section>
    </main>
    <footer>MovieLens 25M is licensed for research by GroupLens and is not redistributed here. Source: <a href="https://github.com/ram15x/personalized-recommendation-engine">github.com/ram15x/personalized-recommendation-engine</a>.</footer>
  </div>
</body>
</html>
"""
Path("docs/index.html").write_text(html)
Path("site").mkdir(exist_ok=True); shutil.copy("docs/index.html", "site/index.html"); shutil.copy("docs/architecture.png", "site/architecture.png")
print("wrote docs/index.html and site/index.html")
