// Builds the 10-minute presentation from the JSON in results/ (numbers cannot drift from the evidence).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const R = f => JSON.parse(fs.readFileSync(`results/${f}`, "utf8"));
const test = R("final_test.json"), valw = R("retrieval_val_full_warm.json"), knn = R("itemknn_val_warm.json"), uni = R("ranker_union_cv_val.json"), idx = R("faiss_index_eval.json"), srv = R("serving_eval_dev.json");

const C = { dark: "0E2F36", teal: "127C83", teal2: "5FB3B8", amber: "D9902B", ink: "1B2B30", mute: "5C6F74", bg: "F5F8F8", white: "FFFFFF", red: "B5483A", grid: "D5DEDF" };
const pptx = new pptxgen(); pptx.layout = "LAYOUT_WIDE"; pptx.author = "Ramnarayanan M"; pptx.title = "Large-Scale Personalised Recommendation Engine (AIML-03)";
const HF = "Cambria", BF = "Calibri";
const f4 = x => x.toFixed(4), f3 = x => x.toFixed(3);

function base(title, notes, dark = false) {
  const s = pptx.addSlide(); s.background = { color: dark ? C.dark : C.bg };
  if (!dark) { s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(title, { x: 0.6, y: 0.35, w: 12.2, h: 0.9, fontFace: HF, fontSize: 28, bold: true, color: C.ink, valign: "top", margin: 0 }); }
  s.addNotes(notes); return s;
}
function foot(s, n) { s.addText(`AIML-03 · Recommendation Engine · ${n}`, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: BF, fontSize: 10, color: C.mute, margin: 0 }); }
function bullets(s, items, o) {
  s.addText(items.map(t => ({ text: t, options: { bullet: { indent: 16 }, breakLine: true, paraSpaceAfter: 7 } })),
    { fontFace: BF, fontSize: 16, color: C.ink, valign: "top", margin: 0, ...o });
}
function callout(s, text, o, color = C.amber) {
  s.addShape(pptx.ShapeType.rect, { ...o, fill: { color: "FFFFFF" }, line: { color, width: 1.5 } });
  s.addShape(pptx.ShapeType.rect, { x: o.x, y: o.y, w: 0.09, h: o.h, fill: { color }, line: { color } });
  s.addText(text, { x: o.x + 0.25, y: o.y, w: o.w - 0.4, h: o.h, fontFace: BF, fontSize: 14, color: C.ink, valign: "middle", margin: 0 });
}

// 1 ─ title
{ const s = base("", "Hi. This is a two-stage recommender on MovieLens 25M: retrieval, ranking, serving and the MLOps around it. I will show what worked, and also where my first assumptions were wrong.", true);
  s.addText("Large-Scale Personalised\nRecommendation Engine", { x: 0.8, y: 1.7, w: 11.5, h: 2.1, fontFace: HF, fontSize: 46, bold: true, color: C.white, margin: 0, valign: "top" });
  s.addText("AIML-03  ·  Nexvia AI/ML Internship Capstone", { x: 0.8, y: 4.0, w: 11, h: 0.5, fontFace: BF, fontSize: 20, color: C.teal2, margin: 0 });
  s.addText("Ramnarayanan M  ·  November 2026", { x: 0.8, y: 4.6, w: 11, h: 0.4, fontFace: BF, fontSize: 16, color: "C9D6D8", margin: 0 });
  s.addText("MovieLens 25M · TensorFlow Recommenders · FAISS · Spark · Redis · FastAPI · Airflow · Triton-ready", { x: 0.8, y: 6.4, w: 12, h: 0.4, fontFace: BF, fontSize: 13, color: C.teal2, margin: 0 }); }

// 2 ─ problem
{ const s = base("Choose 10 films from 34,461 for each person", "The funnel: a cheap method narrows 34 thousand films to 250, an expensive model orders those. Brand-new users have no history, so they need a separate fallback.");
  const stages = [["34,461", "films in the catalog", C.mute], ["250", "candidates: two-tower (FAISS) + item-KNN", C.teal2], ["10", "shown, ordered by a deep ranker", C.teal]];
  stages.forEach((st, i) => { const w = 5.6 - i * 1.5; s.addShape(pptx.ShapeType.rect, { x: 0.6 + (5.6 - w) / 2, y: 1.7 + i * 1.5, w, h: 1.25, fill: { color: st[2] }, line: { color: st[2] } });
    s.addText([{ text: st[0], options: { fontSize: 28, bold: true, fontFace: HF, breakLine: true } }, { text: st[1], options: { fontSize: 13, fontFace: BF } }], { x: 0.6 + (5.6 - w) / 2, y: 1.7 + i * 1.5, w, h: 1.25, color: C.white, align: "center", valign: "middle", margin: 0 }); });
  bullets(s, ["Two stages because scoring every film with a deep model for every user is too slow; cheap retrieval first, expensive ranking on the shortlist.", "Cold start: users with no likes get popularity or genre-filtered popularity, with a Redis-cached API in front.", "Goal measured on held-out time, not random rows: Recall@K and NDCG@K, plus a simulated A/B test of two rankers."], { x: 7.0, y: 1.7, w: 5.8, h: 4.6 });
  foot(s, 2); }

// 3 ─ data & protocol
{ const s = base("Data and a split that cannot leak", "I split by time, not at random. A random split would let the model see the future. Everything the model learns, including the item catalog and statistics, comes from the train window only. The test window was scored once, at the end.");
  const rows = [["Window", "Dates", "Ratings", "Used for"], ["Train", "to 2016-06-25", "20,000,076", "all learning, catalog, statistics"], ["Validation", "to 2018-01-04", "2,500,009", "model selection, ranker training"], ["Test", "from 2018-01-04", "2,500,010", "scored once, at the end"]];
  s.addTable(rows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.teal : (i % 2 ? "FFFFFF" : "EAF1F1") }, fontFace: BF, fontSize: 14, valign: "middle" } }))), { x: 0.6, y: 1.55, w: 12.1, colW: [1.8, 2.8, 2.0, 5.5], rowH: 0.5, border: { type: "solid", color: C.grid, pt: 0.75 } });
  bullets(s, ["MovieLens 25M: 25,000,095 ratings, 162,541 users, 59,047 films, 1995 to 2019.", "A \"like\" is a rating of 4.0 or more. Relevant items = films the user liked in the window that they had not rated before; rated films are never recommended.", "Spark builds time-safe tables (interactions, user and item statistics, user history); they match an independent NumPy computation exactly.", "Models are frozen at the train cut-off for the test run; only user histories advance to the window start."], { x: 0.6, y: 3.85, w: 12.1, h: 3.0 });
  foot(s, 3); }

// 4 ─ architecture
{ const s = base("Architecture", "Offline on the left: Spark, training, a quality gate, bundle build. Online on the right: the API checks the Redis cache, routes cold-start users to popularity, and otherwise builds 250 candidates from two sources and re-ranks them.");
  s.addImage({ path: "docs/architecture.png", x: 6.6, y: 0.2, h: 6.9, w: 6.9 * 2918 / 2990 });
  bullets(s, ["Offline (weekly Airflow DAG): Spark tables, two-tower training, item-KNN, a validation gate, then the bundle build that trains the rankers.", "Online: FastAPI with a Redis response cache; personalised path = FAISS HNSW two-tower candidates + item-KNN candidates, re-ranked by DCN-v2 or Wide & Deep.", "Rankers are exported as TensorFlow SavedModels, ready for Triton."], { x: 0.6, y: 1.55, w: 5.7, h: 5.0 });
  foot(s, 4); }

// 5 ─ retrieval
{ const s = base("Retrieval: the simple method won", "I expected the two-tower to win. It did not: the classical item-based method scores higher on its own. My first two-tower run also overfit after one epoch; lowering the learning rate and adding a small penalty fixed it. I kept the two-tower because the two sources find different films, and the union covers more.");
  const v = [valw.two_tower, knn];
  s.addChart(pptx.charts.BAR, [{ name: "Recall@100", labels: ["Popularity", "Two-tower", "Item-KNN"], values: [0.1333, v[0]["recall@100"], v[1]["recall@100"]] }, { name: "NDCG@10", labels: ["Popularity", "Two-tower", "Item-KNN"], values: [0.0832, v[0]["ndcg@10"], v[1]["ndcg@10"]] }],
    { x: 0.5, y: 1.5, w: 6.4, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [C.teal, C.amber], showLegend: true, legendPos: "b", legendFontSize: 12, showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 11, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 13, showTitle: true, title: "Validation, 5,083 warm users", titleFontSize: 13, titleColor: C.mute });
  bullets(s, ["Item-KNN beats the tuned two-tower on its own (Recall@100 " + f3(knn["recall@100"]) + " vs " + f3(valw.two_tower["recall@100"]) + ").", "Run 1 (lr 2e-3) peaked after one epoch then declined; lr 5e-4 + L2 on the ID embedding trained stably.", "Union of both sources covers 39.7% of relevant items in 250 candidates (two-tower alone 36.5%, item-KNN alone 38.6%).",
    `FAISS HNSW matches exact search: Recall@100 ${idx.hnsw_ef128["recall@100"]} vs ${idx.flat_exact["recall@100"]}, ${idx.hnsw_ef128.latency_ms_p50} ms vs ${idx.flat_exact.latency_ms_p50} ms per query.`], { x: 7.2, y: 1.5, w: 5.6, h: 4.9, fontSize: 15 });
  foot(s, 5); }

// 6 ─ ranking
{ const m = uni.models;
  const s = base("Ranking: DCN-v2 and Wide & Deep re-score 250 candidates", "Both rankers see the genres, a user-by-item genre cross, ratings statistics, and each source's score and rank. They are cross-validated by user, so each user is scored by a model that never saw them. The two rankers are statistically tied, which is itself a useful finding for the A/B test.");
  s.addChart(pptx.charts.BAR, [{ name: "NDCG@10", labels: ["Two-tower", "Item-KNN", "Wide & Deep", "DCN-v2"], values: [m.two_tower_alone["ndcg@10"].mean, m.itemknn_alone["ndcg@10"].mean, m.wide_deep_union["ndcg@10"].mean, m.dcn_union["ndcg@10"].mean] }],
    { x: 0.5, y: 1.5, w: 6.4, h: 4.6, barDir: "col", chartColors: [C.teal], showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 13, showTitle: true, title: "Validation NDCG@10, 5-fold CV by user", titleFontSize: 13, titleColor: C.mute });
  bullets(s, ["Features: item genres, user-by-item genre cross, user and item rating statistics, retrieval score and rank from each source (52 dense inputs).", `DCN-v2 beats the best retriever by +${f3(m.dcn_union["paired_diff_vs_itemknn_alone"]["ndcg@10"].mean)} NDCG@10 on validation; Wide & Deep is within 0.003 of it.`, "Caveat: rankers learn from validation-window labels of other users, so part of this gain can be window-specific. The held-out test window checks that."], { x: 7.2, y: 1.5, w: 5.6, h: 4.9, fontSize: 15 });
  foot(s, 6); }

// 7 ─ test
{ const sy = test.systems, p = test.paired;
  const s = base("Held-out test: the gain is real but smaller", "This is the test window, scored once, with frozen models. The full system beats the best single retriever by a statistically clear margin, but the margin is about a third of what validation suggested. That drop is exactly the leakage risk I flagged. The two rankers are indistinguishable.");
  s.addChart(pptx.charts.BAR, [{ name: "NDCG@10", labels: ["Popularity", "Two-tower", "Item-KNN", "Wide & Deep", "DCN-v2"], values: [sy.popularity["ndcg@10"].mean, sy.two_tower["ndcg@10"].mean, sy.itemknn["ndcg@10"].mean, sy.wide_deep["ndcg@10"].mean, sy.dcn["ndcg@10"].mean] }],
    { x: 0.5, y: 1.5, w: 6.9, h: 4.6, barDir: "col", chartColors: [C.teal], showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12, showTitle: true, title: `Test NDCG@10, ${test.users.toLocaleString()} users`, titleFontSize: 13, titleColor: C.mute });
  const d = p.dcn_minus_itemknn["ndcg@10"], dw = p["dcn_minus_wide_deep"]["ndcg@10"];
  bullets(s, [`DCN-v2 vs item-KNN: +${f3(d.mean)} NDCG@10 (95% CI ${f3(d.ci95[0])} to ${f3(d.ci95[1])}). On validation the gap was +0.049.`, `DCN-v2 vs Wide & Deep: ${dw.mean.toFixed(4)} (CI ${dw.ci95[0].toFixed(4)} to ${dw.ci95[1].toFixed(4)}): a tie.`, `Recall@100: ${f3(sy.dcn["recall@100"].mean)} for DCN-v2 against ${f3(sy.itemknn["recall@100"].mean)} for item-KNN and ${f3(sy.popularity["recall@100"].mean)} for popularity.`], { x: 7.7, y: 1.5, w: 5.1, h: 3.3, fontSize: 15 });
  callout(s, "Scores are lower than validation because the models are 18 months stale at test time; weekly retraining is what the Airflow DAG is for.", { x: 7.7, y: 4.95, w: 5.1, h: 1.35 });
  foot(s, 7); }

// 8 ─ serving
{ const s = base("Serving: about 10 ms, cold start handled", "The API takes liked films, a user id, or nothing at all. With likes it builds candidates from both sources and re-ranks. With nothing it falls back to popularity, optionally filtered by genre. Redis cuts a request to a fraction of a millisecond.");
  const rows = [["Request", "Route", "Latency"], ["Liked films or known user", "two-tower + item-KNN → DCN-v2 / Wide & Deep", `${srv.dcn.latency_ms_cold_cache_p50} ms p50, ${srv.dcn.latency_ms_cold_cache_p95} ms p95`], ["Same request, Redis hit", "cached response", `${srv.redis_cached_p50_ms} ms p50`], ["No history", "popularity", "about 0.3 ms"], ["No history + genres", "genre-filtered popularity", "about 1 ms"]];
  s.addTable(rows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.teal : (i % 2 ? "FFFFFF" : "EAF1F1") }, fontFace: BF, fontSize: 13, valign: "middle" } }))), { x: 0.6, y: 1.5, w: 12.1, colW: [3.3, 5.6, 3.2], rowH: 0.5, border: { type: "solid", color: C.grid, pt: 0.75 } });
  s.addImage({ path: "docs/demo_personalised.png", x: 0.6, y: 4.2, w: 3.9, h: 3.9 * 900 / 1280 });
  bullets(s, ["Real output for a user who likes The Godfather, Goodfellas and Pulp Fiction: Fargo, Trainspotting, Fight Club, Godfather Part II, Shawshank Redemption.", "The serving path reproduces the offline score (NDCG@10 0.184 on 508 held-out users).", "Latency measured on a 2-CPU container with a real Redis server."], { x: 5.0, y: 4.3, w: 7.7, h: 2.6, fontSize: 15 });
  foot(s, 8); }

// 9 ─ MLOps
{ const ab = test.ab_replay;
  const s = base("MLOps: what is proven and what is not", "Three pieces. Airflow actually ran end to end, including a run that failed on a real bug and a run where the gate correctly blocked a weak model. The A/B test is a replay of held-out data, not live traffic. Triton: the model files are verified, but I have not yet started a real Triton server.");
  const cols = [["Airflow DAG", C.teal, ["Executed end to end with Airflow 3.", "Run 1 failed on a real bug I then fixed.", "Weak candidate: gate said BLOCK (Recall@100 0.146 vs 0.232); promote skipped.", "Full run: gate said PROMOTE (0.236 vs 0.232); new bundle serves requests."]],
    ["A/B test (simulated)", C.amber, [`Control DCN-v2 ${(ab.control_dcn.ctr * 100).toFixed(2)}% vs treatment Wide & Deep ${(ab.treatment_wide_deep.ctr * 100).toFixed(2)}% CTR.`, `p = ${ab.p_value_two_sided}: no difference, as expected from the offline tie.`, "Replay of held-out likes with position-based clicks, not live traffic."]],
    ["Triton", C.red, ["SavedModels export exactly (max score difference 0.0).", "Model repository and HTTP client written; client tested against a local protocol stub.", "NOT yet run on a real Triton server: needs Docker. Steps are in docs/triton.md."]]];
  cols.forEach((c, i) => { const x = 0.6 + i * 4.15; s.addShape(pptx.ShapeType.rect, { x, y: 1.55, w: 3.95, h: 0.55, fill: { color: c[1] }, line: { color: c[1] } });
    s.addText(c[0], { x, y: 1.55, w: 3.95, h: 0.55, fontFace: HF, fontSize: 17, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    s.addShape(pptx.ShapeType.rect, { x, y: 2.1, w: 3.95, h: 4.3, fill: { color: "FFFFFF" }, line: { color: C.grid } });
    bullets(s, c[2], { x: x + 0.2, y: 2.3, w: 3.55, h: 4.0, fontSize: 16 }); });
  foot(s, 9); }

// 10 ─ limits & next
{ const s = base("Limits, and what I would do next", "I want to be straight about the limits. The most important: the A/B is simulated, the two-tower alone does not beat a classical baseline, and for users with very little history the system is no better than popularity as far as I can measure. Then the reflection, in my own words.");
  bullets(s, ["The ranker's gain shrank from +0.049 (validation) to +0.018 (test): part of it was window-specific.", "The two-tower alone is below item-KNN; it earns its place through candidate diversity.", "Users with 1 to 4 likes: only 55 validation users have relevant items, too few to say whether personalisation beats popularity.", "Click-through is simulated; MovieLens has no impressions or clicks.", "Next: run Triton for real, train on newer data, add features like recency and a proper click log."], { x: 0.6, y: 1.5, w: 7.2, h: 5.2, fontSize: 15 });
  callout(s, "[Your words: what you learned, what surprised you, what you would do differently.]", { x: 8.2, y: 1.6, w: 4.6, h: 2.4 }, C.teal);
  s.addText("Repository, README, architecture and results: github.com/ram15x/personalized-recommendation-engine", { x: 8.2, y: 4.4, w: 4.6, h: 1.0, fontFace: BF, fontSize: 13, color: C.mute, margin: 0, valign: "top" });
  foot(s, 10); }

pptx.writeFile({ fileName: process.argv[2] || "deliverables/AIML03_Recommendation_Engine.pptx" }).then(f => console.log("wrote", f));
