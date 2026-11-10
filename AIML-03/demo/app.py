"""Interactive demo of the recommender (Gradio). Runs the real Recommender on a bundle; no mocked output.

  BUNDLE=./bundle python demo/app.py            # local
  (the hosted copy ships a trimmed bundle without user histories; see docs/deploy_space.md)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gradio as gr

from rec.features import GENRES
from rec.serving import Recommender

rec = Recommender(os.environ.get("BUNDLE", "bundle"))
TITLES = sorted(rec.cat.titles.items(), key=lambda kv: -rec.cat.train_count[rec.lookup[kv[0]]])[:3000]       # 3,000 most-liked films for the picker
CHOICES = [f"{t}  [{m}]" for m, t in TITLES]


def run(liked, genres, ranker, k):
    ids = [int(c.rsplit("[", 1)[1].rstrip("]")) for c in (liked or [])]
    out = rec.recommend(None, int(k), ranker, liked_movie_ids=ids or None, genres=list(genres or []) or None)
    rows = [[i + 1, x["title"], ", ".join(x["genres"]), x.get("score", "")] for i, x in enumerate(out["items"])]
    why = {"cold_start_popularity": "No liked films given: showing the most popular films.",
           "cold_start_popularity_genre": "No liked films given: most popular films in the chosen genres."}.get(out["route"], f"Personalised: two-tower + item-KNN candidates re-ranked by {ranker}.")
    return rows, f"{why}  Route: `{out['route']}` · {out['latency_ms']} ms · model {out['model_version']}"


with gr.Blocks(title="Personalised movie recommender") as demo:
    gr.Markdown("# Personalised movie recommender\nPick films you like and get ranked suggestions, or leave it empty for the cold-start fallback. "
                "MovieLens 25M, two-source retrieval (TensorFlow Recommenders two-tower via FAISS + item-KNN) and a deep ranker. Models were trained on ratings up to mid-2016.")
    with gr.Row():
        with gr.Column():
            liked = gr.Dropdown(CHOICES, multiselect=True, label="Films you like (type to search)")
            genres = gr.CheckboxGroup([g for g in GENRES if g != "(no genres listed)"], label="Genres (used only when no films are picked)")
            ranker = gr.Radio(["dcn", "wide_deep"], value="dcn", label="Ranker")
            k = gr.Slider(5, 30, value=10, step=1, label="How many")
            go = gr.Button("Recommend", variant="primary")
        with gr.Column():
            table = gr.Dataframe(headers=["#", "Title", "Genres", "Ranker logit"], interactive=False, wrap=True, max_height=520, column_widths=["8%", "42%", "36%", "14%"])
            info = gr.Markdown()
    go.click(run, [liked, genres, ranker, k], [table, info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
