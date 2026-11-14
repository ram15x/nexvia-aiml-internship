---
title: Personalised Movie Recommender
emoji: 🎬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Two-stage movie recommender on MovieLens 25M
---

Interactive demo of a two-stage movie recommender (two-tower + item-KNN retrieval, DCN-v2 / Wide & Deep ranking).
Pick films you like to get ranked suggestions; leave the picker empty for the cold-start fallback.

Code, results and limitations: https://github.com/ram15x/nexvia-aiml-internship/tree/main/AIML-03

Models were trained on MovieLens 25M ratings up to mid-2016, so recent films are not recommended.
MovieLens data is from GroupLens (https://grouplens.org/datasets/movielens/); this Space ships no per-user ratings.
