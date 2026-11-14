# Reflection: AIML-03, personalised movie recommendation engine

**Ramnarayanan M, Nexvia Technologies AI/ML internship**

## What I owned

I owned the recommender end to end: the time-based data split on MovieLens 25M, the Spark feature tables, the two retrieval sources, the DCN-v2 and Wide & Deep rankers, the Redis-cached FastAPI service, the Triton setup, the Airflow retraining pipeline, the simulated A/B replay and the demo. I also made the calls on what to measure and how to report it.

## What I built

A two-stage system. A TensorFlow Recommenders two-tower model searched through FAISS, plus item-KNN, narrows 34,461 films to about 250 candidates. A DCN-v2 ranker orders them and returns the top 10, with popularity as the fallback for new users. A weekly Airflow job retrains the models and only promotes a new one if it clears a quality gate.

## What I learned

- **Split by time, not at random.** The models are trained once, frozen at the cut-off and scored a single time on the later test window. It is a stricter test, and it gave me numbers I trust: NDCG@10 of 0.1272 against 0.0728 for popularity and 0.1091 for the best single retriever, a gain of +0.018 (95% CI 0.015 to 0.021).
- **Keep the simple baseline.** Item-KNN beat my neural two-tower model on its own. Rather than drop it, I combined the two, since they find different films and together cover 39.7% of what a user later likes.
- **Report the smaller number.** The ranker's gain looked like +0.049 on validation but was +0.018 on the later test window. The test figure is the one I quote.
- **Orchestration finds real bugs.** The first Airflow run failed because a folder did not exist yet, a bug I would not have found by running scripts by hand. The next run was blocked by the quality gate, as it should be, and the full run was promoted.
- **Offline and online must match.** The serving path reproduces the offline NDCG@10 of 0.184, and the rankers give the same scores on a real Triton server (maximum difference 9.5e-7, same top 10). Median latency is 10.6 ms uncached and 0.15 ms on a cache hit.

## Challenges

My first training run overfit, and I fixed that before reporting anything. The variety re-rank I added gave only a small gain and leaned toward popular films, so I kept it optional and off by default. For users with one to four likes, only 55 validation users had relevant films, so I could not say whether personalisation beats popularity there, and the write-up says so.

## Limits I want to be upfront about

The A/B result is a simulated replay of held-out likes, so it shows the mechanism works and says nothing about real users. The models are frozen about 18 months before the test window, and MovieLens is ratings from a single domain.

## What I would do next

Collect real clicks through the A/B setup, study users with only a few likes, and try a larger catalogue where the choice of search index matters more.
