# Presentation Outline

## Slide 1: Title

Predicting Negative Cross-Community Hyperlinks on Reddit Using Temporal Signed Network Features

## Slide 2: Motivation

- Reddit communities interact through hyperlinks.
- Some cross-community relationships become negative-dominant.
- Early prediction can support community monitoring and social media analysis.

## Slide 3: Dataset

- Stanford SNAP Reddit Hyperlink Network.
- 858,488 hyperlink records.
- 67,180 unique subreddits in the raw combined data.
- Time span: 2013-12-31 to 2017-04-30.
- Each row has `LINK_SENTIMENT` and 86 text-property values.

## Slide 4: Graph Formulation

- Node: subreddit.
- Directed edge: hyperlink from source subreddit to target subreddit.
- Edge sign: positive/neutral or negative.
- Target: future negative-dominant source-target relationship.

## Slide 5: Anti-Leakage Temporal Design

| Split | History Window | Label Window |
| --- | --- | --- |
| Train | <= 2015-12-31 | 2016-01-01 to 2016-06-30 |
| Validation | <= 2016-06-30 | 2016-07-01 to 2016-12-31 |
| Test | <= 2016-12-31 | 2017-01-01 to 2017-04-30 |

## Slide 6: Feature Groups

- Pair history: positive/negative counts, negative ratio, sentiment balance.
- Network centrality: degree, PageRank, betweenness, reciprocity.
- Community structure: clustering coefficient, community size, community negativity.
- Structural balance: signed local-neighborhood patterns.
- Text properties: 86 SNAP text-property features.

## Slide 7: Models and Baselines

- Dummy most frequent.
- Dummy prior.
- Historical negative-ratio heuristic.
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.

## Slide 8: Main Result

| Feature Set | Model | Test PR-AUC | ROC-AUC | F1 |
| --- | --- | ---: | ---: | ---: |
| hybrid | Logistic Regression | 0.1840 | 0.7569 | 0.2700 |
| graph_only | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| hybrid | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

## Slide 9: Interpretation

- The best model combines graph/history and text-property features.
- Graph-only models are close to hybrid, so network structure is a strong signal.
- Text-only models are weaker, but text properties help inside the hybrid model.
- Structural balance is useful for explanation, but its predictive gain is small.

## Slide 10: Limitations and Future Work

- `LINK_SENTIMENT` is a proxy label, not direct proof of real-world conflict.
- K-core filtering restricts the analysis to a denser graph.
- The current task focuses on observed source-target relationships.
- Future work: signed graph embeddings, temporal GNNs, SHAP explanations, robustness to label noise.

## Slide 11: Closing Message

Temporal signed-network features provide practical signals for predicting future negative-dominant relationships between Reddit communities.
