# Presentation Outline

## Slide 1: Title

Predicting Negative Cross-Community Hyperlinks on Reddit Using Temporal Signed Network Features

## Slide 2: Motivation

- Reddit communities interact through hyperlinks.
- Some cross-community relationships become negative-dominant.
- Early prediction can support community monitoring and social media analysis.

## Slide 3: Dataset

- Kaggle `Signed Graphs` mirror, file family `soc-RedditHyperlinks`.
- Original academic source: Stanford SNAP Reddit Hyperlink Network / Kumar et al.
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

## Slide 7: Network and Community Evidence

- Show the readable subreddit network sample colored by detected community.
- Show the community-level negative-ratio chart.
- Show the community-pair negative-ratio heatmap.
- Message: the project is not only a classifier; it first analyzes the signed social network structure.

## Slide 8: Models and Baselines

- Dummy most frequent.
- Dummy prior.
- Historical negative-ratio heuristic.
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.

## Slide 9: Main Result

| Feature Set | Model | Test PR-AUC | ROC-AUC | F1 |
| --- | --- | ---: | ---: | ---: |
| hybrid | Logistic Regression | 0.1840 | 0.7569 | 0.2700 |
| graph_only | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| hybrid | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

## Slide 10: Robustness and Error Analysis

- The best model combines graph/history and text-property features.
- K-core sensitivity checks at `k=3`, `k=5`, and `k=10` keep PR-AUC above the dummy prevalence baseline.
- Threshold analysis shows how precision, recall, and F1 trade off.
- Local contribution cases explain true positives, false positives, and false negatives.

## Slide 11: Limitations and Future Work

- `LINK_SENTIMENT` is a proxy label, not direct proof of real-world conflict.
- K-core filtering restricts the analysis to a denser graph.
- The current task focuses on observed source-target relationships.
- Future work: signed graph embeddings, temporal GNNs, SHAP explanations, robustness to label noise.

## Slide 12: Closing Message

Temporal signed-network features provide practical signals for predicting future negative-dominant relationships between Reddit communities.

## Deck Artifact

The final deck is generated at `docs/final_presentation.pptx` by `scripts/create_presentation.py` and uses figures from `reports/figures/`.
