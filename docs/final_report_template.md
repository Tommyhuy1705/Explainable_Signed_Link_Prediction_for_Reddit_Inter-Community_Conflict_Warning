# Final Report Template

## Title

Predicting Negative Cross-Community Hyperlinks on Reddit Using Temporal Signed Network Features

## Abstract

Briefly state the problem, dataset, temporal signed-network approach, main baselines, best model, and key finding. Emphasize that graph and historical signed features improve negative-link prediction under a strict temporal split.

## 1. Introduction

- Online communities interact through hyperlinks, mentions, and cross-posts.
- Some cross-community links are neutral or positive, while others are negative.
- Early identification of negative cross-community interactions can support community monitoring and moderation.
- This project studies Reddit subreddits as nodes and hyperlinks as directed signed edges.

Suggested contribution claims:

1. A temporal signed-network analysis of Reddit inter-community hyperlinks.
2. A leakage-aware prediction protocol using historical features and future labels.
3. A comparison of graph-only, text-only, and hybrid feature sets.
4. An interpretable feature-importance analysis of future negative interactions.

## 2. Related Work

Cover three areas:

- Social network analysis and centrality measures.
- Signed networks and structural balance theory.
- Misinformation/conflict/negative-interaction prediction in online communities.

Minimum citations:

- SNAP Reddit Hyperlink Network page.
- Kumar et al., "Community Interaction and Conflict on the Web", WWW 2018.
- A network science reference or textbook.
- A recent signed/dynamic graph or misinformation detection survey.

## 3. Dataset

Describe:

- Source: Stanford SNAP Reddit Hyperlink Network.
- Files: title and body hyperlink files.
- Rows: 858,488.
- Time span: 2013-12-31 to 2017-04-30.
- Labels: `-1` negative, `1` positive/neutral.
- Text properties: 86 numeric features.
- Important limitation: label is a derived proxy.

Include:

- Label distribution figure.
- Monthly negative-ratio figure.
- Top negative sources and targets.

## 4. Problem Formulation

Represent the data as a temporal signed directed multigraph:

- Node: subreddit.
- Edge: hyperlink from source subreddit to target subreddit.
- Edge sign: `LINK_SENTIMENT`.
- Edge time: `TIMESTAMP`.

Prediction task:

Given interactions up to history cutoff `t`, predict whether a source-target pair has a negative-dominant interaction in the future label window.

## 5. Methodology

### 5.1 Data Preparation

- Standardize columns.
- Concatenate title/body files.
- Remove self-loops and invalid rows.
- Apply k-core filtering for a denser graph.

### 5.2 Feature Engineering

Graph/history features:

- Interaction count.
- Positive/negative count.
- Negative ratio.
- Reciprocal edge.
- Source/target degree.
- PageRank.
- Betweenness.
- Reciprocity.

Structural balance features:

- Common neighbors.
- Balanced and imbalanced triad pattern counts.

Text features:

- Mean historical `text_property_00` to `text_property_85`.
- Title/body source share.

### 5.3 Temporal Evaluation

Use strict history-label separation:

- Train: history <= 2015-12-31, label window 2016-01-01 to 2016-06-30.
- Validation: history <= 2016-06-30, label window 2016-07-01 to 2016-12-31.
- Test: history <= 2016-12-31, label window 2017-01-01 to 2017-04-30.

### 5.4 Models

- Dummy most frequent.
- Dummy prior.
- Historical negative-ratio heuristic.
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.

Feature-set ablations:

- Text-only.
- Graph-only.
- Hybrid.

## 6. Experiments and Metrics

Main metric:

- PR-AUC.

Supporting metrics:

- ROC-AUC.
- F1 for negative class.
- Macro-F1.
- Precision.
- Recall.
- Balanced accuracy.
- Confusion matrix.

Explain why accuracy is not a headline metric because negative links are rare.

## 7. Results

Include:

- Model comparison table.
- Ablation table.
- PR-AUC bar chart.
- Best-model confusion matrix.
- Feature-importance plot.

Discuss:

- Whether hybrid outperforms graph-only and text-only.
- Which features are most predictive.
- Tradeoff between precision and recall after threshold tuning.

## 8. Discussion

Interpret the model:

- Prior negative history and source/target signed degrees may indicate persistent inter-community tension.
- Structural-balance features capture local signed-neighborhood context.
- Text-property features add content-side signals but may be weaker than historical graph features for early warning.

## 9. Limitations

- `LINK_SENTIMENT` is a proxy label.
- Negative hyperlinks do not prove real-world conflict or harassment.
- Dataset is historical, from 2014 to 2017.
- K-core filtering restricts analysis to a denser subset.
- The current approach predicts future source-target relationships, not brand-new source-target pairs with no history.

## 10. Conclusion and Future Work

Conclusion:

- Signed temporal network features provide useful signals for negative cross-community hyperlink prediction.
- The hybrid model and ablation study support the value of combining network history, signed structure, and text properties.

Future work:

- True early-warning model for unseen pairs.
- Temporal graph neural networks.
- Signed graph embeddings.
- Explainability with SHAP.
- Robustness to label noise.
