# Predicting Negative Cross-Community Hyperlinks on Reddit Using Temporal Signed Network Features

## Abstract

This project studies negative inter-community relationships on Reddit using the Stanford SNAP Reddit Hyperlink Network. Subreddits are represented as nodes and hyperlinks as directed signed edges. Given historical interactions up to a cutoff date, the task is to predict whether a source-target subreddit pair becomes negative-dominant in a future label window. A relationship is labeled negative when future negative hyperlinks outnumber future positive/neutral hyperlinks. The pipeline compares temporal graph features, text-property features, hybrid features, and several baselines under a strict anti-leakage temporal split. The latest verified `.venv` run shows that the best model is a hybrid Logistic Regression model with test PR-AUC 0.1840 and ROC-AUC 0.7569, outperforming the historical negative-ratio baseline and dummy baselines.

## 1. Introduction

Online communities interact through hyperlinks, mentions, and cross-community references. Some of these interactions are positive or neutral, while others are negative. Detecting early signals of negative inter-community relationships can support social media monitoring, community health analysis, and moderation research.

This project frames Reddit as a temporal signed directed network. Each subreddit is a node, each hyperlink from one subreddit to another is a directed edge, and the edge sign is provided by `LINK_SENTIMENT`. The course-project scope is negative-dominant relationship prediction, not direct proof of real-world conflict, harassment, or raids.

## 2. Dataset

The project uses the Stanford SNAP Reddit Hyperlink Network:

| File | Rows | Negative Rows | Negative Ratio |
|---|---:|---:|---:|
| `soc-redditHyperlinks-body.tsv` | 286,561 | 21,070 | 7.35% |
| `soc-redditHyperlinks-title.tsv` | 571,927 | 61,140 | 10.69% |
| Combined | 858,488 | 82,210 | 9.58% |

The local audit confirms that every row has the expected core fields and exactly 86 numeric text-property values. The observed timestamp range is from 2013-12-31 to 2017-04-30. The body and title files are concatenated, and `dataset_source` records whether a row came from the title or body file.

## 3. Problem Formulation

Let each interaction be represented as `(source_subreddit, target_subreddit, timestamp, sign, text_properties)`. For each temporal split, features are computed only from interactions before the history cutoff. Labels are computed from a disjoint future window.

The target is:

- `negative_label = 1` if `future_negative_count > future_positive_count`.
- `negative_label = 0` otherwise.

This makes the task stricter than detecting a single negative edge. It predicts whether the future relationship between a pair of subreddits is negative-dominant.

## 4. Methodology

The pipeline has four phases:

1. Data preparation: load raw TSV files, standardize columns, parse timestamps, concatenate title/body data, and apply k-core filtering for a denser graph.
2. Network construction: build a directed signed multigraph where subreddits are nodes and hyperlinks are signed directed edges.
3. Feature engineering: create temporal graph, pair-history, community, structural-balance, and text-property features.
4. Modeling and evaluation: train models under strict temporal splits and compare metrics on validation and test windows.

The main feature groups are:

| Group | Examples |
|---|---|
| Pair history | interaction count, positive count, negative count, negative ratio, sentiment balance |
| Node/network | in-degree, out-degree, signed degree, PageRank, betweenness, reciprocity |
| Community/clustering | clustering coefficient, community size, same-community flag, community negativity gap |
| Structural balance | common neighbors, signed triad-pattern counts |
| Text properties | `text_property_00` to `text_property_85`, title/body source indicators |

The strict temporal split is:

| Split | History Window | Label Window | Rows |
|---|---|---|---:|
| Train | <= 2015-12-31 | 2016-01-01 to 2016-06-30 | 25,045 |
| Validation | <= 2016-06-30 | 2016-07-01 to 2016-12-31 | 26,450 |
| Test | <= 2016-12-31 | 2017-01-01 to 2017-04-30 | 24,185 |

## 5. Models and Evaluation

The latest verified run compares 41 model/feature-set rows across:

- Dummy most frequent.
- Dummy prior.
- Historical negative-ratio heuristic.
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.

The feature-set ablations are:

- `history_only`
- `text_only`
- `graph_only`
- `graph_no_balance`
- `hybrid`
- `hybrid_no_balance`

Because negative-dominant relationships are rare, accuracy is not the headline metric. The main metric is PR-AUC, with ROC-AUC, F1, Macro-F1, precision, recall, balanced accuracy, and confusion counts as supporting metrics.

## 6. Results

Best model by test PR-AUC:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | F1 | Macro-F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `hybrid` | Logistic Regression | 0.1840 | 0.7569 | 0.2700 | 0.5935 | 0.2050 | 0.3954 |

Selected comparison:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | F1 |
|---|---|---:|---:|---:|
| `hybrid` | Logistic Regression | 0.1840 | 0.7569 | 0.2700 |
| `hybrid_no_balance` | Logistic Regression | 0.1837 | 0.7561 | 0.2673 |
| `graph_only` | Logistic Regression | 0.1812 | 0.7508 | 0.2625 |
| `hybrid` | LightGBM | 0.1792 | 0.7626 | 0.2560 |
| `hybrid` | XGBoost | 0.1755 | 0.7532 | 0.2348 |
| `graph_no_balance` | Random Forest | 0.1730 | 0.7481 | 0.2441 |
| `text_only` | Logistic Regression | 0.1398 | 0.7118 | 0.2202 |
| `history_only` | Logistic Regression | 0.1424 | 0.6905 | 0.2341 |
| `graph_only` | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| `hybrid` | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

The best model's test confusion counts are:

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 19,911 | 2,587 | 1,020 | 667 |

## 7. Interpretation

The results support three main findings.

First, temporal graph/history features are useful. Graph-only and graph-no-balance models are close to the hybrid model, and both are clearly stronger than the historical negative-ratio baseline and dummy baselines.

Second, text-property features are helpful mainly when combined with graph/history features. The best model is the hybrid Logistic Regression model, while text-only models remain weaker than graph-based models.

Third, structural-balance features are interpretable but have a small predictive contribution in the current strict temporal setting. The hybrid model slightly outperforms `hybrid_no_balance`, but the difference is small.

The most important features for the best model include text-property coefficients, source out-degree, source signed degree, target in-degree, source negative degree, PageRank, betweenness, and community negativity ratios. This suggests that future negative-dominant relationships are shaped by both the content-side signal in hyperlink text properties and the historical position of subreddits in the signed interaction network.

## 8. Limitations

The label is derived from `LINK_SENTIMENT` and should be treated as a proxy, not direct evidence of real-world conflict. Negative hyperlinks do not prove harassment, raids, or coordinated attacks. The dataset is historical and covers 2013-2017. K-core filtering improves tractability and reduces sparsity, but it restricts the analysis to a denser subgraph. The current task predicts future relationships among historically observed source-target pairs rather than completely unseen pairs.

## 9. Conclusion

This project demonstrates a practical, leakage-aware, paper-style pipeline for predicting negative-dominant cross-community relationships on Reddit. The strongest result comes from a hybrid temporal signed-network model, and the ablation study shows that graph/history features are the most stable signal family. The project is suitable as a problem-based Social Media Data Analysis final project and can be extended toward research on explainable temporal signed link prediction.

## References

See `docs/references.md` for the project reference list.
