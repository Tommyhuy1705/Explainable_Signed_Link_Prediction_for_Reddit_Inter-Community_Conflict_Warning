# Presentation Q&A Cheat Sheet

## 1. Why is this a social media data analysis project?

The data comes from Reddit, a social media platform organized into communities. We model subreddits as nodes and hyperlinks between subreddits as directed signed edges, then analyze centrality, reciprocity, structural balance, and future negative-dominant relationships.

## 2. Why not use accuracy as the main metric?

Negative links are a minority class. A model can obtain high accuracy by predicting most pairs as non-negative. PR-AUC, F1, precision, recall, and balanced accuracy better reflect performance on the negative class.

## 3. How do you avoid data leakage?

For each split, graph and text-history features are computed only from interactions before the history cutoff. Labels are computed from a future window after that cutoff. Future label counts are excluded from the feature matrix.

## 4. Why use k-core filtering?

The full graph is very sparse. K-core filtering keeps a denser subgraph where network features such as PageRank, reciprocity, and structural-balance counts are more meaningful. This is a practical course-project choice and is stated as a limitation.

## 5. What does `LINK_SENTIMENT = -1` mean?

It indicates that the source post is negative toward the target subreddit according to the dataset's derived sentiment label. It is a proxy for negative interaction, not direct proof of harassment or a raid.

## 6. What is the difference between graph-only, text-only, and hybrid?

- Text-only uses aggregated historical text-property features from the `PROPERTIES` vector.
- Graph-only uses interaction counts, signed degrees, PageRank, reciprocity, and structural-balance features.
- Hybrid combines both.

## 7. Why use historical negative ratio as a baseline?

It is a strong intuitive baseline: if a source-target pair was negative before, it may be negative again. A model should outperform or at least meaningfully compare against this heuristic.

## 8. What does structural balance contribute?

Structural balance captures signed local-neighborhood patterns. For example, if two communities share many neighbors with certain positive/negative relationships, those triadic patterns may indicate whether their future relationship is likely to become negative-dominant.

## 9. Why is this not full conflict detection?

A negative hyperlink is only one signal of negative inter-community interaction. Real conflict detection would require richer evidence such as comment behavior, moderation events, user mobilization, and cross-community raids.

## 10. How can this become a publishable paper?

Extend the project from pair-level prediction to explainable temporal signed link prediction, add stronger signed/temporal graph models, evaluate robustness to label noise, and compare on more datasets or newer social platforms.
