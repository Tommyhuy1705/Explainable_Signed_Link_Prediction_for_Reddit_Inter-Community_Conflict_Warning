# Data Provenance and Reproducibility Audit

## Dataset Source

This project uses the Reddit Hyperlink Network files from the Kaggle mirror:

- Kaggle dataset: `Signed Graphs`
- Kaggle URL: <https://www.kaggle.com/datasets/wolfram77/graphs-signed>
- Files used: `soc-redditHyperlinks-body.tsv` and `soc-redditHyperlinks-title.tsv`

The original academic source is the Stanford SNAP Reddit Hyperlink Network:

- SNAP URL: <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>
- Paper: Srijan Kumar, William L. Hamilton, Jure Leskovec, and Dan Jurafsky. "Community Interaction and Conflict on the Web." WWW 2018.

The report cites both sources: Kaggle is the reproducible data access point for the course requirement, while SNAP/Kumar et al. are the original provenance and methodological context.

## Why This Dataset Fits Social Media Data Analysis

Reddit is a social media platform organized around topical communities. The dataset records cross-community hyperlinks where each row links a source subreddit to a target subreddit. This makes it suitable for social network analysis because subreddits can be modeled as nodes and hyperlinks as directed signed edges.

The dataset supports both network analysis and predictive modeling:

- Community interaction graph construction.
- Centrality, reciprocity, clustering, and community detection.
- Signed edge analysis through `LINK_SENTIMENT`.
- Temporal prediction using timestamps.
- Text-side ablation through 86 precomputed `PROPERTIES` values.

## Raw Files and Schema

| File | Role |
| --- | --- |
| `soc-redditHyperlinks-body.tsv` | Hyperlinks appearing in Reddit post bodies |
| `soc-redditHyperlinks-title.tsv` | Hyperlinks appearing in Reddit post titles |

Expected raw schema:

| Column | Meaning |
| --- | --- |
| `SOURCE_SUBREDDIT` | Source subreddit containing the hyperlink post |
| `TARGET_SUBREDDIT` | Target subreddit referenced by the hyperlink |
| `POST_ID` | Reddit post identifier |
| `TIMESTAMP` | Post timestamp |
| `LINK_SENTIMENT` | `-1` negative, `1` positive/neutral |
| `PROPERTIES` | 86 comma-separated numeric text-property features |

## Local Audit Results

The audit script `scripts/audit_dataset.py` validates the local raw files before modeling.

| File | Rows | Negative Rows | Negative Ratio | Malformed Property Rows |
| --- | ---: | ---: | ---: | ---: |
| body | 286,561 | 21,070 | 7.35% | 0 |
| title | 571,927 | 61,140 | 10.69% | 0 |
| combined | 858,488 | 82,210 | 9.58% | 0 |

Timestamp range: 2013-12-31 16:20:20 to 2017-04-30 16:58:21.

The latest local audit found:

- No missing required columns.
- No missing core fields.
- No invalid labels outside `-1` and `1`.
- No invalid timestamps.
- No malformed `PROPERTIES` vectors.

## Reproduction Command

```bash
python scripts/audit_dataset.py --raw-dir data/raw --json-out data/processed/dataset_audit.json
```

Raw data is intentionally excluded from Git because of file size. To reproduce the project, download the two Reddit hyperlink files from Kaggle, place them in `data/raw/`, and then run the audit command above before executing the modeling pipeline.
