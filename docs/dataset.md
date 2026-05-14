# Dataset Description

## Source

The project uses the Stanford SNAP Reddit Hyperlink Network:

- `soc-redditHyperlinks-body.tsv`
- `soc-redditHyperlinks-title.tsv`

Source URL: <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>

## Raw Schema

| Column | Meaning |
|---|---|
| `SOURCE_SUBREDDIT` | Source subreddit containing the hyperlink post |
| `TARGET_SUBREDDIT` | Target subreddit referenced by the hyperlink |
| `POST_ID` | Reddit post identifier |
| `TIMESTAMP` | Post timestamp |
| `LINK_SENTIMENT` | `-1` for negative, `1` for positive/neutral |
| `PROPERTIES` | 86 numeric text-property features |

## Local Audit Results

The two raw files contain 858,488 data rows:

| File | Rows | Negative Ratio |
|---|---:|---:|
| body | 286,561 | 7.35% |
| title | 571,927 | 10.69% |
| combined | 858,488 | 9.58% |

The local files have no malformed rows, missing core fields, invalid labels, invalid timestamps, or malformed property vectors. Every row has exactly 86 text-property values.

## Modeling Convention

The body and title files are concatenated, not joined by `POST_ID`. A new column, `dataset_source`, records whether each hyperlink came from the body or title file.

The course-project task is:

> Predict whether a future source-target subreddit relationship becomes negative using historical signed-network and text-property features.

The report should describe `LINK_SENTIMENT` as a derived proxy label, not as direct ground truth for real-world conflict.
