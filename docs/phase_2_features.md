# Phase 2: Network Construction and Feature Engineering

## Checklist

- [x] Build a signed directed MultiDiGraph with NetworkX.
- [x] Extract node features: in-degree, out-degree, signed degree, PageRank, betweenness, reciprocity.
- [x] Extract clustering coefficient and Louvain/greedy-modularity community features.
- [x] Extract community-level negativity features and same-community pair features.
- [x] Extract pair features: interaction count, positive count, negative count, negative ratio, reciprocal edge.
- [x] Extract approximate structural-balance features from signed local neighborhoods.
- [x] Parse the 86 `PROPERTIES` values into numeric text features.
- [x] Aggregate text-property features at source-target pair level.
- [x] Export reusable feature tables.

## Deliverables

- `notebooks/02_network_construction.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- `data/processed/phase2/phase2_node_features.csv`
- `data/processed/phase2/phase2_edge_features.csv`
- `data/processed/phase2/phase2_triadic_features.csv`
- `data/processed/phase2/phase2_text_features.csv`
- `data/processed/phase2/phase2_modeling_table.csv`

## Feature Groups for Ablation

| Feature group | Examples |
| --- | --- |
| Graph/history | degree, PageRank, reciprocity, previous positive/negative counts |
| Community/clustering | clustering coefficient, community size, same-community flag, community negativity gap |
| Structural balance | common neighbors, `balance_+++`, `balance_++-`, `balance_+--`, `balance_---` |
| Text | `text_property_00` to `text_property_85`, body/title share |
| Hybrid | all graph, balance, temporal-history, and text features |

## Notes for the Report

Structural balance features are local approximations. They are useful for an interpretable course project, but the limitation should be stated clearly.
