# Explainable_Signed_Link_Prediction_for_Reddit_Inter-Community_Conflict_Warning

## Project Overview
This project is a final assignment for the **Social Data Analysis** course. The goal is to predict the sentiment (positive or negative) of hyperlinks between Reddit communities (subreddits) by leveraging signed directed network characteristics and structural balance theory over time.

## Objectives
- Build a temporal, signed, and directed network from Stanford's SNAP Reddit dataset.
- Extract node-level, edge-level, and triadic-level (Structural Balance) features.
- Address data leakage via temporal splitting and handle highly imbalanced class distributions.
- Train and evaluate Machine Learning models (Logistic Regression, Random Forest, XGBoost) to classify negative cross-community interactions.

## Dataset
The project utilizes the [Stanford SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html).
- `soc-redditHyperlinks-body.tsv`
- `soc-redditHyperlinks-title.tsv`

*Note: Due to file size limits, the `data/raw/` directory is ignored in this repository. Please download the datasets from the link above and place them in `data/raw/` before running the notebooks.*

## Installation & Setup
1. Clone the repository:
   \`\`\`bash
   git clone [https://github.com/your-username/SNA_Reddit_Project.git](https://github.com/Tommyhuy1705/SNA_Reddit_Project.git)
   cd SNA_Reddit_Project
   \`\`\`
2. Create a virtual environment and activate it:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   \`\`\`
3. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## How to Run
Follow the numbered Jupyter Notebooks in the `notebooks/` directory sequentially:
1. `01_data_exploration.ipynb`: Cleans data and performs Exploratory Data Analysis.
2. `02_network_construction.ipynb`: Builds the Multi-DiGraph using NetworkX.
3. `03_feature_engineering.ipynb`: Extracts Centrality and Triadic Balance features.
4. `04_modeling_and_evaluation.ipynb`: Trains ML models and evaluates predictions using F1-score and PR-AUC.

## Team Members
- Trần Viết Gia Huy - 31231027056
- Nguyễn Minh Nhựt - [MSSV]
- Nguyễn Trọng Hưởng - [MSSV]
- Tô Xuân Đông - [MSSV]

## License
This project is for educational purposes under the coursework of UEH (University of Economics Ho Chi Minh city).
