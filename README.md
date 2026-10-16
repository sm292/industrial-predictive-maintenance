# industrial-predictive-maintenance


# Industrial Machine Intelligence

## Business problem
Identify machine records with elevated failure risk to support
maintenance prioritisation in industrial operations.

## Dataset
UCI AI4I 2020 Predictive Maintenance Dataset:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset is synthetic and is used for educational evaluation.

## Technology
Python, Pandas, scikit-learn, Plotly, Streamlit.

## Run locally or in GitHub Codespaces

```bash
python -m pip install -r requirements.txt
python download_data.py
python train_model.py
streamlit run app.py --server.address 0.0.0.0
```

## Evaluation
Report precision, recall, F1, PR-AUC and ROC-AUC on a held-out
test set. Discuss class imbalance, threshold selection and limitations.

## Limitations
The data is synthetic. Risk scores are not verified against real
factory operating conditions and must not be used as the sole basis
for maintenance decisions.
