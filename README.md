
# Industrial Machine Intelligence
## Machine Failure Prediction & Predictive Maintenance

An end-to-end Data Science and Machine Learning project designed to identify industrial machine failure risk and support proactive maintenance prioritization using operational data.

The project applies machine learning to the UCI AI4I 2020 Predictive Maintenance Dataset and provides an interactive Streamlit application for exploring machine-risk predictions.

**Repository:** https://github.com/sm292/industrial-predictive-maintenance

## Project Overview

Unexpected equipment failures can result in production downtime, maintenance costs, and operational inefficiencies.

This project explores how data-driven predictive maintenance can help identify machine records with elevated failure risk. It demonstrates a practical workflow covering dataset acquisition, data preparation, model training, evaluation, and interactive visualization.

The dataset is synthetic and intended for educational and experimental evaluation.

## Business Objectives

- Analyze machine operating conditions and failure patterns.
- Develop a machine failure prediction workflow using machine learning.
- Evaluate classification performance on held-out data.
- Explore the effects of class imbalance and decision thresholds.
- Present machine-risk insights through an interactive dashboard.
- Demonstrate how predictive analytics can inform maintenance prioritization.

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains machine operating variables and failure indicators suitable for exploring predictive maintenance classification.

Common variables in the dataset include:

- Product quality type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure indicator
- Failure-type indicators

The precise input features and target definition should follow the implementation in the training script.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Source | UCI Machine Learning Repository |
| Version Control | Git and GitHub |

## Project Workflow

### 1. Dataset Acquisition

The data download script retrieves the dataset required for the project workflow.

### 2. Data Preparation

Prepare the data for machine learning, including appropriate input feature selection, target definition, and validation.

### 3. Model Training

Train the machine failure classification model using the project's training pipeline.

### 4. Model Evaluation

Evaluate predictions on a held-out test set using appropriate classification metrics:

- Precision
- Recall
- F1-score
- Precision-Recall AUC (PR-AUC)
- ROC-AUC

Because machine failures may be relatively uncommon, accuracy alone is not sufficient to assess model performance.

### 5. Interactive Dashboard

Use the Streamlit application to explore the functionality provided by the project and present machine-risk insights in an accessible interface.

## Repository Structure

```text
industrial-predictive-maintenance/
├── README.md
├── .gitignore
├── requirements.txt
├── download_data.py
├── train_model.py
└── app.py
```

## Installation and Execution

### Prerequisites

- Python installed
- Git installed
- Internet access for initial dataset retrieval

### 1. Clone the repository

```bash
git clone https://github.com/sm292/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python download_data.py
```

### 5. Train the model

```bash
python train_model.py
```

Review the training script for the generated model artifacts and evaluation outputs.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Evaluation and Responsible Use

Model quality should be assessed using held-out data and metrics that account for class imbalance.

Precision measures the proportion of predicted failures that are actual failures. Recall measures the proportion of actual failures identified by the model. F1-score balances precision and recall, while PR-AUC and ROC-AUC provide complementary views of discrimination performance.

Threshold selection should reflect the relative costs of missed failures and unnecessary inspections.

No performance figures are claimed in this README. Refer to reproducible evaluation outputs from the current training pipeline when reporting results.

## Potential Industrial Applications

- Maintenance prioritization
- Machine-risk screening
- Operational data analysis
- Failure-pattern investigation
- Predictive maintenance research
- Decision support for maintenance teams

These are potential applications rather than verified operational outcomes.

## Limitations

- The dataset is synthetic and may not represent real factory conditions.
- Predictions do not guarantee that a machine will or will not fail.
- Performance on this dataset does not establish generalization to unseen industrial environments.
- Real-world use would require validation using representative equipment data, operational testing, and appropriate engineering oversight.

The model should not be used as the sole basis for safety-critical maintenance decisions.

## Future Improvements

- Add explainability using feature importance or SHAP.
- Explore cost-sensitive threshold selection.
- Introduce model comparison and cross-validation.
- Add automated tests and model artifact versioning.
- Integrate validated real-world industrial sensor data.
- Extend monitoring and reporting capabilities.

## Author

**Sumit Mondal**

Data Science | Machine Learning | Python | Predictive Analytics

GitHub: https://github.com/sm292

## License

Refer to the repository's license information for permitted use and distribution.
