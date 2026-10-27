'''Evidently AI monitoring script for data drift and classification metrics'''

from pathlib import Path

import pandas as pd

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# Load reference (training) data and current data
current_dir = Path(__file__).resolve().parent
reference = pd.read_csv(current_dir / "train_data.csv")
current = pd.read_csv(current_dir / "reference_data.csv")

# 1. Tell Evidently your prediction configuration is for a classification problem
#    (It will match 'target' and 'prediction' automatically if named exactly that)
data_schema = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_labels="prediction"
        )
    ]
)

# Wrap DataFrames in Dataset containers and explicitly enforce the task
reference_dataset = Dataset.from_pandas(reference, data_definition=data_schema)
current_dataset = Dataset.from_pandas(current, data_definition=data_schema)

# Create the report with valid metric presets
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

# Run the report
result = report.run(reference_data=reference_dataset, current_data=current_dataset)

# Save the report as HTML
result.save_html(f"{current_dir}/evidently_report.html")

print("Report successfully generated: evidently_report.html")
