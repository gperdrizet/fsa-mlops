from pathlib import Path

import pandas as pd
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

# Load reference (training) data and current data
current_dir = Path(__file__).resolve().parent
reference = pd.read_csv(f"{current_dir}/train_data.csv")
current = pd.read_csv(f"{current_dir}/reference_data.csv")

# Create the report with valid metric presets
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

# Run the report
result = report.run(reference_data=reference, current_data=current)

# Save the report as HTML
result.save_html("evidently_report.html")

print("Report successfully generated: evidently_report.html")
