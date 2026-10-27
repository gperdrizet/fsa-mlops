# Use Python official slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy script into the container
COPY train.py .

# Define command to run the script automatically
CMD ["python", "train.py"]
