# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir pandas numpy scikit-learn streamlit jsonpickle xgboost mlflow

# Copy the application code and all necessary files into the container
COPY bangalore_app.py /app/
COPY best_model.pkl /app/
COPY scaler.pkl /app/
COPY features.pkl /app/
COPY requirements.txt /app/

# Expose port 8501
EXPOSE 8501

# Run the Streamlit app on port 8501
CMD ["streamlit", "run", "bangalore_app.py"]