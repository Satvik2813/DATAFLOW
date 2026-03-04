# Cloud Cost Predictor (DATAFLOW)

A Streamlit web application designed to predict Cloud Server Operational Costs using a pre-trained Multiple Linear Regression model. 

## Features
- **Predictive Modeling**: Utilizes a pre-trained machine learning model (`model/cloud_cost_model.pkl`) to estimate operational costs based on server metrics.
- **Interactive Interface**: Clean, modern, and responsive user interface built with Streamlit.
- **Instant Estimation**: Input metrics into the left panel and get instant cost estimations directly on the screen.

## Installation

1. Ensure you have Python installed.
2. Clone the repository (if applicable) and navigate to the project directory:
   ```bash
   cd "Cloud Cost predictor"
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application locally, use the following command:

```bash
streamlit run app.py
```

The application will start, and you can view it in your browser at the local URL provided in the terminal (usually `http://localhost:8501`).
