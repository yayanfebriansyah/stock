# LSTM Stock Forecasting Framework

This project is a Python implementation of the stock forecasting framework proposed in the paper *"An Advisor Neural Network framework using LSTM-based Informative Stock Analysis"*. Its goal is to predict daily stock trends using a Long Short-Term Memory (LSTM) model enriched with a diverse range of features.

The project also includes the classic statistical model **ARIMA** and a **Naive Forecast** as baselines to compare against and validate the superior performance of the LSTM model.



## 📜 Table of Contents
* [Key Features](#-key-features)
* [Research Inspiration](#-research-inspiration)
* [Technologies Used](#-technologies-used)
* [Installation](#-installation)
* [How to Run](#-how-to-run)
* [Code Structure](#️-code-structure)
* [Example Output](#-example-output)

---
## ✨ Key Features
* **Multi-Feature LSTM Model**: Implements an LSTM architecture that combines four types of data:
    1.  Historical Data (OHLC)
    2.  Technical Indicators (RSI, ADX, AO, etc.)
    3.  Seasonal Data (Day, Month, Quarter)
    4.  News Sentiment Data (Simulated)
* **Baseline Models**: Includes ARIMA and Naive Forecast models for a comparative performance evaluation.
* **Comparative Visualization**: Generates a plot that compares the prediction results from all three models against the actual data.
* **Modular Code**: Written with a clean, function-based structure that is easy to read and extend.

---
## 🔬 Research Inspiration
This project is inspired by and based on the methodology presented in the following research paper:

> Ricchiuti, F., & Sperlí, G. (2025). *An Advisor Neural Network framework using LSTM-based Informative Stock Analysis for Daily investment Advice*. Expert Systems With Applications, 259, 125299.

---
## 🛠️ Technologies Used
* **Python 3.x**
* **TensorFlow & Keras**: For building and training the LSTM model.
* **Pandas**: For data manipulation.
* **Pandas_TA**: For engineering technical indicator features.
* **Statsmodels**: For the ARIMA model implementation.
* **Scikit-learn**: For data preprocessing and evaluation metrics.
* **yfinance**: For downloading stock market data.
* **Matplotlib**: For data visualization.

---
## 📦 Installation
1.  Ensure you have Python 3.7 or newer installed.
2.  Clone this repository (if applicable) or save the `.py` code file in your local directory.
3.  Open your terminal or command prompt and install all the required libraries with a single command:
    ```bash
    pip install yfinance pandas_ta statsmodels tensorflow matplotlib scikit-learn
    ```

---
## 🚀 How to Run
1.  Open the Python code file (`.py` or `.ipynb` notebook).
2.  Navigate to the `main()` function and adjust the parameters in the **"MAIN PARAMETERS"** section as needed:
    ```python
    # ------------------- MAIN PARAMETERS -------------------
    TICKER = 'AAPL'            # Change stock ticker, e.g., 'GOOGL', 'TSLA'
    START_DATE = '2019-11-01'  # Data start date
    END_DATE = '2023-10-31'    # Data end date
    WINDOW_SIZE = 3            # Window size for the LSTM
    
    LSTM_PARAMS = { ... }      # LSTM hyperparameters
    ARIMA_ORDER = (5, 1, 0)    # Order for ARIMA
    # ----------------------------------------------------
    ```
3.  Execute the file. If using a `.py` file in the terminal:
    ```bash
    python your_script_name.py
    ```
    If using Google Colab or a Jupyter Notebook, simply run the cell containing the code.

---
## 🏗️ Code Structure
The code is organized into several key functions for modularity:
* `install_and_import_libraries()`: Handles the installation and import of all dependencies.
* `download_data()`: Fetches stock data from yfinance.
* `engineer_features()`: Performs feature engineering based on the paper's methodology.
* `preprocess_data_for_lstm()`: Splits, scales, and creates windowed data for the LSTM.
* `build_and_train_lstm()`: Builds and trains the LSTM model.
* `evaluate_lstm()`: Evaluates the LSTM model and inverse transforms the results back to the original scale.
* `run_naive_forecast()` & `run_arima_forecast()`: Run and evaluate the baseline models.
* `main()`: The main function that orchestrates the entire workflow from start to finish.

---
## 📊 Example Output
After the script finishes, you will see two primary outputs:

1.  **Performance Summary in the Console**: A comparison of the **Root Mean Squared Error (RMSE)** scores from the three models. A lower RMSE score indicates better performance.
    ```
    ==================================================
    📊 MODEL PERFORMANCE COMPARISON 📊
    ==================================================
    Evaluation Metric: Root Mean Squared Error (RMSE)
    The lower the RMSE, the better the model.

      - Naive Forecast RMSE: 2.1534
      - ARIMA RMSE         : 2.1189
      - LSTM (Paper) RMSE  : 1.8742
    ==================================================
    ```

2.  **Visualization Plot**: A chart displaying a comparison between the actual *Daily Return* values and the values predicted by the LSTM, ARIMA, and Naive models.