import os
# MANDATORY: Force Keras 2 legacy behavior
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# --- MODEL LOADING ---
try:
    # Ensure this file exists in your main folder
    model = load_model('stock_dl_model.h5', compile=False)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. Error: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # FIX 1: Support both 'stock' and 'ticker' names from HTML
        stock = request.form.get('stock') or request.form.get('ticker')
        if not stock:
            stock = 'POWERGRID.NS'
        
        start = dt.datetime(2010, 1, 1)
        end = dt.datetime.now()
        start=end-dt.timedelta(days=365*2)
        # Download stock data
        df = yf.download(stock, start=start, end=end)
        
        if df.empty:
            return f"Error: No data found for ticker '{stock}'. Check symbol on Yahoo Finance."

        # FIX 2: Flatten MultiIndex columns (Crucial for 2026 yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Descriptive Data
        data_desc = df.describe()
        
        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()
        
        # Data splitting logic
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
        
        # FIX 3: Dynamic Scaling
        # We re-fit the scaler to the current stock's price range 
        # so the model receives values between 0 and 1.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame(df['Close'])) 
        
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        
        input_data = scaler.transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
            
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Prediction
        y_predicted = model.predict(x_test)
        
        # FIX 4: Inverse scaling using the dynamic scale factor
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        
        # Ensure static folder exists for plots
        if not os.path.exists('static'):
            os.makedirs('static')

        # Generate Plots
        # Plot 1
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.index, df.Close, 'y', label='Closing Price')
        ax1.plot(df.index, ema20, 'g', label='EMA 20')
        ax1.plot(df.index, ema50, 'r', label='EMA 50')
        ax1.set_title(f"{stock} - 20 & 50 Days EMA")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)
        
        # Plot 2
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.index, df.Close, 'y', label='Closing Price')
        ax2.plot(df.index, ema100, 'g', label='EMA 100')
        ax2.plot(df.index, ema200, 'r', label='EMA 200')
        ax2.set_title(f"{stock} - 100 & 200 Days EMA")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # Plot 3
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price")
        ax3.plot(y_predicted, 'r', label="Predicted Price")
        ax3.set_title(f"{stock} Prediction Trend")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)
        
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template('index.html', 
                               plot_path_ema_20_50=ema_chart_path, 
                               plot_path_ema_100_200=ema_chart_path_100_200, 
                               plot_path_prediction=prediction_chart_path, 
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)