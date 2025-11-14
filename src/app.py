from flask import Flask, jsonify, render_template
import joblib
import pandas as pd
import requests
from datetime import datetime

app = Flask(__name__, template_folder="../templates")

# Load mô hình và scaler
ridge_model = joblib.load("models/ridge_model.pkl")
scaler = joblib.load("models/scaler_ridge.pkl")

FEATURES = [
    'gas_t-1', 'gas_t-5', 'gas_t-15',
    'gas_rolling_5', 'gas_rolling_15',
    'tx_count', 'total_gas_used', 'avg_block_full_ratio',
    'year', 'month', 'day', 'hour', 'minute'
]

# API key từ Etherscan
ETHERSCAN_API_KEY = "5BVG3PWSMMK93T9PVEIKBFF1Q2JJXPEAJ9"

def get_gas_oracle():
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=gastracker&action=gasoracle&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url).json()
    return response["result"]

def get_block_info(block_number):
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=block&action=getblockreward&blockno={block_number}&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url).json()
    return response["result"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Lấy dữ liệu hiện tại từ Etherscan
        oracle = get_gas_oracle()
        gas_now_gwei = float(oracle["ProposeGasPrice"])
        last_block = int(oracle["LastBlock"])

        block_info = get_block_info(last_block)

        tx_count = int(block_info.get("txCount", 0))
        gas_used = int(block_info.get("gasUsed", 0))
        gas_limit = int(block_info.get("gasLimit", 1))
        avg_block_full_ratio = gas_used / gas_limit if gas_limit > 0 else 0

        now = datetime.now()
        gas_now_wei = gas_now_gwei * 1e9

        # Tạo dữ liệu đầu vào cho mô hình
        input_data = {
            "gas_t-1": gas_now_wei,
            "gas_t-5": gas_now_wei,
            "gas_t-15": gas_now_wei,
            "gas_rolling_5": gas_now_wei,
            "gas_rolling_15": gas_now_wei,
            "tx_count": tx_count,
            "total_gas_used": gas_used,
            "avg_block_full_ratio": avg_block_full_ratio,
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute
        }

        df = pd.DataFrame([input_data], columns=FEATURES)
        X_scaled = scaler.transform(df)

        # Dự đoán giá gas sau 1 phút
        prediction_wei = ridge_model.predict(X_scaled)[0]
        prediction_gwei = prediction_wei / 1e9

        return jsonify({
            "current_gas_gwei": gas_now_gwei,
            "predicted_gas_gwei": round(prediction_gwei, 4)
        })

    except Exception as e:
        print("Error in predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
