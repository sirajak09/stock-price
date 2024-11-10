import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from flask import Flask, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# API Endpoints and Configuration
STOCK_ENDPOINT = "https://www.alphavantage.co/query"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
STOCK_NAME = 'TSLA'
COMPANY_NAME = 'Tesla Inc'
STOCK_API_KEY = "96FEDIFUL2R1BJ0Y"
NEWS_API_KEY = "ca8337f6f0f947a28529ad4dbcb26176"

def get_stock_data():
    stock_params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": STOCK_NAME,
        "apikey": STOCK_API_KEY
    }
    response = requests.get(STOCK_ENDPOINT, params=stock_params)
    data = response.json().get('Time Series (Daily)', {})
    return [value for (key, value) in data.items()]

def get_news():
    news_params = {
        "apiKey": NEWS_API_KEY,
        "qInTitle": COMPANY_NAME
    }
    new_response = requests.get(NEWS_ENDPOINT, params=news_params)
    articles = new_response.json().get('articles', [])
    return articles[:3]  # Get top 3 articles

def predict_next_day_price_with_confidence(data_list):
    # Prepare data for model
    prices = [float(day["4. close"]) for day in data_list[:5]]  # Use last 5 days
    days = np.array(range(1, len(prices) + 1)).reshape(-1, 1)
    prices = np.array(prices).reshape(-1, 1)
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(days, prices)
    # Predict the next day (day 6 in this case)
    next_day = np.array([[len(prices) + 1]])
    predicted_price = model.predict(next_day)[0][0]
    # Calculate MAPE on past data to get confidence level
    predictions = model.predict(days)
    mape = mean_absolute_percentage_error(prices, predictions)
    confidence_level = (1 - mape) * 100  # Convert MAPE to confidence percentage
    
    return predicted_price, confidence_level

@app.route('/')
def index():
    # Stock Data
    data_list = get_stock_data()
    if len(data_list) < 2:
        return "Insufficient data to display stock prices."

    yesterday_data = data_list[0]
    day_before_yesterday_data = data_list[1]
    yesterday_closing_price = float(yesterday_data["4. close"])
    day_before_yesterday_closing_price = float(day_before_yesterday_data["4. close"])
    difference = abs(yesterday_closing_price - day_before_yesterday_closing_price)
    difference_percentage = (difference / yesterday_closing_price) * 100
    
    # News Data
    articles = get_news()
    
    # Prediction Data
    predicted_price, confidence_level = predict_next_day_price_with_confidence(data_list)

    return render_template("index.html",
                           stock_name=STOCK_NAME,
                           company_name=COMPANY_NAME,
                           yesterday_closing_price=yesterday_closing_price,
                           day_before_yesterday_closing_price=day_before_yesterday_closing_price,
                           difference_percentage=difference_percentage,
                           articles=articles,
                           predicted_price=predicted_price,
                           confidence_level=confidence_level)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)