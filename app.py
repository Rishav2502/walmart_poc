from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    store_no = request.form['storeNo']
    forecast_days = int(request.form['forecastDays'])
    product_family = request.form['productFamily']
    promotion1 = int(request.form['promotion1'])
    promotion2 = int(request.form['promotion2'])
    promotion3 = int(request.form['promotion3'])

    # Load ML models and perform forecasting calculations
    # Here, you can use your preferred ML library (e.g., scikit-learn) and models

    # Generate dummy forecasted data (example)
    dates = pd.date_range(start='2023-07-05', periods=forecast_days)
    forecasted_values = np.random.randint(100, 1000, forecast_days)

    # Pass the forecasted data to the template
    return render_template('forecast.html', store_no=store_no, forecast_days=forecast_days,
                           product_family=product_family, promotion1=promotion1, promotion2=promotion2,
                           promotion3=promotion3, dates=dates, forecasted_values=forecasted_values)

if __name__ == '__main__':
    app.run(debug=True)
