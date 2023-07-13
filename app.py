from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import save_model, load_model
global plot_no
plot_no = 1


app = Flask(__name__)

#with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_models_1.pkl', 'rb') as file:
  #model = pickle.load(file)
model = load_model('C:/Users/Admin/PycharmProject/walmart_poc/lstm_models/store_44_model.h5', compile=False)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/time_cov.pkl', 'rb') as file:
  time_cov = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_store_specific.pkl', 'rb') as file:
  list_of_store_specific = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_train_data.pkl', 'rb') as file:
  list_of_train_data = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_test_data.pkl', 'rb') as file:
  list_of_test_data = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_train.pkl', 'rb') as file:
  list_of_train = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_test.pkl', 'rb') as file:
  list_of_test = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_final_df.pkl', 'rb') as file:
  list_of_final_df = pickle.load(file)
with open('C:/Users/Admin/PycharmProject/walmart_poc/pickle_files/list_of_scaler.pkl', 'rb') as file:
  list_of_scaler = pickle.load(file)

def prediction_fn(store_nbr,n_steps,promotion_weightage):
  scaler=list_of_scaler[store_nbr-1]
  train_data=list_of_train[store_nbr-1]
  test_data=list_of_test[store_nbr-1]
  target_columns = range(33)
  train_data_scaled = list_of_train_data[store_nbr-1]
  test_data_scaled = list_of_test_data[store_nbr-1]
  final_df=list_of_final_df[store_nbr-1]
  train_ratio = 0.95
  train_size = int(train_ratio * len(final_df))
  def create_sequences(data, sequence_length):
    X = []
    y = []
    for k in range(len(data) - sequence_length):
        X.append(data[k:k+sequence_length,-45:])
        y.append(data[k+sequence_length, target_columns])
    return np.array(X), np.array(y)
  sequence_length=10
  X_train, y_train=create_sequences(train_data_scaled, sequence_length)
  X_test, y_test = create_sequences(test_data_scaled, sequence_length)
  remaining_data = final_df.iloc[train_size:, :]*promotion_weightage
  remaining_data_scaled = scaler.transform(remaining_data)
  X_future, y_future = create_sequences(remaining_data_scaled, sequence_length)
  predictions = model.predict(X_future)
  predictions_df=pd.DataFrame(columns=final_df.columns)
  for i in range(0,len(predictions_df.columns)):
    if i<33:
      predictions_df.iloc[:,i]=pd.DataFrame(predictions)[i]
    else:
      predictions_df.iloc[:,i]=pd.DataFrame(test_data_scaled).iloc[-75:,i].reset_index(drop=True)
  predictions_rescaled = scaler.inverse_transform(predictions_df)
  rescaled_predictions=pd.DataFrame(predictions_rescaled).iloc[:,:33]
  rescaled_predictions.columns=test_data.iloc[-75:,:33].columns
  rescaled_predictions=rescaled_predictions.set_index(test_data.iloc[-75:,:33].index)
  actuals=test_data.iloc[-75:,:33]
  mape_df=pd.DataFrame()
  mape=[]
  cols=rescaled_predictions.columns
  for i in cols:
    actual_values=actuals[i]
    predicted_values=rescaled_predictions[i]
    mape_val = np.nanmean(np.abs((actual_values - predicted_values) / np.where((actual_values != 0) & (predicted_values != 0), actual_values, np.nan))) * 100
    mape.append(mape_val)
    print(f"MAPE value for {i} is: {mape_val}%")
  mape_df['Product Name']=cols
  mape_df['MAPE']=mape
  return train_data,actuals,rescaled_predictions.iloc[:n_steps,:]

def plotting(store_nbr,n_steps,promotion_weightage, product_name):
    global plot_no
    train,actuals,prediction1=prediction_fn(store_nbr, n_steps, promotion_weightage)
    # Calculate the upper and lower bounds for the interval
    interval = prediction1['sales_'+product_name] * (14.84 / 100)

    """# Plotting the train data
    plt.plot(train.index, train['sales_'+product_name].values, label='Train Data')"""

    # Plotting the forecasted data
    plt.plot(prediction1.index, prediction1['sales_'+product_name].values, label='Forecasted Data')

    # Plotting the interval
    plt.fill_between(prediction1.index, prediction1['sales_'+product_name].values - interval,
                     prediction1['sales_'+product_name].values + interval,
                     color='gray', alpha=0.3, label='Interval')

    # Adding labels and title to the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Forecasted Data with Interval')

    # Adding legend
    plt.legend()
    plot_no = plot_no + 1
    image_path = plt.savefig(f"C:/Users/Admin/PycharmProject/walmart_poc/static/prediction{plot_no}.png")
    return image_path

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/plot', methods=['POST'])
def forecast():
    store_no = int(request.form['storeNo'])
    forecast_days = int(request.form['forecastDays'])
    promotion = float(request.form['promotion'])
    #promotion2 = float(request.form['promotion2'])
    #promotion3 = float(request.form['promotion3'])
    product_family = request.form.getlist('productFamily')
    print(store_no)
    print(forecast_days)
    #print(promotion)
    #print(promotion2)
    #print(promotion3)


    chart = plotting(store_no, forecast_days, promotion, product_family[0])
    #Perform the forecast calculation and obtain the forecast data

    #return render_template('forecast1.html', store_no=store_no, forecast_days=forecast_days, promotion=promotion, product_family=product_family)

    return render_template('plot.html', image_path = chart)

if __name__ == '__main__':
    app.run(debug=True)
