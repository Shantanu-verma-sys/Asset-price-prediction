
import numpy as np
import pandas as pd
import requests
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

def fetchAndInit(tf):
    #params for api data request
    options = {
        "symbol": 'BTCUSDT',
        "interval": tf,
        "limit": "1000"
    }
    global times,space,df
    space = 25
    #fetch btc price data in json
    x = requests.get(
        url="https://api.binance.com/api/v3/klines", params=options)
    df = pd.DataFrame(x.json())

    #label the column for dataframe
    df.columns = ['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time',
                  'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']

    #convert price from string to float64
    df["c"] = df["c"].astype('float64')

    #store latest data row time in unix ms
    times = list(df.close_time)

    #drop unnessary columns for now
    df.drop(columns=['open_time', 'o', 'h', 'l', 'v', 'close_time', 'qav',
            'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'], inplace=True)
    
    #convert data to time series which can use supervised SVM algorithm for regression
    df["Prediction"] = df["c"].shift(-space)

def trainModel(tf):
    global svr_rbf,confidence
    fetchAndInit(tf)
    #input data X
    x = np.array(df.drop(['Prediction'], 1))
    x = x[:len(df)-space]

    #output data y (time shifted by 'space' rows)
    y = np.array(df["Prediction"])
    y = y[:-space]

    #20% test 80% train dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #use radial kernel C = trade errors for stablity , gamma=how far the influence of a single training example reaches
    svr_rbf = SVR(kernel="rbf", C=1000, gamma=0.00001)

    #perform regr
    svr_rbf.fit(x_train, y_train)

    #print how accurate model is
    confidence = svr_rbf.score(x_test, y_test)
    print(confidence)

def runPrediction(tf):
    global final_df,full_prediction
    fetchAndInit(tf)
    pred_days = np.array(df.drop(["Prediction"], 1))[-space:]

    #predict 
    predicted_price_future = svr_rbf.predict(pred_days)
    full_prediction = svr_rbf.predict(np.array(df.drop(['Prediction'], 1)))

    #store forecast in new dataframe
    final_df = pd.DataFrame({"prediction": full_prediction})

#run ML algo
trainModel('5m')
runPrediction('5m')

#run fast server
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#for index.html as template to serve with ML data
templates = Jinja2Templates(directory="templates")

#root url data response (index.html + forecasted data)
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/prediction')
async def mlApp(tf: Optional[str] = '5m',train:Optional[str] = None):
    if train == 'yes':
        print("training")
        trainModel(tf)
    runPrediction(tf)
    return {"data":list(final_df["prediction"]),"closeTime":times[0+space],"timeframe":tf,"orig":list(df["c"][space:])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


