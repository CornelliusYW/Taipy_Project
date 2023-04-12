from taipy.gui import Gui, notify
import pandas as pd
import yfinance as yf
import pickle
import prophet

from create_mod_core import *
tickers = yf.Tickers('msft aapl goog')

property_chart = {"type":"lines",
                  "x":"Date",
                  "y[1]":"Open",
                  "y[2]":"Close",
                  "y[3]":"High",
                  "y[4]":"Low",
                  "color[1]":"green",
                  "color[2]":"grey",
                  "color[3]":"red",
                  "color[4]":"yellow"
                 }

property_chart_pred = {
    'type':'lines',
    'x':'Date',
    'y[1]': 'Close_Prediction'
}
df = pd.DataFrame([], columns = ['Date', 'High', 'Low', 'Open', 'Close'])
df_pred = pd.DataFrame([], columns = ['Date','Close_Prediction'])

model_list = {
    'AAPL': 'aapl_model.pkl',
    'MSFT': 'msft_model.pkl', 
    'GOOG': 'goog_model.pkl'
}

stock_text = "No Stock to Show"
stock = ""
chart_text = 'No Chart to Show'
pred_text = 'No Prediction to Show'

page = """
# Stock Portfolio

### Choose the stock to show
<|toggle|theme|>

<|layout|columns=1 1|
<|
<|{stock_text}|>

<|{stock}|selector|lov=MSFT;AAPL;GOOG;Reset|dropdown|>

<|Press for Stock|button|on_action=on_button_action|>
|>


<|
<|{chart_text}|>
<|{df}|chart|properties={property_chart}|>
|>

|>

<|{pred_text}|>
<|{df_pred}|chart|properties={property_chart_pred}|>

<|Update Model|button|on_action=update_model|>
"""

def on_button_action(state):
    if state.stock == 'Reset':
        state.stock_text = "No Stock to Show"
        state.chart_text = 'No Chart to Show'
        state.df = pd.DataFrame([], columns = ['Date', 'High', 'Low', 'Open', 'Close'])
        state.pred_text = 'No Prediction to Show'
    else:
        state.stock_text = f"The stock is {state.stock}"
        state.chart_text = f"Monthly history of stock {state.stock}"
        state.df = tickers.tickers[state.stock].history().reset_index()

def on_change(state, var_name, var_value):
    if var_name == "stock" and var_value == "Reset":
        pass
    else:
        with open(model_list[state.stock], 'rb') as f:
            state.pred_text = f'1 Year Close Prediction of Stock {state.stock}'
            model = pickle.load(f)  
            state.df_pred = model.predict(model.make_future_dataframe(periods=365))[['ds', 'yhat']].rename(columns = {'ds':'Date', 'yhat':'Close_Prediction'})

        return        

def update_model(state):
    print("Update Model Clicked")
    pipeline = create_and_submit_pipeline()
    notify(state, 'info', 'Model have finish training')


def create_and_submit_pipeline():
    print("Execution of pipeline...")
    # Create the pipeline
    retrain_pipeline = tp.create_pipeline(retraining_model_pipeline_cfg)
    # Submit the pipeline (Execution)
    tp.submit(retrain_pipeline)
    return retrain_pipeline

tp.Core().run()
Gui(page).run(use_reloader=True)
