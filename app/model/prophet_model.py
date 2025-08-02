from prophet import Prophet
import pandas as pd


class ProphetModel:
    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def train(self, df: pd.DataFrame):
        df_copy = df.copy()
        columns = [item for item in df.columns.to_list() if item not in ['item_name', 'date', 'sales_qty']]
        df_copy = df_copy.drop(columns=columns, axis=1)
        df_copy = df_copy.rename(columns={'date': 'ds', 'sales_qty': 'y'})

        for item, group in df_copy.groupby('item_name'):
            model = Prophet()
            model.fit(group)

            future = model.make_future_dataframe(periods = 30)
            forecast = model.predict(future)
            
            self.models[item] = model
            self.forecasts[item] = forecast


    def predict(self, item_name: str, period: int):
        return self.forecasts[item_name]


