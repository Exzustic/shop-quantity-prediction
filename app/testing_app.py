import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

model_dir = 'saved_models'  # папка, де лежать моделі

def show_graph_1(df):
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales_qty'].sum()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(monthly_sales.index.astype(str), monthly_sales.values, c='blue', label='Trend Line')
    ax.scatter(monthly_sales.index.astype(str), monthly_sales.values, c='red', label='Data Points')
    ax.set_xlabel('Month', weight='bold')
    ax.set_ylabel('Total Sales Quantity', weight='bold')
    ax.set_title('Monthly Sales (2023-2024)', weight='bold')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def show_graph_2(df):
    mean_quantity_per_day = df.groupby('category')['sales_qty'].mean()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(mean_quantity_per_day.index, mean_quantity_per_day.values, color=plt.get_cmap('tab10').colors[:len(mean_quantity_per_day)])
    ax.set_title('Average Sales Quantity per Day', weight='bold')
    ax.set_xlabel('Category', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    plt.grid(axis='y')
    plt.tight_layout()
    return fig

def show_graph_3(df):
    mean_day_sales = df.groupby(df['date'].dt.day)['sales_qty'].mean()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(mean_day_sales.index.astype(str), mean_day_sales.values)
    ax.set_title('Average Sales Quantity per Day of the Month', weight='bold')
    ax.set_xlabel('Day of Month', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    plt.tight_layout()
    return fig

def show_graph_4(df):
    total_sales_per_item = df.groupby('item_name')['sales_qty'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(total_sales_per_item.index, total_sales_per_item.values, color=plt.get_cmap('tab10').colors)
    ax.set_title('Total Sales Quantity per Product', weight='bold')
    ax.set_xlabel('Product Name', weight='bold')
    ax.set_ylabel('Total Sales Quantity', weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def show_graphs(df: pd.DataFrame):
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(show_graph_1(df))
        st.pyplot(show_graph_3(df))
    with col2:
        st.pyplot(show_graph_2(df))
        st.pyplot(show_graph_4(df))

def predict(item_name: str, horizon: int = 30) -> pd.DataFrame | None:
    model_path = os.path.join(model_dir, f'prophet_{item_name}.pkl')
    if not os.path.exists(model_path):
        st.error(f"Модель для '{item_name}' не знайдена.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        st.error(f"Помилка прогнозу: {e}")
        return None

def show_analys_per_name(df: pd.DataFrame):
    unique_items = df['item_name'].unique()
    selected_item = st.selectbox('Select product', unique_items)
    filtered_df = df[df['item_name'] == selected_item]

    sales_by_date = (
        filtered_df
        .groupby(filtered_df['date'].dt.to_period('D'))['sales_qty']
        .sum()
        .reset_index()
    )
    sales_by_date['date'] = sales_by_date['date'].dt.to_timestamp()

    min_date = sales_by_date['date'].min().to_pydatetime()
    max_date = sales_by_date['date'].max().to_pydatetime()
    date_range = st.slider(
        'Select date',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format='YYYY-MM'
    )
    filtered_sales = sales_by_date[
        (sales_by_date['date'] >= date_range[0]) &
        (sales_by_date['date'] <= date_range[1])
    ]

    forecast = predict(selected_item, 30)
    if forecast is None:
        return

    last_date = filtered_df['date'].max()
    forecast_future = forecast[forecast['ds'] > last_date]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(filtered_sales['date'], filtered_sales['sales_qty'], marker='o', label='Historical Sales')
    ax.plot(forecast_future['ds'], forecast_future['yhat'], color='green', label='Forecast')
    ax.fill_between(
        forecast_future['ds'],
        forecast_future['yhat_lower'],
        forecast_future['yhat_upper'],
        color='green', alpha=0.2, label='Confidence Interval'
    )
    ax.set_title(f'Sales and Forecast for {selected_item}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Quantity')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    df_for_show = forecast_future.drop(columns=['trend_lower', 'trend_upper'], errors='ignore')
    df_for_show = df_for_show.reset_index(drop=True)
    df_for_show['From-To'] = df_for_show.apply(
        lambda row: f"{row['yhat_lower']:.0f} - {row['yhat_upper']:.0f}", axis=1
    )
    df_for_show['trend'] = df_for_show.apply(
        lambda row: f"{row['trend']:.0f}" if 'trend' in df_for_show.columns else '', axis=1
    )
    columns_to_keep = ['ds', 'yhat', 'From-To', 'trend']
    df_for_show = df_for_show[columns_to_keep]
    df_for_show = df_for_show.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
    df_for_show['Date'] = df_for_show['Date'].dt.strftime('%Y-%m-%d')

    st.dataframe(df_for_show)

def load_and_prepare_df():
    uploaded_file = st.file_uploader('Choose CSV-file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Data Preview:')
        st.dataframe(df.head())

        cols = df.columns.tolist()
        item_col = st.selectbox('Select column for item_name', cols)
        date_col = st.selectbox('Select column for date', cols)
        sales_col = st.selectbox('Select column for sales_qty', cols)

        df = df.rename(columns={
            item_col: 'item_name',
            date_col: 'date',
            sales_col: 'sales_qty'
        })

        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"Помилка конвертації колонки date в datetime: {e}")
            return None

        df['item_name'] = df['item_name'].astype('category')

        st.success('Дані успішно підготовлені!')
        return df
    else:
        return None

def main():
    st.title("Sales Forecasting (Load models from pickle)")

    df = load_and_prepare_df()
    if df is not None:
        st.write("Prepared Data:")
        st.dataframe(df.head())
        show_graphs(df)
        show_analys_per_name(df)

if __name__ == '__main__':
    main()
