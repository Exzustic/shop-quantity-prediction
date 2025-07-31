import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.prophet_model import ProphetModel

prophet_model: ProphetModel = None

def show_graph_1(df):
    
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales_qty'].sum()

    fig, ax =  plt.subplots(figsize=(10,6))
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
    mean_quantity_holiday = df.groupby('is_holiday')['sales_qty'].mean()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar([0,1], mean_quantity_holiday.values, color=plt.get_cmap('tab10').colors[:2])
    plt.xticks([0, 1], ['Regular Day', 'Holiday'])

    ax.set_xlabel('Day Type', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    ax.set_title('Sales on Regular Days vs Holidays', weight='bold')
    plt.tight_layout()

    return fig


def show_graphs(df: pd.DataFrame):

    col1, col2 = st.columns(2)

    with(col1):
        st.pyplot(show_graph_1(df))
        st.pyplot(show_graph_3(df))

    with(col2):
        st.pyplot(show_graph_2(df))
        st.pyplot(show_graph_4(df))



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
    sales_by_date['date'] = sales_by_date['date'].dt.to_timestamp()  # <- Ключевая строка

    min_date = sales_by_date['date'].min().to_pydatetime()
    max_date = sales_by_date['date'].max().to_pydatetime()
    date_range = st.slider(
        'Select date',
        min_value=min_date,
        max_value=max_date,
        value=(min_date,max_date),
        format='YYYY-MM'
        )
    
    filtered_sales = sales_by_date[
        (sales_by_date['date'] >= date_range[0]) &
        (sales_by_date['date'] <= date_range[1])
    ]

    forecast = prophet_model.predict(selected_item, 30)
    last_date = filtered_df['date'].max()
    forecast_future = forecast[forecast['ds'] > last_date]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(filtered_sales['date'], filtered_sales['sales_qty'], marker='o', label='Historical Sales')

    ax.plot(forecast_future['ds'], forecast_future['yhat'], color='green', label='Forecast')
    ax.fill_between(
        forecast_future['ds'],
        forecast_future['yhat_lower'],
        forecast_future['yhat_upper'],
        color='green', alpha=0.2, label='Confidence Interval'
    )

    ax.set_title(f'Sales and Forecast for {selected_item}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales Quantity')
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)


def upload_csv():
    global prophet_model
    
    if 'df' not in st.session_state or 'prophet_model' not in st.session_state:
        uploaded_file = st.file_uploader('Choose CSV-file', type='csv')

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write('Data Preview:')
            st.dataframe(df)

            df['date']=pd.to_datetime(df['date'])
            df['item_name'] = df['item_name'].astype('category')
            
            prophet_model = ProphetModel()
            prophet_model.train(df)

            st.session_state.df = df
            st.session_state.prophet_model = prophet_model

            show_graphs(df)
            show_analys_per_name(df)    
    else:
        df = st.session_state.df
        prophet_model = st.session_state.prophet_model

        show_graphs(df)
        show_analys_per_name(df)
    

def main():
    st.title('Sales Analysis')
    upload_csv()

if __name__ == '__main__':
    main()
