import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model.prophet_model import ProphetModel

# Global variables to hold data and model instances
prophet_model: ProphetModel = None
df: pd.DataFrame = None


def show_graph_1():
    """
    Plot the total monthly sales trend.
    
    Groups sales quantity by month and displays a line plot with data points.
    Returns a matplotlib figure object.
    """
    global df

    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales_qty'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
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


def show_graph_2():
    """
    Plot the average sales quantity per product category.
    
    Calculates mean sales quantity grouped by category and displays a bar chart.
    Returns a matplotlib figure object.
    """
    global df

    mean_quantity_per_day = df.groupby('category')['sales_qty'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(mean_quantity_per_day.index, mean_quantity_per_day.values,
           color=plt.get_cmap('tab10').colors[:len(mean_quantity_per_day)])
    ax.set_title('Average Sales Quantity per Day', weight='bold')
    ax.set_xlabel('Category', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    plt.grid(axis='y')
    plt.tight_layout()

    return fig


def show_graph_3():
    """
    Plot average sales quantity by day of the month.
    
    Groups sales by the day of the month and plots the mean sales quantity.
    Returns a matplotlib figure object.
    """
    global df

    mean_day_sales = df.groupby(df['date'].dt.day)['sales_qty'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_day_sales.index.astype(str), mean_day_sales.values)
    ax.set_title('Average Sales Quantity per Day of the Month', weight='bold')
    ax.set_xlabel('Day of Month', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    plt.tight_layout()

    return fig


def show_graph_4():
    """
    Compare average sales quantity on holidays vs regular days.
    
    Groups sales by holiday indicator and displays a bar chart comparing averages.
    Returns a matplotlib figure object.
    """
    global df

    mean_quantity_holiday = df.groupby('is_holiday')['sales_qty'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([0, 1], mean_quantity_holiday.values, color=plt.get_cmap('tab10').colors[:2])
    plt.xticks([0, 1], ['Regular Day', 'Holiday'])

    ax.set_xlabel('Day Type', weight='bold')
    ax.set_ylabel('Average Sales Quantity', weight='bold')
    ax.set_title('Sales on Regular Days vs Holidays', weight='bold')
    plt.tight_layout()

    return fig


def show_graphs():
    """
    Display selected sales analysis graphs side by side in Streamlit columns.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(show_graph_1())
        # Uncomment below lines to add more graphs
        # st.pyplot(show_graph_3())

    with col2:
        st.pyplot(show_graph_3())
        # st.pyplot(show_graph_2())
        # st.pyplot(show_graph_4())


def show_analys_per_name():
    """
    Display historical sales and forecast for a selected product.
    
    Allows the user to:
    - Select a product name
    - Specify the number of days to forecast
    - Filter the historical sales data by date range
    Shows a plot with historical sales and forecast, including confidence intervals.
    Also displays forecast data in a table.
    """
    global df

    unique_items = df['item_name'].unique()
    selected_item = st.selectbox('Select product', unique_items)
    selected_date = st.number_input('Write days to predict', min_value=1, max_value=30)

    filtered_df = df[df['item_name'] == selected_item]

    # Aggregate sales by date for the selected product
    sales_by_date = (
        filtered_df
        .groupby(filtered_df['date'].dt.to_period('D'))['sales_qty']
        .sum()
        .reset_index()
    )
    # Convert period to timestamp for plotting and filtering
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

    # Filter historical sales within selected date range
    filtered_sales = sales_by_date[
        (sales_by_date['date'] >= date_range[0]) &
        (sales_by_date['date'] <= date_range[1])
    ]

    # Generate forecast using the trained Prophet model for the selected product and horizon
    forecast = prophet_model.predict(selected_item, selected_date)
    last_date = filtered_df['date'].max()
    forecast_future: pd.DataFrame = forecast[forecast['ds'] > last_date]

    # Plot historical sales and forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_sales['date'], filtered_sales['sales_qty'], marker='o', label='Historical Sales')

    if date_range[1] == max_date:
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

    # Prepare forecast data for display
    df_for_show = forecast_future.drop(columns=['trend_lower', 'trend_upper'], axis=1)
    df_for_show = df_for_show.reset_index()
    df_for_show['From-To'] = df_for_show.apply(
        lambda row: f"{row['yhat_lower']:.0f} - {row['yhat_upper']:.0f}", axis=1
    )
    df_for_show['trend'] = df_for_show.apply(
        lambda row: f"{row['trend']:.0f}", axis=1
    )
    columns = [item for item in df_for_show.columns.to_list() if item not in ['trend', 'ds', 'From-To']]
    df_for_show = df_for_show.drop(columns=columns, axis=1)
    df_for_show = df_for_show.rename(columns={'ds': 'Date'})
    df_for_show['Date'] = df_for_show['Date'].dt.strftime('%Y-%m-%d')

    st.dataframe(df_for_show)


def upload_csv():
    """
    Upload CSV file, allow column selection for analysis, preprocess data,
    train Prophet models, and store results in session state.
    If data and model already exist in session, skip upload and reuse them.
    """
    global prophet_model, df

    # Check if data and model are already loaded in session state
    if 'df' not in st.session_state or 'prophet_model' not in st.session_state:
        uploaded_file = st.file_uploader('Choose CSV-file', type='csv')

        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.write('### Raw Data Preview')
            st.dataframe(df_raw)

            st.write("### Select Columns for Analysis")

            column_options = df_raw.columns.tolist()

            date_col = st.selectbox("Select Date Column", column_options, key='date_col')
            item_col = st.selectbox("Select Product Name Column", column_options, key='item_col')
            qty_col = st.selectbox("Select Sales Quantity Column", column_options, key='qty_col')

            if st.button("Confirm Selection and Process Data"):
                df = df_raw[[date_col, item_col, qty_col]].copy()
                df.columns = ['date', 'item_name', 'sales_qty']

                # Convert data types
                df['date'] = pd.to_datetime(df['date'])
                df['item_name'] = df['item_name'].astype('category')
                df['sales_qty'] = pd.to_numeric(df['sales_qty'], errors='coerce').fillna(0)

                # Initialize and train Prophet model
                prophet_model = ProphetModel()
                prophet_model.train(df)

                # Save processed data and model in session state
                st.session_state.df = df
                st.session_state.prophet_model = prophet_model

                st.success("Data processed successfully!")

                st.write("### Processed Data Preview")
                st.dataframe(df)

                # Show visualizations and analysis
                show_graphs()
                show_analys_per_name()

    else:
        # If data and model already in session, load and display
        df = st.session_state.df
        prophet_model = st.session_state.prophet_model

        show_graphs()
        show_analys_per_name()


def main():
    """
    Main function to display the Streamlit app title and initiate CSV upload workflow.
    """
    st.title('Sales Analysis') 
    upload_csv()

if __name__ == '__main__':
    main()

