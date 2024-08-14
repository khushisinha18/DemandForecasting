


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute
# from pymongo import MongoClient
# import datetime
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import config  # Import the config file

# # Initialize MongoDB client
# client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
# db = client['forecasting_db']  # Database name
# logs_collection = db['logs']  # Collection for logs
# results_collection = db['results']  # Collection for forecast results
# evaluation_metrics_collection = db['evaluation_metrics']  # Collection for evaluation metrics

# # Initialize session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # 0.0 Show Main Page
# st.title('Demand Forecasting App')

# # Button to load a new file
# if not st.session_state.first_run:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.rerun()

# if not st.session_state.hdm.empty:
#     # Sidebar options
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

# # 0.1 Load the Dataset
# if st.session_state.first_run:
#     historical_demand_monthly = prepare_dataset(filename='Online Retail (2).csv', item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         st.rerun()

# if not st.session_state.first_run:
#     historical_demand_monthly = st.session_state.hdm

# # 0.2 Display Main Page
# if historical_demand_monthly is not None:
#     # 1. Get and Display Some Statistics on the Full Dataset
#     total_items = historical_demand_monthly['item'].nunique()
#     last_date = historical_demand_monthly['year_month'].max()
#     first_date = historical_demand_monthly['year_month'].min()

#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)
#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. Ask User Inputs to Run the Forecast
#     if st.session_state.keep_only_positive_demand:
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]

#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')
#     start_from_item = st.sidebar.text_input('Start from item', value='')
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)

#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected != 'All':
#             st.write(f'Monthly demand for item {item_selected}')
#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
#         else:
#             year_months = historical_demand_monthly['year_month'].nunique()

#         periods = st.slider('Number of periods to forecast', min_value=1, max_value=36, value=12)
#         historical_periods = st.slider('Historical periods to analyze', min_value=1, max_value=year_months, value=min(year_months, 48))
#         test_periods = st.slider('Test periods', min_value=0, max_value=year_months, value=min(round(year_months * 0.2), 12))

#         if item_selected == 'All':
#             number_items = st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items + 1, value=min(total_items, 10))

#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         if st.button('Run forecast'):
#             # 3. Run the Forecast
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#             log_data = {
#                 'timestamp': timestamp,
#                 'parameters': {
#                     'Number of periods to forecast': periods,
#                     'Historical periods to analyze': historical_periods,
#                     'Test periods': test_periods,
#                     'Models selected': model_list
#                 }
#             }

#             latest_iteration = st.empty()
#             progress_bar = st.progress(0)

#             if item_selected == 'All':
#                 items = items_ordered[:number_items]
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()

#             for item in items:
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item].copy()
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

#             # Save logs to MongoDB
#             if st.session_state.export_logs:
#                 logs_collection.insert_one(log_data)

#             # Save results to MongoDB
#             if st.session_state.export_results:
#                 results_collection.insert_one({'timestamp': timestamp, 'forecast': forecast.to_dict('records')})
#                 evaluation_metrics_collection.insert_one({'timestamp': timestamp, 'evaluation_metrics': evaluation_metrics.to_dict('records')})

#             # 4. Show the Results
#             st.balloons()
#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 st.write(f'Forecast for item {item_selected}')
#                 st.line_chart(forecast.drop(['item'], axis=1))
#                 if st.session_state.show_forecast_details:                
#                     st.write(forecast.drop(['item'], axis=1))
#                     st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))
#             else:
#                 if st.session_state.show_forecast_details:
#                     st.write('Evaluation metrics (showing first 10,000 rows)')
#                     st.write(evaluation_metrics.head(10000))

#                     st.write('Forecast (showing first 10,000 rows)')
#                     st.write(forecast.head(10000))

#                 st.write('Evaluation metrics summary')
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)

#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))

#             # Email input and sending at the bottom of the page
#             user_email = st.text_input("Enter your email to receive the results", value="")
#             if user_email:
#                 if st.button("Send Email"):
#                     try:
#                         # Send email with results
#                         send_email(user_email, timestamp, forecast, evaluation_metrics)
#                         st.success(f"Results sent to {user_email}")
#                     except Exception as e:
#                         st.error(f"Failed to send email: {str(e)}")


# def send_email(to_email, timestamp, forecast, evaluation_metrics):
#     """Function to send email with forecast and evaluation metrics as content."""
#     from_email = config.EMAIL_ADDRESS
#     password = config.EMAIL_PASSWORD
#     subject = f"Forecast Results - {timestamp}"

#     # Create the email content
#     body = f"""
#     Forecast results and evaluation metrics generated at {timestamp}.
#     Forecast Results: {forecast.head().to_html()}
#     Evaluation Metrics: {evaluation_metrics.head().to_html()}
#     """

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = from_email
#     msg['To'] = to_email
#     msg['Subject'] = subject
#     msg.attach(MIMEText(body, 'html'))

#     # Send the email
#     server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)  # Use your SMTP server and port from config
#     server.starttls()
#     server.login(from_email, password)
#     server.sendmail(from_email, to_email, msg.as_string())
#     server.quit()

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with



# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute
# from pymongo import MongoClient
# import datetime
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import config  # Import the config file

# def send_email(to_email, timestamp, forecast, evaluation_metrics):
#     """Function to send email with forecast and evaluation metrics as content."""
#     from_email = config.EMAIL_ADDRESS
#     password = config.EMAIL_PASSWORD
#     subject = f"Forecast Results - {timestamp}"

#     # Create the email content
#     body = f"""
#     Forecast results and evaluation metrics generated at {timestamp}.
#     Forecast Results: {forecast.head().to_html()}
#     Evaluation Metrics: {evaluation_metrics.head().to_html()}
#     """

#     # Create the email message
#     msg = MIMEMultipart()
#     msg['From'] = from_email
#     msg['To'] = to_email
#     msg['Subject'] = subject
#     msg.attach(MIMEText(body, 'html'))

#     # Send the email
#     server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)  # Use your SMTP server and port from config
#     server.starttls()
#     server.login(from_email, password)
#     server.sendmail(from_email, to_email, msg.as_string())
#     server.quit()

# # Initialize MongoDB client
# client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
# db = client['forecasting_db']  # Database name
# logs_collection = db['logs']  # Collection for logs
# results_collection = db['results']  # Collection for forecast results
# evaluation_metrics_collection = db['evaluation_metrics']  # Collection for evaluation metrics

# # Initialize session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # 0.0 Show Main Page
# st.title('Demand Forecasting App')

# # Button to load a new file
# if not st.session_state.first_run:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.rerun()

# if not st.session_state.hdm.empty:
#     # Sidebar options
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

# # 0.1 Load the Dataset
# if st.session_state.first_run:
#     historical_demand_monthly = prepare_dataset(filename='Online Retail (2).csv', item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         st.rerun()

# if not st.session_state.first_run:
#     historical_demand_monthly = st.session_state.hdm

# # 0.2 Display Main Page
# if historical_demand_monthly is not None:
#     # 1. Get and Display Some Statistics on the Full Dataset
#     total_items = historical_demand_monthly['item'].nunique()
#     last_date = historical_demand_monthly['year_month'].max()
#     first_date = historical_demand_monthly['year_month'].min()

#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)
#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. Ask User Inputs to Run the Forecast
#     if st.session_state.keep_only_positive_demand:
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]

#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')
#     start_from_item = st.sidebar.text_input('Start from item', value='')
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)

#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected != 'All':
#             st.write(f'Monthly demand for item {item_selected}')
#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
#         else:
#             year_months = historical_demand_monthly['year_month'].nunique()

#         periods = st.slider('Number of periods to forecast', min_value=1, max_value=36, value=12)
#         historical_periods = st.slider('Historical periods to analyze', min_value=1, max_value=year_months, value=min(year_months, 48))
#         test_periods = st.slider('Test periods', min_value=0, max_value=year_months, value=min(round(year_months * 0.2), 12))

#         if item_selected == 'All':
#             number_items = st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items + 1, value=min(total_items, 10))

#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         if st.button('Run forecast'):
#             # 3. Run the Forecast
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#             log_data = {
#                 'timestamp': timestamp,
#                 'parameters': {
#                     'Number of periods to forecast': periods,
#                     'Historical periods to analyze': historical_periods,
#                     'Test periods': test_periods,
#                     'Models selected': model_list
#                 }
#             }

#             latest_iteration = st.empty()
#             progress_bar = st.progress(0)

#             if item_selected == 'All':
#                 items = items_ordered[:number_items]
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()

#             for item in items:
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item].copy()
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

#             # Save logs to MongoDB
#             if st.session_state.export_logs:
#                 logs_collection.insert_one(log_data)

#             # Save results to MongoDB
#             if st.session_state.export_results:
#                 results_collection.insert_one({'timestamp': timestamp, 'forecast': forecast.to_dict('records')})
#                 evaluation_metrics_collection.insert_one({'timestamp': timestamp, 'evaluation_metrics': evaluation_metrics.to_dict('records')})

#             # 4. Show the Results
#             st.balloons()
#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 st.write(f'Forecast for item {item_selected}')
#                 st.line_chart(forecast.drop(['item'], axis=1))
#                 if st.session_state.show_forecast_details:                
#                     st.write(forecast.drop(['item'], axis=1))
#                     st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))
#             else:
#                 if st.session_state.show_forecast_details:
#                     st.write('Evaluation metrics (showing first 10,000 rows)')
#                     st.write(evaluation_metrics.head(10000))

#                     st.write('Forecast (showing first 10,000 rows)')
#                     st.write(forecast.head(10000))

#                 st.write('Evaluation metrics summary')
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)

#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))

#             # Email input and sending at the bottom of the page
#             user_email = st.text_input("Enter your email to receive the results", value="")
#             if user_email:
#                 if st.button("Send Email"):
#                     try:
#                         # Send email with results
#                         send_email(user_email, timestamp, forecast, evaluation_metrics)
#                         st.success(f"Results sent to {user_email}")
#                     except Exception as e:
#                         st.error(f"Failed to send email: {str(e)}")


# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with

import streamlit as st
import pandas as pd
import os
from forecasting_dataset_load import prepare_dataset
from forecasting_dataset_statistics import statistics as stat
from forecasting_compute import forecast as forecast_compute
from pymongo import MongoClient
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import config  # Importing the config file

# Function to send email
def send_email(to_email, subject, body):
    from_email = config.EMAIL_ADDRESS  # Get the email from config.py
    password = config.EMAIL_PASSWORD  # Get the email password from config.py

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        st.write("Attempting to send email...")  # Debugging line
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        st.success(f"Email sent to {to_email}")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client['forecasting_db']  # Database name
logs_collection = db['logs']  # Collection for logs
results_collection = db['results']  # Collection for forecast results
evaluation_metrics_collection = db['evaluation_metrics']  # Collection for evaluation metrics

# Initialize session state variables
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = True

if 'hdm' not in st.session_state:
    st.session_state.hdm = pd.DataFrame()

if 'statistics' not in st.session_state:
    st.session_state.statistics = pd.DataFrame()

# 0.0 Show Main Page
st.title('Demand Forecasting App')

# Button to load a new file
if not st.session_state.first_run:
    if st.sidebar.button('Load a new file'):
        st.session_state.first_run = True
        st.experimental_rerun()

if not st.session_state.hdm.empty:
    # Sidebar options
    show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
    st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
    st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
    st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
    seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
    trend_check = st.sidebar.checkbox('Look for trend', value=True)
    st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
    order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

# 0.1 Load the Dataset
if st.session_state.first_run:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        historical_demand_monthly = prepare_dataset(filename=uploaded_file, item='All', rows='All')
        if historical_demand_monthly is not None:
            st.session_state.hdm = historical_demand_monthly
            st.session_state.first_run = False
            st.experimental_rerun()

if not st.session_state.first_run:
    historical_demand_monthly = st.session_state.hdm

# 0.2 Display Main Page
if historical_demand_monthly is not None:
    # 1. Get and Display Some Statistics on the Full Dataset
    total_items = historical_demand_monthly['item'].nunique()
    last_date = historical_demand_monthly['year_month'].max()
    first_date = historical_demand_monthly['year_month'].min()

    if st.session_state.statistics.empty:
        st.session_state.statistics = stat(historical_demand_monthly, display=False)
    statistics = st.session_state.statistics

    if show_statistics:
        st.write('Dataset statistics')
        stat(historical_demand_monthly, statistics, display=True)

    # 2. Ask User Inputs to Run the Forecast
    if st.session_state.keep_only_positive_demand:
        items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
        historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]

    if order_by == 'Total demand':
        items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
    else:
        items_ordered = historical_demand_monthly['item'].unique().tolist()

    item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)
    skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')
    start_from_item = st.sidebar.text_input('Start from item', value='')
    last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)

    historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
    total_items = historical_demand_monthly['item'].nunique()

    if item_selected != '':
        if item_selected != 'All':
            st.write(f'Monthly demand for item {item_selected}')
            st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])
            year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        else:
            year_months = historical_demand_monthly['year_month'].nunique()

        periods = st.slider('Number of periods to forecast', min_value=1, max_value=36, value=12)
        historical_periods = st.slider('Historical periods to analyze', min_value=1, max_value=year_months, value=min(year_months, 48))
        test_periods = st.slider('Test periods', min_value=0, max_value=year_months, value=min(round(year_months * 0.2), 12))

        if item_selected == 'All':
            number_items = st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items + 1, value=min(total_items, 10))

        model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

        if st.button('Run forecast'):
            # 3. Run the Forecast
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            log_data = {
                'timestamp': timestamp,
                'parameters': {
                    'Number of periods to forecast': periods,
                    'Historical periods to analyze': historical_periods,
                    'Test periods': test_periods,
                    'Models selected': model_list
                }
            }

            latest_iteration = st.empty()
            progress_bar = st.progress(0)

            if item_selected == 'All':
                items = items_ordered[:number_items]
                if skip_items != '':
                    items = [item for item in items if item not in skip_items.split(',')]
                if start_from_item != '':
                    items = items[items.index(start_from_item):]
            else:
                items = [item_selected]

            forecast = pd.DataFrame()
            evaluation_metrics = pd.DataFrame()

            for item in items:
                df = historical_demand_monthly[historical_demand_monthly['item'] == item].copy()
                forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

                forecast_item['item'] = item
                evaluation_metrics_item['item'] = item

                forecast = pd.concat([forecast, forecast_item])
                evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

                progress_bar.progress((items.index(item) + 1) / len(items))
                latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

            # Save logs to MongoDB
            if st.session_state.export_logs:
                logs_collection.insert_one(log_data)

            # Save results to MongoDB
            if st.session_state.export_results:
                results_collection.insert_one({'timestamp': timestamp, 'forecast': forecast.to_dict('records')})
                evaluation_metrics_collection.insert_one({'timestamp': timestamp, 'evaluation_metrics': evaluation_metrics.to_dict('records')})

            # 4. Show the Results
            st.balloons()
            st.write('Results')
            st.write('Number of items processed: ', len(items))

            if item_selected != 'All':
                st.write(f'Forecast for item {item_selected}')
                st.line_chart(forecast.drop(['item'], axis=1))
                if st.session_state.show_forecast_details:                
                    st.write(forecast.drop(['item'], axis=1))
                    st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))
            else:
                if st.session_state.show_forecast_details:
                    st.write('Evaluation metrics (showing first 10,000 rows)')
                    st.write(evaluation_metrics.head(10000))

                    st.write('Forecast (showing first 10,000 rows)')
                    st.write(forecast.head(10000))

                st.write('Evaluation metrics summary')
                evaluation_metrics_summary = evaluation_metrics.reset_index()

                if st.session_state.show_forecast_details:
                    st.write(evaluation_metrics_summary.head(10000))

                evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
                evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
                st.write(evaluation_metrics_summary)

                evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
                st.write(evaluation_metrics_summary)
                st.bar_chart(evaluation_metrics_summary.set_index('Best model'))

            # Email input and sending at the bottom of the page
            user_email = st.text_input("Enter your email to receive the results", value="")
            send_email_button = st.button("Send Email")  # Add a distinct send email button

            if send_email_button and user_email:
                try:
                    subject = f"Forecast Results - {timestamp}"
                    body = f"Forecast results and evaluation metrics generated at {timestamp}."
                    send_email(user_email, subject, body)
                    st.write("Send email function was called!")  # Debugging line
                except Exception as e:
                    st.error(f"Failed to send email: {str(e)}")

