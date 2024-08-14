
# ##run this file to start the program
# ##will create an interative window with streamlit
# ##user will be able to explore time series and apply different algorithms

# import streamlit as st
# import pandas as pd
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# import os

# #intialize the session state variable
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# #0.0 SHOW MAIN PAGE

# st.write('Demand Forecasting App')

# #show a button "load a new file" to load a new file
# if st.session_state.first_run == False:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.experimental_rerun()

# if len(st.session_state.hdm) > 0:

#     #add a selectbox on the left of the screen to select if showing statistics for the full dataset
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

#     #add a selectbox on the left to select if showing forecast details
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

#     #add a selectbox on the left to select if exporting logs
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs ', value=False)

#     #add a selectbox on the left to select if exporting logs and datasets as txt files
#     st.session_state.export_results = st.sidebar.checkbox('Export results ', value=False)

#     #add a selectbox on the left to select if models should look for a seasonal component
#     seasonal_check = st.sidebar.checkbox('Look for seasonality ', value=True)

#     #add a selectbox on the left to select if models should look for a trend component
#     trend_check = st.sidebar.checkbox('Look for trend ', value=True)

#     #ask if keeping only items with positive demand in the last 12 months
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

#     #add a selectbox to select if showing items ordered by total demand or item number
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


# #0.1 LOAD THE DATASET

# if st.session_state.first_run:
#     historical_demand_monthly = prepare_dataset(filename='demand.csv', item='All',  rows='All')  ##item='All', rows='All' are deafult
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         #rerun the program
#         st.experimental_rerun()

# if st.session_state.first_run == False:    
#     historical_demand_monthly=st.session_state.hdm


# #0.2 DISPLAY MAIN PAGE

# if historical_demand_monthly is not None: #check if proper dataset was loaded

#     #1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

#     #get the number of total items
#     total_items = historical_demand_monthly['item'].nunique()

#     #get the last data
#     last_date = historical_demand_monthly['year_month'].max()

#     #get the fist data
#     first_date = historical_demand_monthly['year_month'].min()

#     #check if st.session_state.statistics is empty
#     if len(st.session_state.statistics) == 0:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)

#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)


 
#     #2. ASK THE USER INPUTS TO RUN THE FORECAST
    
#     if st.session_state.keep_only_positive_demand:

#         #from statistics, get the list of items where 'last 12 months with positive demand' is True
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     #add a searcheable selectbox to select the item to be shown
#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

#     #add the option to skip specific items
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

#     #add the option to start the forecast from a specific item
#     start_from_item = st.sidebar.text_input('Start from item', value='')

#     #ask the user to specify the last year_month of the historical period
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()



#     if item_selected != '':
#         if item_selected !='All':
#             #show the monthly demand for the selected item
#             st.write('Monthly demand for item', item_selected)
#             st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

#             #count the the year_months in the period for the selected item
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
#         else:
#             #count the the year_months in the period for all items
#             year_months = historical_demand_monthly['year_month'].nunique()

#         #ask user to define the number of periods to forecast
#         periods=st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
#         if year_months > 1:
#             historical_periods=st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
#         else:
#             historical_periods=1
        
#         test_periods=st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months*0.2),12))
#         #allow to restrict the number of items to be forecasted to 1000
#         if item_selected == 'All':
#             number_items=st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items+1, key='number_items', value=min(total_items, 10))
        
#         #ask the user to choose if using ARIMA, Prophet or both
#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         #show a button to run the forecast
#         if st.button('Run forecast'):

#             #3. RUN THE FORECAST

#             #3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             #check if the log folder exists otherwise create it
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write('Number of periods to forecast: ' + str(periods) + '\n')
#                 f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
#                 f.write('Test periods: ' + str(test_periods) + '\n')
#                 f.write('Models selected: ' + str(model_list) + '\n')
            
#             #3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
#             #create a progress bar object
#             latest_iteration = st.empty() #create a placeholder
#             progress_bar = st.progress(0)
        

#             #creates the real items list based on the user input
#             if item_selected == 'All':
#                 items = items_ordered[:number_items] #apply a cutoff based on the sorting criteria and the number of items selected by the user
#                 #remove the items to be skipped
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             #inizialize the forecast and evaluation_metrics as pandas dataframes
#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()


#             for item in items:
#                 #3.3. PREPARE DATA FOR THE SPECIFIC ITEM

#                 #select data for the selected item
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]

#                 #restrict to the time range selected by the user
#                 df = df.tail(historical_periods)

#                 #drop the column item
#                 df = df.drop(['item'], axis=1)


#                 #3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

#                 #show progress
#                 item_position = items.index(item)
#                 progress_bar.progress((item_position+1)/len(items))
#                 latest_iteration.text('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)))

#                 #log progress
#                 start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

#                 if st.session_state.export_logs:
#                     with open(log_file, 'a') as f:
#                         f.write('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)) +': ' + item + ' started at ' + start_time +  '\n')

#                 #run the forecast and get the evaluation metrics
#                 from forecasting_compute import forecast as forecast_compute
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 #add a column with the item number to the forecast and evaluation_metrics
#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 #append forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
#                 forecast = forecast.append(forecast_item)
#                 evaluation_metrics = evaluation_metrics.append(evaluation_metrics_item)

#                 if st.session_state.export_logs:
#                     #get the timestamp for the end of the forecast
#                     end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

#                     #write the evaluation metrics to the log file
#                     with open(log_file, 'a') as f:
#                         f.write('Forecast completed at ' + end_time +  '\n')
#                         f.write('Evaluation metrics: ' + str(evaluation_metrics_item) +  '\n')  

#             #4. SHOW THE RESULTS

#             st.balloons()

#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 #display data for the selected item only
#                 st.write ('Forecast for item', item)
                
#                 st.line_chart(forecast.drop(['item'], axis=1))
#                 if st.session_state.show_forecast_details:                
#                     st.write(forecast.drop(['item'], axis=1))
#                     st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))

#             else:
#                 if st.session_state.show_forecast_details:
#                     st.write('Evaluation metrics (showing first 10000 rows')
#                     st.write(evaluation_metrics.head(10000))

#                     st.write('Forecast (showing first 10000 rows')
#                     st.write(forecast.head(10000))
                
            
#                 st.write('Evaluation metrics summary')
#                 #reset evaluation_metrics index
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 #select only the records with index = 'maape'
#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 #keep only the columns item and best model
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)
#                 #show a barchart showing for each item the best model
#                 #groupby item and best model and count the number of records
#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 #plot a bar chart with Best model on the x axis and count on the y axis
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                

#                 #5. EXPORT THE RESULTS TO CSV FILES

#                 #export to a text file the forecast and the evaluation metrics
#                 #check if forecast_results folder exists, if not create it
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
#                     forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
#                     evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'
    

#                     #export the forecast and the evaluation metrics to csv files (with time stamp)
#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     #overwrite forecast and eval files
#                     #export the forecast and the evaluation metrics to csv files (without time stamp)
#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

                


# # ##IDEAS FOR IMPROVEMENTS
# # #additional statistics for the dataset
# # #show results with total forecast for each item with the best model for the period

# ## Run this file to start the program
# ## Will create an interactive window with Streamlit
# ## User will be able to explore time series and apply different algorithms

# import streamlit as st
# import pandas as pd
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# import os

# # Initialize the session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # 0.0 SHOW MAIN PAGE
# st.write('Demand Forecasting App')

# # Show a button "load a new file" to load a new file
# if not st.session_state.first_run:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.rerun()

# if not st.session_state.hdm.empty:

#     # Add a checkbox on the left of the screen to show dataset statistics
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

#     # Add a checkbox on the left to show forecast details
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

#     # Add a checkbox on the left to export logs
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)

#     # Add a checkbox on the left to export results as txt files
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)

#     # Add a checkbox on the left to look for a seasonal component in models
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)

#     # Add a checkbox on the left to look for a trend component in models
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)

#     # Checkbox to keep only items with positive demand in the last 12 months
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

#     # Add a selectbox to order items by total demand or item number
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


# # 0.1 LOAD THE DATASET
# if st.session_state.first_run:
#     historical_demand_monthly = prepare_dataset(filename='demand.csv', item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         # Rerun the program
#         st.rerun()

# if not st.session_state.first_run:    
#     historical_demand_monthly = st.session_state.hdm

# # 0.2 DISPLAY MAIN PAGE
# if historical_demand_monthly is not None:  # Check if a proper dataset was loaded

#     # 1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

#     # Get the number of total items
#     total_items = historical_demand_monthly['item'].nunique()

#     # Get the last date in the dataset
#     last_date = historical_demand_monthly['year_month'].max()

#     # Get the first date in the dataset
#     first_date = historical_demand_monthly['year_month'].min()

#     # Check if st.session_state.statistics is empty
#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)

#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. ASK THE USER INPUTS TO RUN THE FORECAST
    
#     if st.session_state.keep_only_positive_demand:
#         # Get the list of items where 'last 12 months with positive demand' is True
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     # Add a searchable selectbox to select the item to be shown
#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

#     # Add the option to skip specific items
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

#     # Add the option to start the forecast from a specific item
#     start_from_item = st.sidebar.text_input('Start from item', value='')

#     # Ask the user to specify the last year_month of the historical period
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected != 'All':
#             # Show the monthly demand for the selected item
#             st.write('Monthly demand for item', item_selected)
#             st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

#             # Count the year_months in the period for the selected item
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
#         else:
#             # Count the year_months in the period for all items
#             year_months = historical_demand_monthly['year_month'].nunique()

#         # Ask user to define the number of periods to forecast
#         periods = st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
#         if year_months > 1:
#             historical_periods = st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
#         else:
#             historical_periods = 1
        
#         test_periods = st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months * 0.2), 12))
        
#         # Allow restricting the number of items to be forecasted to 1000
#         if item_selected == 'All':
#             number_items = st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items + 1, key='number_items', value=min(total_items, 10))
        
#         # Ask the user to choose if using ARIMA, Prophet, or both
#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         # Show a button to run the forecast
#         if st.button('Run forecast'):

#             # 3. RUN THE FORECAST

#             # 3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             # Check if the log folder exists otherwise create it
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write('Number of periods to forecast: ' + str(periods) + '\n')
#                 f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
#                 f.write('Test periods: ' + str(test_periods) + '\n')
#                 f.write('Models selected: ' + str(model_list) + '\n')
            
#             # 3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
#             # Create a progress bar object
#             latest_iteration = st.empty()  # Create a placeholder
#             progress_bar = st.progress(0)
        
#             # Create the real items list based on the user input
#             if item_selected == 'All':
#                 items = items_ordered[:number_items]  # Apply a cutoff based on the sorting criteria and the number of items selected by the user
#                 # Remove the items to be skipped
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             # Initialize the forecast and evaluation_metrics as pandas dataframes
#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()

#             for item in items:
#                 # 3.3. PREPARE DATA FOR THE SPECIFIC ITEM

#                 # Select data for the selected item
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]

#                 # Restrict to the time range selected by the user
#                 df = df.tail(historical_periods)

#                 # Drop the column item
#                 df = df.drop(['item'], axis=1)

#                 # 3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

#                 # Show progress
#                 item_position = items.index(item)
#                 progress_bar.progress((item_position + 1) / len(items))
#                 latest_iteration.text('Forecasting item ' + str(item_position + 1) + ' of ' + str(len(items)))

#                 # Log progress
#                 start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

#                 if st.session_state.export_logs:
#                     with open(log_file, 'a') as f:
#                         f.write('Forecasting item ' + str(item_position + 1) + ' of ' + str(len(items)) + ': ' + item + ' started at ' + start_time + '\n')

#                 # Run the forecast and get the evaluation metrics
#                 from forecasting_compute import forecast as forecast_compute
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 # Add a column with the item number to the forecast and evaluation_metrics
#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 # Concatenate forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
#                 forecast = pd.concat([forecast, forecast_item], ignore_index=True)
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item], ignore_index=True)

#                 if st.session_state.export_logs:
#                     # Get the timestamp for the end of the forecast
#                     end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

#                     # Write the evaluation metrics to the log file
#                     with open(log_file, 'a') as f:
#                         f.write('Forecast completed at ' + end_time + '\n')
#                         f.write('Evaluation metrics: ' + str(evaluation_metrics_item) + '\n')  

#             # 4. SHOW THE RESULTS

#             st.balloons()

#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 # Display data for the selected item only
#                 st.write('Forecast for item', item)
                
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
#                 # Reset evaluation_metrics index
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 # Select only the records with index = 'maape'
#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 # Keep only the columns item and best model
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)
#                 # Show a barchart showing for each item the best model
#                 # Group by item and best model and count the number of records
#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 # Plot a bar chart with Best model on the x-axis and count on the y-axis
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                
#                 # 5. EXPORT THE RESULTS TO CSV FILES

#                 # Export to a text file the forecast and the evaluation metrics
#                 # Check if forecast_results folder exists, if not create it
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
#                     forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
#                     evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'

#                     # Export the forecast and the evaluation metrics to csv files (with timestamp)
#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     # Overwrite forecast and eval files
#                     # Export the forecast and the evaluation metrics to csv files (without timestamp)
#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## IDEAS FOR IMPROVEMENTS
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period




# ##IDEAS FOR IMPROVEMENTS
# #additional statistics for the dataset
# #show results with total forecast for each item with the best model for the period

## Run this file to start the program
## Will create an interactive window with Streamlit
## User will be able to explore time series and apply different algorithms

# import streamlit as st
# import pandas as pd
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# import os

# # Initialize the session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # 0.0 SHOW MAIN PAGE
# st.write('Demand Forecasting App')

# # Show a button "load a new file" to load a new file
# if not st.session_state.first_run:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.rerun()

# if not st.session_state.hdm.empty:

#     # Add a checkbox on the left of the screen to show dataset statistics
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

#     # Add a checkbox on the left to show forecast details
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

#     # Add a checkbox on the left to export logs
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)

#     # Add a checkbox on the left to export results as txt files
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)

#     # Add a checkbox on the left to look for a seasonal component in models
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)

#     # Add a checkbox on the left to look for a trend component in models
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)

#     # Checkbox to keep only items with positive demand in the last 12 months
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

#     # Add a selectbox to order items by total demand or item number
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


# # 0.1 LOAD THE DATASET
# if st.session_state.first_run:
#     historical_demand_monthly = prepare_dataset(filename='demand.csv', item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         # Rerun the program
#         st.rerun()

# if not st.session_state.first_run:    
#     historical_demand_monthly = st.session_state.hdm

# # 0.2 DISPLAY MAIN PAGE
# if historical_demand_monthly is not None:  # Check if a proper dataset was loaded

#     # 1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

#     # Get the number of total items
#     total_items = historical_demand_monthly['item'].nunique()

#     # Get the last date in the dataset
#     last_date = historical_demand_monthly['year_month'].max()

#     # Get the first date in the dataset
#     first_date = historical_demand_monthly['year_month'].min()

#     # Check if st.session_state.statistics is empty
#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)

#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. ASK THE USER INPUTS TO RUN THE FORECAST
    
#     if st.session_state.keep_only_positive_demand:
#         # Get the list of items where 'last 12 months with positive demand' is True
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     # Add a searchable selectbox to select the item to be shown
#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

#     # Add the option to skip specific items
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

#     # Add the option to start the forecast from a specific item
#     start_from_item = st.sidebar.text_input('Start from item', value='')

#     # Ask the user to specify the last year_month of the historical period
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected != 'All':
#             # Show the monthly demand for the selected item
#             st.write('Monthly demand for item', item_selected)
#             st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

#             # Count the year_months in the period for the selected item
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
#         else:
#             # Count the year_months in the period for all items
#             year_months = historical_demand_monthly['year_month'].nunique()

#         # Ask user to define the number of periods to forecast
#         periods = st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
#         if year_months > 1:
#             historical_periods = st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
#         else:
#             historical_periods = 1
        
#         test_periods = st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months * 0.2), 12))
        
#         # Allow restricting the number of items to be forecasted to 1000
#         if item_selected == 'All':
#             number_items = st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items + 1, key='number_items', value=min(total_items, 10))
        
#         # Ask the user to choose if using ARIMA, Prophet, or both
#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         # Show a button to run the forecast
#         if st.button('Run forecast'):

#             # 3. RUN THE FORECAST

#             # 3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             # Check if the log folder exists otherwise create it
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write('Number of periods to forecast: ' + str(periods) + '\n')
#                 f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
#                 f.write('Test periods: ' + str(test_periods) + '\n')
#                 f.write('Models selected: ' + str(model_list) + '\n')
            
#             # 3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
#             # Create a progress bar object
#             latest_iteration = st.empty()  # Create a placeholder
#             progress_bar = st.progress(0)
        
#             # Create the real items list based on the user input
#             if item_selected == 'All':
#                 items = items_ordered[:number_items]  # Apply a cutoff based on the sorting criteria and the number of items selected by the user
#                 # Remove the items to be skipped
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             # Initialize the forecast and evaluation_metrics as pandas dataframes
#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()

#             for item in items:
#                 # 3.3. PREPARE DATA FOR THE SPECIFIC ITEM

#                 # Select data for the selected item
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]

#                 # Restrict to the time range selected by the user
#                 df = df.tail(historical_periods)

#                 # Drop the column item
#                 df = df.drop(['item'], axis=1)

#                 # 3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

#                 # Show progress
#                 item_position = items.index(item)
#                 progress_bar.progress((item_position + 1) / len(items))
#                 latest_iteration.text('Forecasting item ' + str(item_position + 1) + ' of ' + str(len(items)))

#                 # Log progress
#                 start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

#                 if st.session_state.export_logs:
#                     with open(log_file, 'a') as f:
#                         f.write('Forecasting item ' + str(item_position + 1) + ' of ' + str(len(items)) + ': ' + item + ' started at ' + start_time + '\n')

#                 # Run the forecast and get the evaluation metrics
#                 from forecasting_compute import forecast as forecast_compute
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 # Add a column with the item number to the forecast and evaluation_metrics
#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 # Concatenate forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
#                 forecast = pd.concat([forecast, forecast_item], ignore_index=True)
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item], ignore_index=True)

#                 if st.session_state.export_logs:
#                     # Get the timestamp for the end of the forecast
#                     end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

#                     # Write the evaluation metrics to the log file
#                     with open(log_file, 'a') as f:
#                         f.write('Forecast completed at ' + end_time + '\n')
#                         f.write('Evaluation metrics: ' + str(evaluation_metrics_item) + '\n')  

#             # 4. SHOW THE RESULTS

#             st.balloons()

#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 # Display data for the selected item only
#                 st.write('Forecast for item', item)
                
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
#                 # Reset evaluation_metrics index
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 # Select only the records with index = 'maape'
#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 # Keep only the columns item and best model
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)
#                 # Show a barchart showing for each item the best model
#                 # Group by item and best model and count the number of records
#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 # Plot a bar chart with Best model on the x-axis and count on the y-axis
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                
#                 # 5. EXPORT THE RESULTS TO CSV FILES

#                 # Export to a text file the forecast and the evaluation metrics
#                 # Check if forecast_results folder exists, if not create it
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
#                     forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
#                     evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'

#                     # Export the forecast and the evaluation metrics to csv files (with timestamp)
#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     # Overwrite forecast and eval files
#                     # Export the forecast and the evaluation metrics to csv files (without timestamp)
#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## IDEAS FOR IMPROVEMENTS
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 forecast = pd.concat([forecast, forecast_item])
#                 evaluation_metrics = pd.concat([evaluation_metrics, evaluation_metrics_item])

#                 progress_bar.progress((items.index(item) + 1) / len(items))
#                 latest_iteration.text(f'Forecasting item {items.index(item) + 1} of {len(items)}')

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
#                 evaluation_metrics_summary = evaluation_metrics.reset_index(drop=True)

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 evaluation_metrics_summary = evaluation_metrics_summary.loc[evaluation_metrics_summary['index'] == 'maape']
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)

#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# # Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period

# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#     historical_demand_monthly = prepare_dataset(filename=None, item='All', rows='All')
    
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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

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
# import config

# # MongoDB Atlas setup
# client = MongoClient(config.MONGODB_URI)
# db = client[config.DB_NAME]
# forecast_collection = db['forecast_results']
# evaluation_collection = db['evaluation_metrics']

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

# # 0.1 File Upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Read the uploaded CSV file into a DataFrame
#     df = pd.read_csv(uploaded_file)
    
#     # Simulate saving to a temporary file (if needed)
#     temp_file_path = "/tmp/temp_uploaded_file.csv"
#     df.to_csv(temp_file_path, index=False)
    
#     # Load the dataset using the temporary file path
#     st.session_state.first_run = True
#     historical_demand_monthly = prepare_dataset(filename=temp_file_path, item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         st.rerun()

# if not st.session_state.first_run:
#     historical_demand_monthly = st.session_state.hdm

# # 0.2 Display Main Page
# if historical_demand_monthly is not None:
#     # Sidebar options
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

#     # Display dataset statistics and forecasts
#     total_items = historical_demand_monthly['item'].nunique()
#     last_date = historical_demand_monthly['year_month'].max()
#     first_date = historical_demand_monthly['year_month'].min()

#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)
#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

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
#             # Run the Forecast
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#             # Insert results into MongoDB Atlas
#             forecast_records = forecast.to_dict('records')
#             evaluation_records = evaluation_metrics.to_dict('records')
#             forecast_collection.insert_many(forecast_records)
#             evaluation_collection.insert_many(evaluation_records)

#             # Show the Results
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

#                 # Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute
# from pymongo import MongoClient
# import config

# # MongoDB Atlas setup
# client = MongoClient(config.MONGODB_URI)
# db = client[config.DB_NAME]
# forecast_collection = db['forecast_results']
# evaluation_collection = db['evaluation_metrics']

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

# # 0.1 File Upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Read the uploaded CSV file into a DataFrame
#     df = pd.read_csv(uploaded_file)
    
#     # Simulate saving to a temporary file (if needed)
#     temp_file_path = "/tmp/temp_uploaded_file.csv"
#     df.to_csv(temp_file_path, index=False)
    
#     # Load the dataset using the temporary file path
#     st.session_state.first_run = True
#     historical_demand_monthly = prepare_dataset(filename=temp_file_path, item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         st.rerun()

# if not st.session_state.first_run:
#     historical_demand_monthly = st.session_state.hdm
# else:
#     historical_demand_monthly = None

# # 0.2 Display Main Page
# if historical_demand_monthly is not None:
#     # Sidebar options
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

#     # Display dataset statistics and forecasts
#     total_items = historical_demand_monthly['item'].nunique()
#     last_date = historical_demand_monthly['year_month'].max()
#     first_date = historical_demand_monthly['year_month'].min()

#     if st.session_state.statistics.empty:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)
#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

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
#             # Run the Forecast
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#             # Insert results into MongoDB Atlas
#             forecast_records = forecast.to_dict('records')
#             evaluation_records = evaluation_metrics.to_dict('records')
#             forecast_collection.insert_many(forecast_records)
#             evaluation_collection.insert_many(evaluation_records)

#             # Show the Results
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

#                 # Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)
# else:
#     st.write("Please upload a CSV file to proceed with forecasting.")


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute
# from pymongo import MongoClient
# import config
# import datetime

# # MongoDB Atlas setup
# client = MongoClient(config.MONGODB_URI)
# db = client[config.DB_NAME]
# forecast_collection = db['forecast_results']
# evaluation_collection = db['evaluation_metrics']
# logs_collection = db['logs']
# lighting_logs_collection = db['lighting_logs']

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

# # Function to handle large file uploads
# def load_data_in_chunks(file):
#     chunk_size = 10000  # Define your chunk size
#     chunks = []
#     for chunk in pd.read_csv(file, chunksize=chunk_size):
#         # Process chunk here if necessary
#         chunks.append(chunk)
#     return pd.concat(chunks)

# # File upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     with st.spinner('Processing...'):
#         # Load the dataset in chunks
#         df = load_data_in_chunks(uploaded_file)
    
#     # Simulate saving to a temporary file (if needed)
#     temp_file_path = "/tmp/temp_uploaded_file.csv"
#     df.to_csv(temp_file_path, index=False)
    
#     # Load the dataset using the temporary file path
#     st.session_state.first_run = True
#     historical_demand_monthly = prepare_dataset(filename=temp_file_path, item='All', rows='All')
    
#     if historical_demand_monthly is not None:
#         st.session_state.hdm = historical_demand_monthly
#         st.session_state.first_run = False
#         st.rerun()
# else:
#     st.write("Please upload a CSV file to proceed.")

# if not st.session_state.first_run:
#     historical_demand_monthly = st.session_state.hdm
# else:
#     historical_demand_monthly = None

# # 0.2 Display Main Page
# if historical_demand_monthly is not None:
#     # Sidebar options
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs', value=False)
#     st.session_state.export_results = st.sidebar.checkbox('Export results', value=False)
#     seasonal_check = st.sidebar.checkbox('Look for seasonality', value=True)
#     trend_check = st.sidebar.checkbox('Look for trend', value=True)
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])

#     # Display dataset statistics and forecasts
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
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
#             lighting_log_file = 'log/lighting_log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

#             with open(lighting_log_file, 'w') as f:
#                 f.write('Lighting log file created at ' + timestamp + '\n')
#                 f.write('Lighting parameters: ...\n')  # Add relevant lighting parameters if available

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

#             # Insert results into MongoDB Atlas
#             forecast_records = forecast.to_dict('records')
#             evaluation_records = evaluation_metrics.to_dict('records')
#             forecast_collection.insert_many(forecast_records)
#             evaluation_collection.insert_many(evaluation_records)

#             # Save logs to MongoDB
#             with open(log_file, 'r') as f:
#                 log_data = f.read()
#                 logs_collection.insert_one({"timestamp": timestamp, "log": log_data})

#             with open(lighting_log_file, 'r') as f:
#                 lighting_log_data = f.read()
#                 lighting_logs_collection.insert_one({"timestamp": timestamp, "lighting_log": lighting_log_data})

# else:
#     st.write("Please upload a CSV file to proceed.")



# import streamlit as st
# import pandas as pd
# import os
# import datetime
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute
# from pymongo import MongoClient

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

# # File upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# # Button to load a new file
# if not st.session_state.first_run:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.rerun()

# if uploaded_file is not None:
#     # Load the dataset from the uploaded file
#     historical_demand_monthly = prepare_dataset(filename=uploaded_file, item='All', rows='All')
    
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
#                     'periods': periods,
#                     'historical_periods': historical_periods,
#                     'test_periods': test_periods,
#                     'models': model_list
#                 }
#             }
#             logs_collection.insert_one(log_data)  # Save log data to MongoDB

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

#             # Save forecast and evaluation metrics to MongoDB
#             forecast_data = forecast.to_dict("records")
#             results_collection.insert_many(forecast_data)

#             evaluation_metrics_data = evaluation_metrics.to_dict("records")
#             evaluation_metrics_collection.insert_many(evaluation_metrics_data)

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


# import streamlit as st
# import pandas as pd
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# import os

# # Initialize the session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # Initialize the historical_demand_monthly variable
# historical_demand_monthly = None

# # 0.0 SHOW MAIN PAGE

# st.write('Demand Forecasting App')

# # Show a button "Load a new file" to load a new file
# if st.session_state.first_run == False:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.experimental_rerun()

# if len(st.session_state.hdm) > 0:
#     # Add a selectbox on the left of the screen to select if showing statistics for the full dataset
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

#     # Add a selectbox on the left to select if showing forecast details
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

#     # Add a selectbox on the left to select if exporting logs
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs ', value=False)

#     # Add a selectbox on the left to select if exporting logs and datasets as text files
#     st.session_state.export_results = st.sidebar.checkbox('Export results ', value=False)

#     # Add a selectbox on the left to select if models should look for a seasonal component
#     seasonal_check = st.sidebar.checkbox('Look for seasonality ', value=True)

#     # Add a selectbox on the left to select if models should look for a trend component
#     trend_check = st.sidebar.checkbox('Look for trend ', value=True)

#     # Ask if keeping only items with positive demand in the last 12 months
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

#     # Add a selectbox to select if showing items ordered by total demand or item number
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


# # 0.1 LOAD THE DATASET

# if st.session_state.first_run:
#     # Add file uploader for the user to upload a CSV file
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file is not None:
#         # Load the dataset from the uploaded file
#         historical_demand_monthly = prepare_dataset(filename=uploaded_file, item='All', rows='All')
        
#         if historical_demand_monthly is not None:
#             st.session_state.hdm = historical_demand_monthly
#             st.session_state.first_run = False
#             # Rerun the program
#             st.experimental_rerun()

# if st.session_state.first_run == False:    
#     historical_demand_monthly = st.session_state.hdm


# # 0.2 DISPLAY MAIN PAGE

# if historical_demand_monthly is not None:  # Check if proper dataset was loaded

#     # 1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

#     # Get the number of total items
#     total_items = historical_demand_monthly['item'].nunique()

#     # Get the last date
#     last_date = historical_demand_monthly['year_month'].max()

#     # Get the first date
#     first_date = historical_demand_monthly['year_month'].min()

#     # Check if st.session_state.statistics is empty
#     if len(st.session_state.statistics) == 0:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)

#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. ASK THE USER INPUTS TO RUN THE FORECAST
    
#     if st.session_state.keep_only_positive_demand:

#         # From statistics, get the list of items where 'last 12 months with positive demand' is True
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     # Add a searchable selectbox to select the item to be shown
#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

#     # Add the option to skip specific items
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

#     # Add the option to start the forecast from a specific item
#     start_from_item = st.sidebar.text_input('Start from item', value='')

#     # Ask the user to specify the last year_month of the historical period
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected !='All':
#             # Show the monthly demand for the selected item
#             st.write('Monthly demand for item', item_selected)
#             st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

#             # Count the the year_months in the period for the selected item
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
#         else:
#             # Count the the year_months in the period for all items
#             year_months = historical_demand_monthly['year_month'].nunique()

#         # Ask user to define the number of periods to forecast
#         periods=st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
#         if year_months > 1:
#             historical_periods=st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
#         else:
#             historical_periods=1
        
#         test_periods=st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months*0.2),12))
#         # Allow to restrict the number of items to be forecasted to 1000
#         if item_selected == 'All':
#             number_items=st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items+1, key='number_items', value=min(total_items, 10))
        
#         # Ask the user to choose if using ARIMA, Prophet or both
#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         # Show a button to run the forecast
#         if st.button('Run forecast'):

#             # 3. RUN THE FORECAST

#             # 3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             # Check if the log folder exists otherwise create it
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write('Number of periods to forecast: ' + str(periods) + '\n')
#                 f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
#                 f.write('Test periods: ' + str(test_periods) + '\n')
#                 f.write('Models selected: ' + str(model_list) + '\n')
            
#             # 3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
#             # Create a progress bar object
#             latest_iteration = st.empty() # Create a placeholder
#             progress_bar = st.progress(0)
        

#             # Create the real items list based on the user input
#             if item_selected == 'All':
#                 items = items_ordered[:number_items] # Apply a cutoff based on the sorting criteria and the number of items selected by the user
#                 # Remove the items to be skipped
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             # Initialize the forecast and evaluation_metrics as pandas dataframes
#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()


#             for item in items:
#                 # 3.3. PREPARE DATA FOR THE SPECIFIC ITEM

#                 # Select data for the selected item
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]

#                 # Restrict to the time range selected by the user
#                 df = df.tail(historical_periods)

#                 # Drop the column item
#                 df = df.drop(['item'], axis=1)


#                 # 3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

#                 # Show progress
#                 item_position = items.index(item)
#                 progress_bar.progress((item_position+1)/len(items))
#                 latest_iteration.text('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)))

#                 # Log progress
#                 start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

#                 if st.session_state.export_logs:
#                     with open(log_file, 'a') as f:
#                         f.write('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)) +': ' + item + ' started at ' + start_time +  '\n')

#                 # Run the forecast and get the evaluation metrics
#                 from forecasting_compute import forecast as forecast_compute
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 # Add a column with the item number to the forecast and evaluation_metrics
#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 # Append forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
#                 forecast = forecast.append(forecast_item)
#                 evaluation_metrics = evaluation_metrics.append(evaluation_metrics_item)

#                 if st.session_state.export_logs:
#                     # Get the timestamp for the end of the forecast
#                     end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

#                     # Write the evaluation metrics to the log file
#                     with open(log_file, 'a') as f:
#                         f.write('Forecast completed at ' + end_time +  '\n')
#                         f.write('Evaluation metrics: ' + str(evaluation_metrics_item) +  '\n')  

#             # 4. SHOW THE RESULTS

#             st.balloons()

#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 # Display data for the selected item only
#                 st.write ('Forecast for item', item)
                
#                 st.line_chart(forecast.drop(['item'], axis=1))
#                 if st.session_state.show_forecast_details:                
#                     st.write(forecast.drop(['item'], axis=1))
#                     st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))

#             else:
#                 if st.session_state.show_forecast_details:
#                     st.write('Evaluation metrics (showing first 10000 rows')
#                     st.write(evaluation_metrics.head(10000))

#                     st.write('Forecast (showing first 10000 rows')
#                     st.write(forecast.head(10000))
                
            
#                 st.write('Evaluation metrics summary')
#                 # Reset evaluation_metrics index
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 # Select only the records with index = 'maape'
#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 # Keep only the columns item and best model
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)
#                 # Show a bar chart showing for each item the best model
#                 # Group by item and best model and count the number of records
#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 # Plot a bar chart with Best model on the x axis and count on the y axis
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                

#                 # 5. EXPORT THE RESULTS TO CSV FILES

#                 # Export to a text file the forecast and the evaluation metrics
#                 # Check if forecast_results folder exists, if not create it
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
#                     forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
#                     evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'
    

#                     # Export the forecast and the evaluation metrics to csv files (with time stamp)
#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     # Overwrite forecast and evaluation files
#                     # Export the forecast and the evaluation metrics to csv files (without time stamp)
#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)


# ## IDEAS FOR IMPROVEMENTS
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period


# import streamlit as st
# import pandas as pd
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# import os

# # Initialize the session state variables
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True

# if 'show_debug' not in st.session_state:
#     st.session_state.show_debug = True

# if 'hdm' not in st.session_state:
#     st.session_state.hdm = pd.DataFrame()

# if 'statistics' not in st.session_state:
#     st.session_state.statistics = pd.DataFrame()

# # Initialize the historical_demand_monthly variable
# historical_demand_monthly = None

# # 0.0 SHOW MAIN PAGE

# st.write('Demand Forecasting App')

# # Show a button "Load a new file" to load a new file
# if st.session_state.first_run == False:
#     if st.sidebar.button('Load a new file'):
#         st.session_state.first_run = True
#         st.experimental_rerun()

# if len(st.session_state.hdm) > 0:
#     # Add a selectbox on the left of the screen to select if showing statistics for the full dataset
#     show_statistics = st.sidebar.checkbox('Show dataset statistics', value=False)

#     # Add a selectbox on the left to select if showing forecast details
#     st.session_state.show_forecast_details = st.sidebar.checkbox('Show forecast details', value=False)

#     # Add a selectbox on the left to select if exporting logs
#     st.session_state.export_logs = st.sidebar.checkbox('Export logs ', value=False)

#     # Add a selectbox on the left to select if exporting logs and datasets as text files
#     st.session_state.export_results = st.sidebar.checkbox('Export results ', value=False)

#     # Add a selectbox on the left to select if models should look for a seasonal component
#     seasonal_check = st.sidebar.checkbox('Look for seasonality ', value=True)

#     # Add a selectbox on the left to select if models should look for a trend component
#     trend_check = st.sidebar.checkbox('Look for trend ', value=True)

#     # Ask if keeping only items with positive demand in the last 12 months
#     st.session_state.keep_only_positive_demand = st.sidebar.checkbox('Keep only items with positive demand in the last 12 months', value=True)

#     # Add a selectbox to select if showing items ordered by total demand or item number
#     order_by = st.sidebar.selectbox('Order by', ['Total demand', 'Item number'])


# # 0.1 LOAD THE DATASET

# if st.session_state.first_run:
#     # Add file uploader for the user to upload a CSV file
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file is not None:
#         # Load the dataset from the uploaded file
#         historical_demand_monthly = prepare_dataset(filename=uploaded_file, item='All', rows='All')
        
#         if historical_demand_monthly is not None:
#             st.session_state.hdm = historical_demand_monthly
#             st.session_state.first_run = False
#             # Rerun the program
#             st.experimental_rerun()

# if st.session_state.first_run == False:    
#     historical_demand_monthly = st.session_state.hdm


# # 0.2 DISPLAY MAIN PAGE

# if historical_demand_monthly is not None:  # Check if proper dataset was loaded

#     # 1. GET AND DISPLAY SOME STATISTICS ON THE FULL DATASET

#     # Get the number of total items
#     total_items = historical_demand_monthly['item'].nunique()

#     # Get the last date
#     last_date = historical_demand_monthly['year_month'].max()

#     # Get the first date
#     first_date = historical_demand_monthly['year_month'].min()

#     # Check if st.session_state.statistics is empty
#     if len(st.session_state.statistics) == 0:
#         st.session_state.statistics = stat(historical_demand_monthly, display=False)

#     statistics = st.session_state.statistics

#     if show_statistics:
#         st.write('Dataset statistics')
#         stat(historical_demand_monthly, statistics, display=True)

#     # 2. ASK THE USER INPUTS TO RUN THE FORECAST
    
#     if st.session_state.keep_only_positive_demand:

#         # From statistics, get the list of items where 'last 12 months with positive demand' is True
#         items_with_positive_demand = statistics[statistics['last 12 months with positive demand'] == True]
#         historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['item'].isin(items_with_positive_demand['item'])]
            
#     if order_by == 'Total demand':
#         items_ordered = historical_demand_monthly.groupby('item')['demand'].sum().sort_values(ascending=False).index.tolist()
#     else:
#         items_ordered = historical_demand_monthly['item'].unique().tolist()

#     # Add a searchable selectbox to select the item to be shown
#     item_selected = st.sidebar.selectbox('Select item', ['All'] + items_ordered)

#     # Add the option to skip specific items
#     skip_items = st.sidebar.text_input('Skip items (separate by comma)', value='')

#     # Add the option to start the forecast from a specific item
#     start_from_item = st.sidebar.text_input('Start from item', value='')

#     # Ask the user to specify the last year_month of the historical period
#     last_month_cutoff = st.sidebar.text_input('Historical dataset last month (YYYYMM)', value=last_date)
    
#     historical_demand_monthly = historical_demand_monthly[historical_demand_monthly['year_month'] <= last_month_cutoff]
#     total_items = historical_demand_monthly['item'].nunique()

#     if item_selected != '':
#         if item_selected !='All':
#             # Show the monthly demand for the selected item
#             st.write('Monthly demand for item', item_selected)
#             st.write(historical_demand_monthly[historical_demand_monthly['item'] == item_selected])

#             st.line_chart(historical_demand_monthly[historical_demand_monthly['item'] == item_selected].set_index('year_month')['demand'])

#             # Count the the year_months in the period for the selected item
#             year_months = historical_demand_monthly[historical_demand_monthly['item'] == item_selected]['year_month'].nunique()
        
#         else:
#             # Count the the year_months in the period for all items
#             year_months = historical_demand_monthly['year_month'].nunique()

#         # Ask user to define the number of periods to forecast
#         periods=st.slider('Number of periods to forecast', min_value=1, max_value=36, key='periods', value=12)
#         if year_months > 1:
#             historical_periods=st.slider('Historical periods to analyze', min_value=1, max_value=year_months, key='historical_periods', value=min(year_months, 48))
#         else:
#             historical_periods=1
        
#         test_periods=st.slider('Test periods', min_value=0, max_value=year_months, key='test_periods', value=min(round(year_months*0.2),12))
#         # Allow to restrict the number of items to be forecasted to 1000
#         if item_selected == 'All':
#             number_items=st.slider('Number items to be processed - will apply a cutoff based on the sorting criteria', min_value=1, max_value=total_items+1, key='number_items', value=min(total_items, 10))
        
#         # Ask the user to choose if using ARIMA, Prophet or both
#         model_list = st.multiselect('Select model', ['ARIMA', 'ETS', 'STL', 'Prophet', 'Neural Prophet'], default=['Prophet'])

#         # Show a button to run the forecast
#         if st.button('Run forecast'):

#             # 3. RUN THE FORECAST

#             # 3.1. CREATE A LOG FILE AND WRITE THE PARAMETERS SELECTED BY THE USER
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             # Check if the log folder exists otherwise create it
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write('Number of periods to forecast: ' + str(periods) + '\n')
#                 f.write('Historical periods to analyze: ' + str(historical_periods) + '\n')
#                 f.write('Test periods: ' + str(test_periods) + '\n')
#                 f.write('Models selected: ' + str(model_list) + '\n')
            
#             # 3.2 DO SOME PREPARATION BEFORE RUNNING THE FORECAST
            
#             # Create a progress bar object
#             latest_iteration = st.empty() # Create a placeholder
#             progress_bar = st.progress(0)
        

#             # Create the real items list based on the user input
#             if item_selected == 'All':
#                 items = items_ordered[:number_items] # Apply a cutoff based on the sorting criteria and the number of items selected by the user
#                 # Remove the items to be skipped
#                 if skip_items != '':
#                     items = [item for item in items if item not in skip_items.split(',')]
#                 if start_from_item != '':
#                     items = items[items.index(start_from_item):]
#             else:
#                 items = [item_selected]

#             # Initialize the forecast and evaluation_metrics as pandas dataframes
#             forecast = pd.DataFrame()
#             evaluation_metrics = pd.DataFrame()


#             for item in items:
#                 # 3.3. PREPARE DATA FOR THE SPECIFIC ITEM

#                 # Select data for the selected item
#                 df = historical_demand_monthly[historical_demand_monthly['item'] == item]

#                 # Restrict to the time range selected by the user
#                 df = df.tail(historical_periods)

#                 # Drop the column item
#                 df = df.drop(['item'], axis=1)


#                 # 3.4. RUN THE FORECAST FOR THE SPECIFIC ITEM, TRACE LOG AND SHOW PROGRESS

#                 # Show progress
#                 item_position = items.index(item)
#                 progress_bar.progress((item_position+1)/len(items))
#                 latest_iteration.text('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)))

#                 # Log progress
#                 start_time = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

#                 if st.session_state.export_logs:
#                     with open(log_file, 'a') as f:
#                         f.write('Forecasting item ' + str(item_position+1) + ' of ' + str(len(items)) +': ' + item + ' started at ' + start_time +  '\n')

#                 # Run the forecast and get the evaluation metrics
#                 from forecasting_compute import forecast as forecast_compute
#                 forecast_item, evaluation_metrics_item = forecast_compute(df, test_periods, periods, model_list, seasonal=seasonal_check, trend=trend_check)

#                 # Add a column with the item number to the forecast and evaluation_metrics
#                 forecast_item['item'] = item
#                 evaluation_metrics_item['item'] = item

#                 # Append forecast_item and evaluation_metrics_item to forecast and evaluation_metrics
#                 forecast = forecast.append(forecast_item)
#                 evaluation_metrics = evaluation_metrics.append(evaluation_metrics_item)

#                 if st.session_state.export_logs:
#                     # Get the timestamp for the end of the forecast
#                     end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

#                     # Write the evaluation metrics to the log file
#                     with open(log_file, 'a') as f:
#                         f.write('Forecast completed at ' + end_time +  '\n')
#                         f.write('Evaluation metrics: ' + str(evaluation_metrics_item) +  '\n')  

#             # 4. SHOW THE RESULTS

#             st.balloons()

#             st.write('Results')
#             st.write('Number of items processed: ', len(items))

#             if item_selected != 'All':
#                 # Display data for the selected item only
#                 st.write ('Forecast for item', item)
                
#                 st.line_chart(forecast.drop(['item'], axis=1))
#                 if st.session_state.show_forecast_details:                
#                     st.write(forecast.drop(['item'], axis=1))
#                     st.write('Evaluation metrics', evaluation_metrics.drop(['item'], axis=1))

#             else:
#                 if st.session_state.show_forecast_details:
#                     st.write('Evaluation metrics (showing first 10000 rows')
#                     st.write(evaluation_metrics.head(10000))

#                     st.write('Forecast (showing first 10000 rows')
#                     st.write(forecast.head(10000))
                
            
#                 st.write('Evaluation metrics summary')
#                 # Reset evaluation_metrics index
#                 evaluation_metrics_summary = evaluation_metrics.reset_index()

#                 if st.session_state.show_forecast_details:
#                     st.write(evaluation_metrics_summary.head(10000))

#                 # Select only the records with index = 'maape'
#                 evaluation_metrics_summary = evaluation_metrics_summary[evaluation_metrics_summary['index'] == 'maape']
#                 # Keep only the columns item and best model
#                 evaluation_metrics_summary = evaluation_metrics_summary[['item', 'Best model', 'Best model value']]
#                 st.write(evaluation_metrics_summary)
#                 # Show a bar chart showing for each item the best model
#                 # Group by item and best model and count the number of records
#                 evaluation_metrics_summary = evaluation_metrics_summary.groupby(['Best model']).size().reset_index(name='count')
#                 st.write(evaluation_metrics_summary)
#                 # Plot a bar chart with Best model on the x axis and count on the y axis
#                 st.bar_chart(evaluation_metrics_summary.set_index('Best model'))
                

#                 # 5. EXPORT THE RESULTS TO CSV FILES

#                 # Export to a text file the forecast and the evaluation metrics
#                 # Check if forecast_results folder exists, if not create it
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     timestamp = pd.to_datetime('today').strftime('%Y%m%d%H%M%S')
#                     forecast_csv = 'forecast_results/forecast_' + timestamp + '.csv'
#                     evaluation_metrics_csv = 'forecast_results/evaluation_metrics_' + timestamp + '.csv'
    

#                     # Export the forecast and the evaluation metrics to csv files (with time stamp)
#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     # Overwrite forecast and evaluation files
#                     # Export the forecast and the evaluation metrics to csv files (without time stamp)
#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)


# ## IDEAS FOR IMPROVEMENTS
# # Additional statistics for the dataset
# # Show results with total forecast for each item with the best model for the period



# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#     historical_demand_monthly = prepare_dataset(filename=None, item='All', rows='All')
    
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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

# ## Ideas for Improvements
# # Additional statistics for the dataset
# # Show results with total forecast for each item with


# import streamlit as st
# import pandas as pd
# import os
# from forecasting_dataset_load import prepare_dataset
# from forecasting_dataset_statistics import statistics as stat
# from forecasting_compute import forecast as forecast_compute

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
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             if not os.path.exists('log'):
#                 os.makedirs('log')
#             log_file = 'log/log_' + timestamp + '.txt'
            
#             with open(log_file, 'w') as f:
#                 f.write('Log file created at ' + timestamp + '\n')
#                 f.write('Parameters selected by the user: \n')
#                 f.write(f'Number of periods to forecast: {periods}\n')
#                 f.write(f'Historical periods to analyze: {historical_periods}\n')
#                 f.write(f'Test periods: {test_periods}\n')
#                 f.write(f'Models selected: {model_list}\n')

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

#                 # 5. Export the Results to CSV Files
#                 if st.session_state.export_results:                
#                     if not os.path.exists('forecast_results'):
#                         os.makedirs('forecast_results')

#                     forecast_csv = f'forecast_results/forecast_{timestamp}.csv'
#                     evaluation_metrics_csv = f'forecast_results/evaluation_metrics_{timestamp}.csv'

#                     forecast.to_csv(forecast_csv, index=True)
#                     evaluation_metrics_summary.to_csv(evaluation_metrics_csv, index=False)

#                     forecast.to_csv('forecast_results/forecast.csv', index=True)
#                     evaluation_metrics.to_csv('forecast_results/evaluation_metrics.csv', index=True)

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
# from pymongo import MongoClient
# import config
# # import datetime

# # MongoDB Atlas setup
# client = MongoClient(config.MONGODB_URI)
# db = client[config.DB_NAME]

# # Initialize MongoDB client
# # client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
# # db = client['forecasting_db']  # Database name
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

#             # Ask for email input to send the results
#             user_email = st.text_input("Enter your email to receive the results", value="")

#             if user_email:
#                 if st.button("Send Email"):
#                     try:
#                         # Send email with results
#                         send_email(user_email, timestamp, forecast, evaluation_metrics)
#                         st.success(f"Results sent to {user_email}")
#                     except Exception as e:
#                         st.error(f"Failed to send email: {str(e)}")

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


# def send_email(to_email, timestamp, forecast, evaluation_metrics):
#     """Function to send email with forecast and evaluation metrics as content."""
#     from_email = "your-email@example.com"
#     password = "your-email-password"
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
#     server = smtplib.SMTP('smtp.example.com', 587)  # Use your SMTP server and port
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
#     from_email = "your-email@example.com"
#     password = "your-email-password"
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
#     server = smtplib.SMTP('smtp.example.com', 587)  # Use your SMTP server and port
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

