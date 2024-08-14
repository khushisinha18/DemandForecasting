# def forecast(df, test_periods, periods, model_list, seasonal=True, trend=True):

#     import streamlit as st
#     import pandas as pd
#     from forecasting_metrics import evaluate
    
#     #1. PREPARE DATA
    
#     #convert the colum year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month']+'01', format='%Y%m%d')

#     #set year_month as index
#     df = df.set_index('year_month')

#     #split the main dataset in train and test datasets
#     train_periods= (len(df)-test_periods)
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     #create a dataframe with the training and test datasets
#     forecast = pd.concat([train, test], axis=1)
#     forecast.columns = ['train', 'test']

#     #st.write('This is the full forecast dataframe -debug', full_forecast)


#     if 'ARIMA' in model_list:
#         #2. RUN FORECASTING USING ARIMA

#         from pmdarima.arima import auto_arima
        
#         #build the model
#         #essential parameters: y, m

#         arima_model = auto_arima   (y=train, #training dataset
#                                     start_p=2, #number of autoregressive terms (default p=2)
#                                     max_p=5, #(default p=5)
#                                     d=None, #order of differencing (default d=None)
#                                     start_q=2, #order of moving average (default q=2)
#                                     max_q=5, #(default q=5)
#                                     start_P=1, #number of seasonal autoregressive terms (default P=1)
#                                     seasonal=seasonal, #default: True
#                                     m=12, #periodicity of the time series (default =1) for monthly data m=12
#                                     D=None, #order of seasonal differencing (default D=None)
#                                     n_fits=10, #number of fits to try (default 10)
#                                     trace=False, #default: False
#                                     error_action='ignore',
#                                     supress_warnings=True, #default: True
#                                     stepwise=True, #default: True
#                                     information_criterion='aic') #default: aic, other options: bic, aicc, oob)
        
#         #ARIMA parameters explained: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
        
#         #predict values for the test dataset and future periods        
#         forecast_arima = pd.DataFrame(arima_model.predict(test_periods+periods))

#         #add to the forecast dataframe
#         forecast_arima.columns = ['ARIMA']
#         forecast = pd.concat([forecast, forecast_arima], axis=1)
        

#     if 'STL' in model_list:
#         ##3. RUN FORECASTING USING STL DECOMPOSITION
#         from statsmodels.tsa.seasonal import STL

#         #build the model
#         stl_model = STL(train, seasonal=seasonal, period=12).fit()

#         #predict values for the test dataset and future periods
#         #inizialize forecast_stl dataframe
#         forecast_stl = pd.DataFrame()
#         forecast_stl['STL'] = stl_model.predict(test_periods+periods)

#         #replace all negative values with 0
#         forecast_stl[forecast_stl < 0] = 0

#         #add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_stl], axis=1)



#     if 'ETS' in model_list:
#         ##4. RUN FORECASTING USING EXPONENTIAL SMOOTHING
#         from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
#         #from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ets #this is the new version of ets
        
#         #build the model
#         seasonal_ets = None
#         trend_ets = None
#         if seasonal == True:
#             seasonal_ets='add'    
#         if trend == True:
#             trend_ets='add'
        
#         ets_model = ets(train, trend=trend_ets, seasonal=seasonal_ets).fit() 

#         #predict values for the test dataset and future periods
#         #inizialize forecast_ets dataframe
#         forecast_ets = pd.DataFrame()
#         forecast_ets['ETS'] = ets_model.predict(start=train_periods, end=len(df)+periods-1)

#         #replace all negative values with 0
#         forecast_ets[forecast_ets < 0] = 0

#         #add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_ets], axis=1)
    
    
#     if 'Prophet' in model_list:
#     ##5.2 RUN FORECASTING USING PROPHET - TRAINING DATASET

#         from prophet import Prophet

#         pro_model = Prophet()

#         #create a df_pro dataframe with the columns ds and y renamed into year_month and demand
#         #restore the index
#         df_pro = df.reset_index(inplace=True)
#         df_pro = df.rename(columns={'year_month': 'ds', 'demand': 'y'})

#         #build the model
#         pro_model.fit(df_pro)

#         #create a dataframe with the testing and future periods
#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         #predict values
#         forecast_pro = pro_model.predict(test_and_future)

#         #create forecast_pro_renamed with only the columns ds and yhat renamed into year_month and demand
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')

#         #replace all negative values with 0
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0

#         #add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)


    
#     if 'Neural Prophet' in model_list:
#     ##5.3 RUN FORECASTING USING NEURAL PROPHET - TRAINING DATASET
#     #set epochs to 50

#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         #create a df_pro dataframe with the columns ds and y renamed into year_month and demand

#         #restore the index
#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index(inplace=True)
        
#         df_npro = df.rename(columns={'year_month': 'ds', 'demand': 'y'})

#         #build the model
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         #create a dataframe with the testing and future periods
#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         #predict values
#         forecast_npro = npro_model.predict(test_and_future)

#         #create forecast_pro_renamed with only the columns ds and yhat renamed into year_month and demand

#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')

#         #replace all negative values with 0
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0

#         #add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)


#     #if any model is selected, show the forecast and the evaluation metrics
#     if len(model_list)>0:
#         #st.write('Forecasting results')
#         #format year_month as year-month
#         forecast.index = forecast.index.strftime('%Y-%m')

#         #create a dataframe restricted to the test periods
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         #st.write('Dataset used for evaluation', forecast_eval)

#         #calculate the evaluation metrics for each model
#         evaluate_metrics = {}
#         for model in model_list:
#             evaluate_metrics[model]=evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape','rmse', 'mse', 'mae'))
            
#         #convert the dictionary into a dataframe
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)

#         #add a column showing the best model
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)

#         #add a column showing the metric value of the best model
#         evaluate_metrics['Best model value'] = evaluate_metrics.min(axis=1)


#         #MAAPE https://www.sciencedirect.com/science/article/pii/S0169207016000121

#         #Choosing the right forecasting metric is not straightforward.
#         #Let’s review the pro and con of RMSE, MAE, MAPE, and Bias. Spoiler: MAPE is the worst. Don’t use it.
#         #https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d

#     return forecast, evaluate_metrics








# import streamlit as st
# import pandas as pd
# from forecasting_metrics import evaluate

# def forecast(df, test_periods, periods, model_list, seasonal=True, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.concat([train, test], axis=1)
#     forecast.columns = ['train', 'test']

#     if 'ARIMA' in model_list:
#         # 2. RUN FORECASTING USING ARIMA
#         from pmdarima.arima import auto_arima
        
#         # Build the model
#         arima_model = auto_arima(
#             y=train, 
#             start_p=2, 
#             max_p=5, 
#             d=None, 
#             start_q=2, 
#             max_q=5, 
#             start_P=1, 
#             seasonal=seasonal, 
#             m=12, 
#             D=None, 
#             n_fits=10, 
#             trace=False, 
#             error_action='ignore',
#             suppress_warnings=True, 
#             stepwise=True, 
#             information_criterion='aic'
#         )
        
#         # Predict values for the test dataset and future periods        
#         forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods))

#         # Add to the forecast dataframe
#         forecast_arima.columns = ['ARIMA']
#         forecast = pd.concat([forecast, forecast_arima], axis=1)
        

#     if 'STL' in model_list:
#         # 3. RUN FORECASTING USING STL DECOMPOSITION
#         from statsmodels.tsa.seasonal import STL

#         # Build the model
#         stl_model = STL(train, seasonal=seasonal, period=12).fit()

#         # Predict values for the test dataset and future periods
#         forecast_stl = pd.DataFrame()
#         forecast_stl['STL'] = stl_model.predict(test_periods + periods)

#         # Replace all negative values with 0
#         forecast_stl[forecast_stl < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_stl], axis=1)


#     if 'ETS' in model_list:
#         # 4. RUN FORECASTING USING EXPONENTIAL SMOOTHING
#         from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
        
#         # Build the model
#         seasonal_ets = 'add' if seasonal else None
#         trend_ets = 'add' if trend else None
        
#         ets_model = ets(train, trend=trend_ets, seasonal=seasonal_ets).fit() 

#         # Predict values for the test dataset and future periods
#         forecast_ets = pd.DataFrame()
#         forecast_ets['ETS'] = ets_model.predict(start=train_periods, end=len(df) + periods - 1)

#         # Replace all negative values with 0
#         forecast_ets[forecast_ets < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_ets], axis=1)
    
    
#     if 'Prophet' in model_list:
#         # 5.2 RUN FORECASTING USING PROPHET
#         from prophet import Prophet

#         pro_model = Prophet()

#         # Create a df_pro dataframe with columns ds and y renamed from year_month and demand
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})

#         # Build the model
#         pro_model.fit(df_pro)

#         # Create a dataframe with the testing and future periods
#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_pro = pro_model.predict(test_and_future)

#         # Create forecast_pro_renamed with only columns ds and yhat renamed to year_month and Prophet
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)


#     if 'Neural Prophet' in model_list:
#         # 5.3 RUN FORECASTING USING NEURAL PROPHET
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         # Create a df_npro dataframe with columns ds and y renamed from year_month and demand
#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         df_npro = df.rename(columns={'year_month': 'ds', 'demand': 'y'})

#         # Build the model
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         # Create a dataframe with the testing and future periods
#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_npro = npro_model.predict(test_and_future)

#         # Create forecast_npro_renamed with only columns ds and yhat1 renamed to year_month and Neural Prophet
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)


#     # If any model is selected, show the forecast and evaluation metrics
#     if len(model_list) > 0:
#         # Format year_month as year-month
#         forecast.index = forecast.index.strftime('%Y-%m')

#         # Create a dataframe restricted to the test periods
#         forecast_eval = forecast.iloc[train_periods:len(df)]

#         # Calculate the evaluation metrics for each model
#         evaluate_metrics = {model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae')) for model in model_list}

#         # Convert the dictionary into a dataframe
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)

#         # Convert all relevant columns to numeric, coercing errors to NaN
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')

#         # Drop any rows where all elements are NaN
#         evaluate_metrics = evaluate_metrics.dropna(how='all')

#         # Add a column showing the best model
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)

#         # Add a column showing the metric value of the best model
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)

#     return forecast, evaluate_metrics



# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate

# def forecast(df, test_periods, periods, model_list, seasonal=True, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.concat([train, test], axis=1)
#     forecast.columns = ['train', 'test']

#     if 'ARIMA' in model_list:
#         # 2. RUN FORECASTING USING ARIMA
#         from pmdarima.arima import auto_arima
        
#         # Ensure there's enough data and variability before running ARIMA
#         if len(train) > 12 and train['train'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['train'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal, 
#                     m=12, 
#                     D=0,  # Optionally set D=0 if seasonal differencing is problematic
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 # Predict values for the test dataset and future periods
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods))

#                 # Add to the forecast dataframe
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan  # Handle the error gracefully
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         # 3. RUN FORECASTING USING STL DECOMPOSITION
#         from statsmodels.tsa.seasonal import STL

#         # Build the model
#         stl_model = STL(train['train'], seasonal=seasonal, period=12).fit()

#         # Predict values for the test dataset and future periods
#         forecast_stl = pd.DataFrame()
#         forecast_stl['STL'] = stl_model.predict(test_periods + periods)

#         # Replace all negative values with 0
#         forecast_stl[forecast_stl < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_stl], axis=1)

#     if 'ETS' in model_list:
#         # 4. RUN FORECASTING USING EXPONENTIAL SMOOTHING
#         from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
        
#         # Build the model
#         seasonal_ets = 'add' if seasonal else None
#         trend_ets = 'add' if trend else None
        
#         ets_model = ets(train['train'], trend=trend_ets, seasonal=seasonal_ets).fit() 

#         # Predict values for the test dataset and future periods
#         forecast_ets = pd.DataFrame()
#         forecast_ets['ETS'] = ets_model.predict(start=train_periods, end=len(df) + periods - 1)

#         # Replace all negative values with 0
#         forecast_ets[forecast_ets < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_ets], axis=1)
    
    
#     if 'Prophet' in model_list:
#         # 5.2 RUN FORECASTING USING PROPHET
#         from prophet import Prophet

#         pro_model = Prophet()

#         # Create a df_pro dataframe with columns ds and y renamed from year_month and demand
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'train': 'y'})

#         # Build the model
#         pro_model.fit(df_pro)

#         # Create a dataframe with the testing and future periods
#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_pro = pro_model.predict(test_and_future)

#         # Create forecast_pro_renamed with only columns ds and yhat renamed to year_month and Prophet
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         # 5.3 RUN FORECASTING USING NEURAL PROPHET
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         # Create a df_npro dataframe with columns ds and y renamed from year_month and demand
#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         df_npro = df.rename(columns={'year_month': 'ds', 'train': 'y'})

#         # Build the model
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         # Create a dataframe with the testing and future periods
#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_npro = npro_model.predict(test_and_future)

#         # Create forecast_npro_renamed with only columns ds and yhat1 renamed to year_month and Neural Prophet
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # If any model is selected, show the forecast and evaluation metrics
#     if len(model_list) > 0:
#         # Format year_month as year-month
#         forecast.index = forecast.index.strftime('%Y-%m')

#         # Create a dataframe restricted to the test periods
#         forecast_eval = forecast.iloc[train_periods:len(df)]

#         # Calculate the evaluation metrics for each model
#         evaluate_metrics = {model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae')) for model in model_list}

#         # Convert the dictionary into a dataframe
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)

#         # Convert all relevant columns to numeric, coercing errors to NaN
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')

#         # Drop any rows where all elements are NaN
#         evaluate_metrics = evaluate_metrics.dropna(how='all')

#         # Add a column showing the best model
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)

#         # Add a column showing the metric value of the best model
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)

#     return forecast, evaluate_metrics

# import streamlit as st
# import pandas as pd
# import numpy as np  # Import numpy to use np.nan
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']  # Assuming 'demand' is the target column
#     forecast['test'] = test['demand']

#     if 'ARIMA' in model_list:
#         # 2. RUN FORECASTING USING ARIMA
#         from pmdarima.arima import auto_arima
        
#         # Ensure there's enough data and variability before running ARIMA
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,  # Ensure seasonal is valid
#                     m=12, 
#                     D=0,  # Optionally set D=0 if seasonal differencing is problematic
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 # Predict values for the test dataset and future periods
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)

#                 # Add to the forecast dataframe
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan  # Handle the error gracefully
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         # 3. RUN FORECASTING USING STL DECOMPOSITION
#         # Ensure the seasonal parameter is a valid odd integer >= 3
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7  # Default to 7 if invalid

#         try:
#             # Perform STL decomposition
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()

#             # Use the trend and seasonal components for forecasting
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             # Extend the forecast to the desired number of periods using the last known values
#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)  # Use iloc for position-based indexing

#             # Combine the forecast with the existing data
#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             # Add the STL forecast to the forecast dataframe
#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan  # Handle the error gracefully

#     if 'ETS' in model_list:
#         # 4. RUN FORECASTING USING EXPONENTIAL SMOOTHING
#         # Check if we have enough data for seasonality
#         if len(train) >= 24:  # Assuming monthly data with m=12
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None  # Disable seasonality if not enough data
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit() 

#             # Predict values for the test dataset and future periods
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']

#             # Replace all negative values with 0
#             forecast_ets[forecast_ets < 0] = 0

#             # Add to the forecast dataframe
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan  # Handle the error gracefully
    
#     if 'Prophet' in model_list:
#         # 5.2 RUN FORECASTING USING PROPHET
#         from prophet import Prophet

#         pro_model = Prophet()

#         # Create a df_pro dataframe with columns ds and y renamed from year_month and demand
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})

#         # Build the model
#         pro_model.fit(df_pro)

#         # Create a dataframe with the testing and future periods
#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_pro = pro_model.predict(test_and_future)

#         # Create forecast_pro_renamed with only columns ds and yhat renamed to year_month and Prophet
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0

#         # Add to the forecast dataframe
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         # 5.3 RUN FORECASTING USING NEURAL PROPHET
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         # Create a df_npro dataframe with columns ds and y renamed from year_month and demand
#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         else:
#             df_npro = df.copy()

#         df_npro = df_npro.rename(columns={'year_month': 'ds', 'demand': 'y'})

#         # Build the model
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         # Create a dataframe with the testing and future periods
#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
        
#         # Predict values
#         forecast_npro = npro_model.predict(test_and_future)

#         # Create forecast_npro_renamed with only columns ds and yhat1 renamed to year_month and Neural Prophet
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')

#         # Replace all negative values with 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # If any model is selected, show the forecast and evaluation metrics
#     if len(model_list) > 0:
#         # Format year_month as year-month
#         forecast.index = forecast.index.strftime('%Y-%m')

#         # Create a dataframe restricted to the test periods
#         forecast_eval = forecast.iloc[train_periods:len(df)]

#         # Calculate the evaluation metrics for each model
#         evaluate_metrics = {model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae')) for model in model_list}

#         # Convert the dictionary into a dataframe
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)

#         # Convert all relevant columns to numeric, coercing errors to NaN
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')

#         # Drop any rows where all elements are NaN
#         evaluate_metrics = evaluate_metrics.dropna(how='all')

#         # Add a column showing the best model
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)

#         # Add a column showing the metric value of the best model
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)

#     return forecast, evaluate_metrics


# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima
        
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         else:
#             df_npro = df.copy()

#         df_npro = df_npro.rename(columns={'year_month': 'ds', 'demand': 'y'})
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         forecast_npro = npro_model.predict(test_and_future)
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics


# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima
        
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         else:
#             df_npro = df.copy()

#         # Drop unnecessary columns
#         df_npro = df_npro[['year_month', 'demand']].rename(columns={'year_month': 'ds', 'demand': 'y'})
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         forecast_npro = npro_model.predict(test_and_future)
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics


# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima
        
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         if 'Prophet' not in model_list:
#             df_npro = df.reset_index()
#         else:
#             df_npro = df.copy()

#         df_npro = df_npro.rename(columns={'year_month': 'ds', 'demand': 'y'})
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         forecast_npro = npro_model.predict(test_and_future)
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics


# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima
        
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         df_npro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         forecast_npro = npro_model.predict(test_and_future)
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics



# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate_all
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima
        
#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'], 
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )
                
#                 forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=test.index)
#                 forecast_arima.columns = ['ARIMA']
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         df_npro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         df_npro = df_npro.drop(columns=['item'], errors='ignore')  # Drop the item column to avoid errors

#         try:
#             npro_model.fit(df_npro, freq='MS', epochs=50)

#             future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#             test_and_future = future.iloc[train_periods:]
#             forecast_npro = npro_model.predict(test_and_future)
#             forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#             forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#             forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#             forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)
#         except ValueError as e:
#             st.write(f"Neural Prophet failed: {e}")
#             forecast['Neural Prophet'] = np.nan

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate_all(forecast_eval['test'], forecast_eval[model])
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics


# import streamlit as st
# import pandas as pd
# import numpy as np
# from forecasting_metrics import evaluate
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

# def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
#     # 1. PREPARE DATA
    
#     # Convert the column year_month to datetime64
#     df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

#     # Set year_month as index
#     df = df.set_index('year_month')

#     # Ensure frequency is set to MS (month start) for ARIMA and STL
#     if df.index.freq is None:
#         df = df.asfreq('MS')

#     # Split the main dataset into train and test datasets
#     train_periods = len(df) - test_periods
#     train = df.iloc[:train_periods]
#     test = df.iloc[train_periods:]

#     # Create a dataframe with the training and test datasets
#     forecast = pd.DataFrame(index=df.index)
#     forecast['train'] = train['demand']
#     forecast['test'] = test['demand']

#     # Check if forecast columns for each model are non-empty
#     if forecast.empty:
#         st.write("Forecast DataFrame is empty.")
    
#     # Check the forecast and test data
#     st.write("Forecast DataFrame Preview:", forecast.head())
#     st.write("Test Data:", forecast['test'].head())

#     if 'ARIMA' in model_list:
#         from pmdarima.arima import auto_arima

#         if len(train) > 12 and train['demand'].std() > 0:
#             try:
#                 arima_model = auto_arima(
#                     y=train['demand'].diff().dropna(),  # Differencing the data to make it stationary
#                     start_p=2, 
#                     max_p=5, 
#                     d=None, 
#                     start_q=2, 
#                     max_q=5, 
#                     start_P=1, 
#                     seasonal=seasonal >= 3,
#                     m=12, 
#                     D=0,
#                     n_fits=10, 
#                     trace=False, 
#                     error_action='ignore',
#                     suppress_warnings=True, 
#                     stepwise=True, 
#                     information_criterion='aic'
#                 )

#                 arima_forecast = arima_model.predict(n_periods=test_periods + periods)
#                 arima_forecast = pd.Series(arima_forecast, index=test.index)
#                 arima_forecast = arima_forecast.cumsum() + train['demand'].iloc[-1]  # Invert differencing

#                 forecast_arima = pd.DataFrame(arima_forecast, columns=['ARIMA'])
#                 forecast = pd.concat([forecast, forecast_arima], axis=1)
#             except ValueError as e:
#                 st.write(f"ARIMA failed: {e}")
#                 forecast['ARIMA'] = np.nan
#         else:
#             st.write("Not enough data for ARIMA or data is constant.")
#             forecast['ARIMA'] = np.nan

#     if 'STL' in model_list:
#         if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
#             stl_seasonal = seasonal
#         else:
#             stl_seasonal = 7

#         try:
#             stl = STL(train['demand'], seasonal=stl_seasonal, period=12)
#             stl_result = stl.fit()
#             stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

#             future_index = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
#             extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

#             stl_forecast = pd.concat([pd.Series(stl_forecast, index=test.index), extended_forecast])

#             forecast['STL'] = stl_forecast

#         except ValueError as e:
#             st.write(f"STL failed: {e}")
#             forecast['STL'] = np.nan

#     if 'ETS' in model_list:
#         if len(train) >= 24:
#             seasonal_ets = 'add' if seasonal else None
#         else:
#             seasonal_ets = None
        
#         trend_ets = 'add' if trend else None
        
#         try:
#             ets_model = ets(train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
#             forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=test.index)
#             forecast_ets.columns = ['ETS']
#             forecast_ets[forecast_ets < 0] = 0
#             forecast = pd.concat([forecast, forecast_ets], axis=1)
#         except ValueError as e:
#             st.write(f"ETS failed: {e}")
#             forecast['ETS'] = np.nan
    
#     if 'Prophet' in model_list:
#         from prophet import Prophet

#         pro_model = Prophet()
#         df_pro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         pro_model.fit(df_pro)

#         future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
#         test_and_future = future.iloc[train_periods:]
#         forecast_pro = pro_model.predict(test_and_future)
#         forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
#         forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
#         forecast_pro_renamed[forecast_pro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

#     if 'Neural Prophet' in model_list:
#         from neuralprophet import NeuralProphet

#         npro_model = NeuralProphet()

#         df_npro = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
#         npro_model.fit(df_npro, freq='MS', epochs=50)

#         future = npro_model.make_future_dataframe(df_npro, periods=periods, n_historic_predictions=len(df))
#         test_and_future = future.iloc[train_periods:]
#         forecast_npro = npro_model.predict(test_and_future)
#         forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
#         forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
#         forecast_npro_renamed[forecast_npro_renamed < 0] = 0
#         forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

#     # Check if forecast contains valid predictions
#     for model in model_list:
#         if model not in forecast.columns or forecast[model].isna().all():
#             st.write(f"No valid predictions for model: {model}")
#         else:
#             st.write(f"Valid predictions found for model: {model}")

#     # Calculate the evaluation metrics
#     try:
#         forecast_eval = forecast.iloc[train_periods:len(df)]
#         evaluate_metrics = {
#             model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
#             for model in model_list if model in forecast.columns and not forecast[model].isna().all()
#         }
#         evaluate_metrics = pd.DataFrame(evaluate_metrics)
#     except Exception as e:
#         st.write(f"Error calculating evaluation metrics: {e}")

#     # Check if evaluate_metrics is empty
#     if evaluate_metrics.empty:
#         st.write("Evaluation metrics are empty.")
#     else:
#         evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
#         evaluate_metrics = evaluate_metrics.dropna(how='all')
#         evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
#         numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
#         evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
#         st.write("Evaluation Metrics Summary:", evaluate_metrics)

#     return forecast, evaluate_metrics


import streamlit as st
import pandas as pd
import numpy as np
from forecasting_metrics import evaluate
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets

def forecast(df, test_periods, periods, model_list, seasonal=7, trend=True):
    # 1. PREPARE DATA
    
    # Convert the column year_month to datetime64
    df['year_month'] = pd.to_datetime(df['year_month'] + '01', format='%Y%m%d')

    # Set year_month as index
    df = df.set_index('year_month')

    # Split the main dataset into train and test datasets
    train_periods = len(df) - test_periods
    train = df.iloc[:train_periods]
    test = df.iloc[train_periods:]

    # Create a dataframe with the training and test datasets
    forecast = pd.DataFrame(index=df.index)
    forecast['train'] = train['demand']
    forecast['test'] = test['demand']

    # ARIMA Model
    if 'ARIMA' in model_list:
        from pmdarima.arima import auto_arima
        
        arima_train = train[['demand']].copy()
        arima_test = test[['demand']].copy()

        if len(arima_train) > 12 and arima_train['demand'].std() > 0:
            try:
                arima_model = auto_arima(
                    y=arima_train['demand'], 
                    start_p=2, 
                    max_p=5, 
                    d=None, 
                    start_q=2, 
                    max_q=5, 
                    start_P=1, 
                    seasonal=seasonal >= 3,
                    m=12, 
                    D=0,
                    n_fits=10, 
                    trace=False, 
                    error_action='ignore',
                    suppress_warnings=True, 
                    stepwise=True, 
                    information_criterion='aic'
                )
                
                forecast_arima = pd.DataFrame(arima_model.predict(test_periods + periods), index=arima_test.index)
                forecast_arima.columns = ['ARIMA']
                forecast = pd.concat([forecast, forecast_arima], axis=1)
            except ValueError as e:
                st.write(f"ARIMA failed: {e}")
                forecast['ARIMA'] = np.nan
        else:
            st.write("Not enough data for ARIMA or data is constant.")
            forecast['ARIMA'] = np.nan

    # STL Model
    if 'STL' in model_list:
        stl_train = train[['demand']].copy()
        stl_test = test[['demand']].copy()

        if isinstance(seasonal, int) and seasonal >= 3 and seasonal % 2 == 1:
            stl_seasonal = seasonal
        else:
            stl_seasonal = 7

        try:
            stl = STL(stl_train['demand'], seasonal=stl_seasonal, period=12)
            stl_result = stl.fit()
            stl_forecast = stl_result.trend[-test_periods:] + stl_result.seasonal[-test_periods:]

            future_index = pd.date_range(start=stl_test.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
            extended_forecast = pd.Series(stl_forecast.iloc[-1], index=future_index)

            stl_forecast = pd.concat([pd.Series(stl_forecast, index=stl_test.index), extended_forecast])

            forecast['STL'] = stl_forecast

        except ValueError as e:
            st.write(f"STL failed: {e}")
            forecast['STL'] = np.nan

    # ETS Model
    if 'ETS' in model_list:
        ets_train = train[['demand']].copy()
        ets_test = test[['demand']].copy()

        if len(ets_train) >= 24:
            seasonal_ets = 'add' if seasonal else None
        else:
            seasonal_ets = None
        
        trend_ets = 'add' if trend else None
        
        try:
            ets_model = ets(ets_train['demand'], trend=trend_ets, seasonal=seasonal_ets).fit()
            forecast_ets = pd.DataFrame(ets_model.predict(start=train_periods, end=len(df) + periods - 1), index=ets_test.index)
            forecast_ets.columns = ['ETS']
            forecast_ets[forecast_ets < 0] = 0
            forecast = pd.concat([forecast, forecast_ets], axis=1)
        except ValueError as e:
            st.write(f"ETS failed: {e}")
            forecast['ETS'] = np.nan
    
    # Prophet Model
    if 'Prophet' in model_list:
        from prophet import Prophet

        prophet_df = df.reset_index().rename(columns={'year_month': 'ds', 'demand': 'y'})
        pro_model = Prophet()

        pro_model.fit(prophet_df)

        future = pro_model.make_future_dataframe(periods=periods, freq='MS', include_history=True)
        test_and_future = future.iloc[train_periods:]
        forecast_pro = pro_model.predict(test_and_future)
        forecast_pro_renamed = forecast_pro[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'Prophet'})
        forecast_pro_renamed = forecast_pro_renamed.set_index('year_month')
        forecast_pro_renamed[forecast_pro_renamed < 0] = 0
        forecast = pd.concat([forecast, forecast_pro_renamed], axis=1)

    # Neural Prophet Model
    if 'Neural Prophet' in model_list:
        from neuralprophet import NeuralProphet

        # Create a new DataFrame with just the 'ds' and 'y' columns
        neuralprophet_df = df.reset_index()[['year_month', 'demand']].rename(columns={'year_month': 'ds', 'demand': 'y'})
        
        npro_model = NeuralProphet()

        npro_model.fit(neuralprophet_df, freq='MS', epochs=50)

        future = npro_model.make_future_dataframe(neuralprophet_df, periods=periods, n_historic_predictions=len(df))
        test_and_future = future.iloc[train_periods:]
        forecast_npro = npro_model.predict(test_and_future)
        forecast_npro_renamed = forecast_npro[['ds', 'yhat1']].rename(columns={'ds': 'year_month', 'yhat1': 'Neural Prophet'})
        forecast_npro_renamed = forecast_npro_renamed.set_index('year_month')
        forecast_npro_renamed[forecast_npro_renamed < 0] = 0
        forecast = pd.concat([forecast, forecast_npro_renamed], axis=1)

    # Calculate the evaluation metrics
    try:
        forecast_eval = forecast.iloc[train_periods:len(df)]
        evaluate_metrics = {
            model: evaluate(forecast_eval['test'], forecast_eval[model], metrics=('mape', 'maape', 'rmse', 'mse', 'mae'))
            for model in model_list if model in forecast.columns and not forecast[model].isna().all()
        }
        evaluate_metrics = pd.DataFrame(evaluate_metrics)
    except Exception as e:
        st.write(f"Error calculating evaluation metrics: {e}")

    # Check if evaluate_metrics is empty
    if evaluate_metrics.empty:
        st.write("Evaluation metrics are empty.")
    else:
        evaluate_metrics = evaluate_metrics.apply(pd.to_numeric, errors='coerce')
        evaluate_metrics = evaluate_metrics.dropna(how='all')
        evaluate_metrics['Best model'] = evaluate_metrics.idxmin(axis=1)
        numeric_cols = evaluate_metrics.select_dtypes(include=['number']).columns
        evaluate_metrics['Best model value'] = evaluate_metrics[numeric_cols].min(axis=1)
        st.write("Evaluation Metrics Summary:", evaluate_metrics)

    return forecast, evaluate_metrics
