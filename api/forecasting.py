import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import matplotlib.dates as mdates


# Function to fetch data from the API
def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()  # Returns a list of dictionaries
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

# Function to prepare the data for forecasting
def prepare_data(data):
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert 'datetime' to pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set datetime as the index
    df.set_index('datetime', inplace=True)
    
    # Optional: Convert any other columns to numeric (if not already)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])
    df['Global_intensity'] = pd.to_numeric(df['Global_intensity'])
    df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'])
    df['Voltage'] = pd.to_numeric(df['Voltage'])
    
    return df

# Function to group by month and aggregate (e.g., sum or mean)
def aggregate_by_month(df, target_column):
    # Resample the data by month and aggregate (e.g., sum or mean)
    monthly_data = df.resample('M').sum()[target_column]
    return monthly_data

# Function to train SARIMA model for monthly forecasting
def train_monthly_sarima_model(df, target_column):
    # Aggregate data by month
    monthly_data = aggregate_by_month(df, target_column)
    
    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(len(monthly_data) * 0.8)
    train, test = monthly_data[:train_size], monthly_data[train_size:]
    
    # Fit the SARIMA model with seasonal_order=(1, 1, 1, 12) for monthly seasonality
    seasonal_order = (1, 1, 1, 12)  # Seasonal period of 12 for monthly forecasting
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order)
    model_fit = model.fit()
    
    # Save the model using pickle
    model_filename = f"{target_column}_sarima_monthly_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model_fit, f)
    print(f"SARIMA monthly model for {target_column} saved to {model_filename}")
    
    # Make predictions with the trained model
    predictions = model_fit.forecast(steps=len(test))
    
    # Calculate RMSE (Root Mean Squared Error) to evaluate the model
    error = rmse(test, predictions)
    print(f"RMSE for {target_column}: {error}")
    
    return model_fit, test, predictions

# Function to plot the forecast results
def plot_forecast(df, target_column, test, predictions):
    plt.figure(figsize=(10, 6))
    
    # Plot the actual data (monthly)
    plt.plot(df.resample('M').sum().index, df.resample('M').sum()[target_column], label='Actual', color='blue')
    
    # Plot the predicted data
    plt.plot(test.index, predictions, label='Forecasted', color='red')
    
    # Format the plot
    plt.title(f'{target_column} Monthly SARIMA Forecasting')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.legend()
    
    # Rotate date labels for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()



# Main function to execute the script
def main():
    api_url = "http://127.0.0.1:5000/api/full_query?"  # Your API endpoint
    
    # Fetch data from the API
    data = fetch_data(api_url)
    
    # Prepare the data
    df = prepare_data(data)
    
    # Train and forecast for each metric
    metrics = ['Global_active_power', 'Global_intensity', 'Global_reactive_power', 'Voltage']
    
    for metric in metrics:
        print(f"\nTraining SARIMA model for {metric} (Monthly Forecasting)...")
        model_fit, test, predictions = train_monthly_sarima_model(df, metric)
        
        # Plot the forecast
        plot_forecast(df, metric, test, predictions)

# Run the script
if __name__ == "__main__":
    main()
