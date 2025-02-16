from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from flask_cors import CORS
import joblib

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, Vercel!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    
CORS(app, origins=["https://wattwise-mc6fiopcw-jaime-luceros-projects.vercel.app","https://wattwise-one.vercel.app", "http://localhost:3000"])

# Connect to the SQLite database
engine = create_engine('sqlite:///energy_data.db')

# Define the model loading function
def load_model(metric):
    """
    Load the appropriate pre-trained Seasonal ARIMA model based on the metric.
    """
    try:
        model_map = {
            'Global_active_power': 'models/Global_active_power_sarima_monthly_model.pkl',
            'Global_reactive_power': 'models/Global_reactive_power_sarima_monthly_model.pkl',
            'Global_intensity': 'models/Global_intensity_sarima_monthly_model.pkl',
            'Voltage': 'models/Voltage_sarima_monthly_model.pkl',
        }
        
        model_path = model_map.get(metric)
        if not model_path:
            raise ValueError(f"No model found for the selected metric: {metric}")
        
        # Load the model from the file
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

@app.route('/api/full_query', methods=['GET'])
def full_query():
    """
    Perform a flexible query on the SQLite database with optional filtering, column selection, and pagination.
    """
    try:
        # Parse query parameters
        columns = request.args.getlist('columns')  # List of columns to fetch
        filters = request.args.to_dict(flat=True)  # Dictionary of filters
        filters.pop('columns', None)  # Remove 'columns' if present in filters
        limit = filters.pop('limit', None)  # Extract limit parameter
        offset = filters.pop('offset', None)  # Extract offset parameter

        # Default values for limit and offset
        limit = int(limit) if limit else None  # No limit if not provided
        offset = int(offset) if offset else None  # No offset if not provided

        # Build base query
        base_query = "SELECT * FROM energy_consumption"
        where_clauses = []
        query_params = {}

        # Handle date-based filters (year, month, day)
        year = filters.get('year')
        month = filters.get('month')
        day = filters.get('day')

        # Apply filters for year, month, and day using strftime
        if year:
            where_clauses.append("strftime('%Y', datetime) = :year")
            query_params['year'] = year

        if month:
            where_clauses.append("strftime('%m', datetime) = :month")
            query_params['month'] = month

        if day:
            where_clauses.append("strftime('%d', datetime) = :day")
            query_params['day'] = day

        # Handle other filters (such as user_id)
        user_id = filters.get('user_id')
        if user_id:
            where_clauses.append("consumer_id = :user_id")
            query_params['user_id'] = user_id

        # Add WHERE clause if any filters are applied
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)

        # Handle column selection
        if columns:
            columns_str = ", ".join(columns)
            base_query = base_query.replace("*", columns_str)

        # Add LIMIT and OFFSET if provided
        if limit is not None:
            base_query += " LIMIT :limit"
            query_params['limit'] = limit
        if offset is not None:
            base_query += " OFFSET :offset"
            query_params['offset'] = offset

        # Execute the query
        with engine.connect() as conn:
            result = conn.execute(text(base_query), query_params)
            data = [dict(row) for row in result.mappings()]

        # Return the result as JSON
        if not data:
            return jsonify({"message": "No data found for the specified query"}), 404

        return jsonify(data)

    except SQLAlchemyError as e:
        return jsonify({"error": "Database query failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def summary():
    """
    Provide dataset summary.
    """
    try:
        with engine.connect() as conn:
            # Get the total number of rows
            total_rows = conn.execute(text("SELECT COUNT(*) FROM energy_consumption")).scalar()

            # Get table structure details
            columns_query = conn.execute(text("PRAGMA table_info(energy_consumption)"))
            columns = [{"name": row[1], "type": row[2]} for row in columns_query]  # Use correct tuple indices

        return jsonify({
            'total_rows': total_rows,
            'columns': columns
        })

    except SQLAlchemyError as e:
        return jsonify({"error": "Failed to retrieve summary data", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """
    Handle forecasting requests based on selected metric and number of months.
    """
    try:
        # Ensure the request is in JSON format
        if request.content_type != 'application/json':
            return jsonify({"error": "Content-Type must be 'application/json'"}), 415
        
        # Parse the JSON body of the request
        data = request.get_json()

        # Extract metric and forecast_months from the request data
        metric = data.get('metric')
        forecast_months = data.get('forecast_months')

        # Validate input
        if not metric or not forecast_months:
            return jsonify({"error": "Metric and forecast_months are required"}), 400

        # Load the trained seasonal ARIMA model based on the selected metric
        model = load_model(metric)  # Ensure you have this function to load the model
        forecasted_values = model.forecast(steps=forecast_months)

        # Return the forecasted values
        return jsonify({"forecast": forecasted_values.tolist()})

    except Exception as e:
        # Handle any errors during the forecasting process
        return jsonify({"error": "Error during forecast generation", "details": str(e)}), 500

# Main entry point for the app
if __name__ == '__main__':
    app.run(debug=True)
