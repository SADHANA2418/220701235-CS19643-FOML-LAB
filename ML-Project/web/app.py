# # # # from flask import Flask, render_template, jsonify
# # # # import pandas as pd
# # # # import os

# # # # app = Flask(__name__)

# # # # # Route to render the HTML page
# # # # @app.route('/')
# # # # def index():
# # # #     return render_template('index.html')

# # # # # API route to serve prediction data
# # # # @app.route('/get_predictions')
# # # # def get_predictions():
# # # #     csv_path = 'web/live_predictions.csv'
    
# # # #     if not os.path.exists(csv_path):
# # # #         return jsonify({'error': 'Prediction file not found'}), 404

# # # #     try:
# # # #         df = pd.read_csv(csv_path)

# # # #         # If you want to limit how many rows are sent
# # # #         df = df[['Date', 'Server_ID', 'Predicted_Failure_Probability']]
# # # #         data = df.to_dict(orient='records')
# # # #         return jsonify(data)
# # # #     except Exception as e:
# # # #         return jsonify({'error': str(e)}), 500

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)


# # # from flask import Flask, render_template, jsonify, request
# # # import pandas as pd
# # # import os

# # # app = Flask(__name__)

# # # # Route to render the HTML page
# # # @app.route('/')
# # # def index():
# # #     return render_template('index.html')

# # # # API route to serve prediction data
# # # @app.route('/get_predictions')
# # # def get_predictions():
# # #     csv_path = 'web/live_predictions.csv'
    
# # #     if not os.path.exists(csv_path):
# # #         return jsonify({'error': 'Prediction file not found'}), 404

# # #     try:
# # #         df = pd.read_csv(csv_path)

# # #         # Select relevant columns
# # #         df = df[['Date', 'Server_ID', 'Predicted_Failure_Probability']]
        
# # #         # Get top 5 servers by failure probability
# # #         top_5 = df.sort_values(by='Predicted_Failure_Probability', ascending=False).drop_duplicates(subset='Server_ID').head(5)
        
# # #         data = top_5.to_dict(orient='records')
# # #         return jsonify(data)
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # # API route to fetch specific server details by Server_ID
# # # @app.route('/get_server_details/<server_id>')
# # # def get_server_details(server_id):
# # #     csv_path = 'web/live_predictions.csv'
    
# # #     if not os.path.exists(csv_path):
# # #         return jsonify({'error': 'Prediction file not found'}), 404

# # #     try:
# # #         df = pd.read_csv(csv_path)

# # #         # Get details for the specified Server_ID
# # #         server_data = df[df['Server_ID'] == server_id]
        
# # #         if server_data.empty:
# # #             return jsonify({'error': 'Server not found'}), 404

# # #         # Get the first row (since Server_ID is unique)
# # #         server_info = server_data.iloc[0]
# # #         result = {
# # #             "Server_ID": server_info["Server_ID"],
# # #             "Predicted_Failure_Probability": server_info["Predicted_Failure_Probability"],
# # #             "Risk_Level": "High" if server_info["Predicted_Failure_Probability"] > 0.7 else "Medium" if server_info["Predicted_Failure_Probability"] > 0.4 else "Low",
# # #             "Suggestions": "Consider load balancing" if server_info["Predicted_Failure_Probability"] > 0.7 else "Monitor server performance"
# # #         }
# # #         return jsonify(result)
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 500

# # # if __name__ == '__main__':
# # #     app.run(debug=True)


# # from flask import Flask, render_template, jsonify
# # import pandas as pd
# # import os

# # app = Flask(__name__)

# # # Route to render the HTML page
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # # API route to serve prediction data
# # @app.route('/get_predictions')
# # def get_predictions():
# #     csv_path = 'web/live_predictions.csv'
    
# #     if not os.path.exists(csv_path):
# #         return jsonify({'error': 'Prediction file not found'}), 404

# #     try:
# #         df = pd.read_csv(csv_path)

# #         # Ensure Date is in datetime format
# #         df['Date'] = pd.to_datetime(df['Date'])

# #         # Group by Server_ID and get the most recent record for each server
# #         recent_server_data = df.sort_values('Date', ascending=False).drop_duplicates('Server_ID')

# #         # Sort these servers by the Predicted_Failure_Probability (highest first)
# #         top_5_servers = recent_server_data.sort_values('Predicted_Failure_Probability', ascending=False).head(5)

# #         # Prepare data for the response
# #         data = top_5_servers[['Date', 'Server_ID', 'Predicted_Failure_Probability']].to_dict(orient='records')

# #         return jsonify(data)
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # # API route to get details for a specific server
# # @app.route('/get_server_details/<server_id>')
# # def get_server_details(server_id):
# #     csv_path = 'web/live_predictions.csv'
    
# #     if not os.path.exists(csv_path):
# #         return jsonify({'error': 'Prediction file not found'}), 404

# #     try:
# #         # Read the CSV
# #         df = pd.read_csv(csv_path)

# #         # Filter the data for the specific server
# #         server_data = df[df['Server_ID'] == server_id]

# #         # Sort by date to get the most recent value
# #         server_data['Date'] = pd.to_datetime(server_data['Date'])
# #         most_recent_data = server_data.sort_values(by='Date', ascending=False).iloc[0]

# #         # Extract the most recent predicted failure probability
# #         failure_probability = most_recent_data['Predicted_Failure_Probability']
# #         return jsonify({
# #             'Server_ID': server_id,
# #             'Predicted_Failure_Probability': failure_probability,
# #             'Date': most_recent_data['Date'].strftime('%Y-%m-%d %H:%M:%S')
# #         })
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, render_template, jsonify
# import pandas as pd
# import os

# app = Flask(__name__)

# # Route to render the HTML page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # API route to serve prediction data
# @app.route('/get_predictions')
# def get_predictions():
#     csv_path = 'web/live_predictions.csv'
    
#     if not os.path.exists(csv_path):
#         return jsonify({'error': 'Prediction file not found'}), 404

#     try:
#         df = pd.read_csv(csv_path)

#         # Ensure Date is in datetime format
#         df['Date'] = pd.to_datetime(df['Date'])

#         # Group by Server_ID and get the most recent record for each server
#         recent_server_data = df.sort_values('Date', ascending=False).drop_duplicates('Server_ID')

#         # Sort these servers by the Predicted_Failure_Probability (highest first)
#         top_5_servers = recent_server_data.sort_values('Predicted_Failure_Probability', ascending=False).head(5)

#         # Prepare data for the response
#         data = top_5_servers[['Date', 'Server_ID', 'Predicted_Failure_Probability']].to_dict(orient='records')

#         return jsonify(data)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # API route to get details for a specific server
# @app.route('/get_server_details/<server_id>')
# def get_server_details(server_id):
#     csv_path = 'web/live_predictions.csv'
    
#     if not os.path.exists(csv_path):
#         return jsonify({'error': 'Prediction file not found'}), 404

#     try:
#         # Read the CSV
#         df = pd.read_csv(csv_path)

#         # Filter the data for the specific server
#         server_data = df[df['Server_ID'] == server_id]

#         # Sort by date to get the most recent value
#         server_data['Date'] = pd.to_datetime(server_data['Date'])
#         most_recent_data = server_data.sort_values(by='Date', ascending=False).iloc[0]

#         # Extract the most recent predicted failure probability
#         failure_probability = most_recent_data['Predicted_Failure_Probability']
#         return jsonify({
#             'Server_ID': server_id,
#             'Predicted_Failure_Probability': failure_probability,
#             'Date': most_recent_data['Date'].strftime('%Y-%m-%d %H:%M:%S')
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Path to the CSV file
CSV_PATH = 'web/live_predictions.csv'


@app.route('/')
def index():
    """Render the main dashboard HTML page."""
    return render_template('index.html')


@app.route('/get_predictions')
def get_predictions():
    """
    Return prediction data for all servers.
    Used for:
      - Scrollable table
      - Top 5 risky servers chart
    """
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'Prediction file not found'}), 404

    try:
        df = pd.read_csv(CSV_PATH)

        # Ensure Date column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by latest date for each server
        latest_per_server = df.sort_values('Date', ascending=False).drop_duplicates('Server_ID')

        # Sort descending by failure probability
        latest_per_server = latest_per_server.sort_values('Predicted_Failure_Probability', ascending=False)

        # Format response
        data = latest_per_server[['Date', 'Server_ID', 'Predicted_Failure_Probability']] \
            .sort_values('Date', ascending=False) \
            .to_dict(orient='records')

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_server_details/<server_id>')
def get_server_details(server_id):
    """
    Return the most recent prediction for the given server ID.
    Used when searching a specific server.
    """
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'Prediction file not found'}), 404

    try:
        df = pd.read_csv(CSV_PATH)
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter for the server ID
        server_data = df[df['Server_ID'] == server_id].sort_values('Date', ascending=False)

        if server_data.empty:
            return jsonify({'error': f"No data found for Server_ID: {server_id}"}), 404

        latest = server_data.iloc[0]

        return jsonify({
            'Server_ID': latest['Server_ID'],
            'Predicted_Failure_Probability': round(latest['Predicted_Failure_Probability'], 4),
            'Date': latest['Date'].strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
