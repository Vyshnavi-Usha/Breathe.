import flask
from flask import Flask, render_template, request
import requests
#global city_name
app = Flask(__name__)

#city_name=None
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


@app.route('/', methods=['GET', 'POST'])
def index():
    city_data = None
    uv_ozone_data = None
    pollutant_ids = []
    ml_result=[]

    if request.method == 'POST':
        #global city_name
        city_name = request.form.get('city')
        city_data = get_city_data(city_name)
        
        if city_data and 'coordinates' in city_data:
            coordinates = city_data['coordinates']
            uv_ozone_data = get_uv_ozone_data(coordinates)
            pollutant_ids = [pollutant['id'] for pollutant in city_data['pollutant_data']]
            ml_result = city_give(city_data)
            print("Machine Learning Result:", ml_result)

    return render_template('index.html', city_data=city_data, uv_ozone_data=uv_ozone_data,pollutant_ids=pollutant_ids,ml_result=ml_result)

def get_city_data(city_name):
    api_url = 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69'
    api_key = '579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b'

    response_API = requests.get(f'{api_url}?api-key={api_key}&format=json')
    
    if response_API.status_code == 200:
        data = response_API.json()
        records = [record for record in data.get('records', []) if record.get('city', '').lower() == city_name.lower()]
        print(records)
        if records:
            '''lis=[]
            lis.append(records[0].get('state', ''))
            lis.append(records[0].get('city', ''))
            lis.append([{'id': record.get('pollutant_id', ''), 'avg': record.get('pollutant_avg', '')} for record in records])
            lis.append(get_lat_lng(city_name))
            return lis'''

           # city_give(city_name)

            return {
                'state': records[0].get('state', ''),
                'city': records[0].get('city', ''),
                'pollutant_data': [{'id': record.get('pollutant_id', ''), 'avg': record.get('pollutant_avg', '')} for record in records],
                'coordinates': get_lat_lng(city_name)

            }
    return None


def get_lat_lng(city_name):
    base_url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': city_name,
        'format': 'json',
        'limit': 1  # Limit the number of results to 1
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        if data and len(data) > 0:
            # Assuming the first result is the desired location
            return float(data[0]['lat']), float(data[0]['lon'])

    print(f'Could not find coordinates for {city_name}')
    return None

def get_uv_ozone_data(coordinates):
    if coordinates is None:
        print('Coordinates are None. Unable to fetch UV and Ozone data.')
        return None

    api_url = 'https://api.openuv.io/api/v1/uv'
    api_key = 'openuv-bytfqeorlrvtmbw8-io'
    
    headers = {
        'x-access-token': api_key,
    }

    params = {
        'lat': coordinates[0],
        'lng': coordinates[1],
    }

    response_API = requests.get(api_url, params=params, headers=headers)

    if response_API.status_code == 200:
        data = response_API.json()
        return {
            'uv': data.get('result', {}).get('uv', ''),
            'ozone': data.get('result', {}).get('ozone', ''),
        }
    
    print('Failed to retrieve UV and Ozone data from the API.')
    return None


def city_give(data):
    
# Load the data
    file_path = r'C:\Users\harin\Downloads\archive\station_day.csv'
    df = pd.read_csv(file_path,low_memory=False)

# Drop specified columns
    df = df.drop(['StationId', 'Date'], axis=1)

# Encode aqi bucket
# Define the mapping dictionary
    aqi_bucket_mapping = {
        'Very poor': -2,
        'Poor': -1,
        'Moderate': 0,
        'Satisfactory': 1,
        'Good': 2
    }

# Apply the mapping to the 'AQI Bucket' column
    df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_bucket_mapping)

# Set the threshold for NaN values
    threshold = 7

# Drop rows with more than 7 NaN values
    df = df.dropna(thresh=df.shape[1] - threshold)

# Fill NaN values with the mean for each column
    df = df.fillna(df.mean())

# Convert 'AQI_Bucket' column to integers
    df['AQI_Bucket'] = df['AQI_Bucket'].astype(int)


# Display the updated DataFrame
    print(df.head(20))

# Create feature matrix X (all columns except 'AQI')
    X = df.drop(['AQI','AQI_Bucket'], axis=1)

# Create target variable y ('AQI' column)
    y = df['AQI']

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    d=data
    print(d)
    r=[]
    for i in d:
        if i=='pollutant_data':
            r.append(d[i])
    print(r)
    l=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','OZONE','Benzene','Toulene','Xylene']
    l1=[]
    pollutant_data = r[0]

# Extracting average values as a dictionary
    avg_values = {pollutant['id']: float(pollutant['avg']) for pollutant in pollutant_data}

    print(avg_values)
    for i in l:
        if i in avg_values:
            l1.append(int(avg_values[i]))
        else:
            l1.append(0)
    print(l1)
            
    
    

    l1_reshaped = np.array(l1).reshape(1, -1)
    print(l1_reshaped)

    l2=pd.DataFrame(l1_reshaped)
    print(l2)
    future_predictions = model.predict(l1_reshaped)

# Print the predictions
    print("Future AQI Predictions based on new pollutant values:")
    print(future_predictions)
    

    return future_predictions

@app.route("/second")
def three():
    return render_template('three.html')

if __name__ == '__main__':
    app.run(debug=True)