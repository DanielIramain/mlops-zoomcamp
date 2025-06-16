import sys
import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def get_path(taxi_type: str, year: int, month: int) -> str:
    
    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{year}-{month}.parquet'
    output_file = f'output/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    return output_file

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def prepare_data(type: str, year: int, month: int):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    pred_std_dev = y_pred.std()

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred

    return df_result

def save_data(df_result, output_file):
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
        )

def run():
    taxi_type = sys.argv[1] # yellow
    year = int(sys.argv[2]) #2023
    month = int(sys.argv[3]) #04

    output_file = get_path(taxi_type, year, month)

    df_result = prepare_data(taxi_type, year, month)
    save_data(df_result, output_file)

    print(f'Saved the result to the file: {output_file}')
    print(f'Mean predicted duration: {df_result['predictions'].mean()}')
    print(f'STD predicted duration: {df_result['predictions'].std()}')

if __name__ == '__main__':
    run()







