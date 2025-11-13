
import os
import pandas as pd

chunk_size = 10 ** 6
output_file = './btc_usdt_data/full_btc_usdt_data_cleaned.csv'
headers = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

def clean_data(df):
    df = df.dropna()
    df = df.drop(columns=['Ignore', 'Close time'])
    # Ensure numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

dtypes = {
    'Open time': 'int64',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'float64',
    'Close time': 'int64',
    'Quote asset volume': 'float64',
    'Number of trades': 'int64',
    'Taker buy base asset volume': 'float64',
    'Taker buy quote asset volume': 'float64',
    'Ignore': 'int64'
}

i=0
for chunk in pd.read_csv('./btc_usdt_data/full_btc_usdt_data.csv', chunksize=chunk_size, header=None, names=headers, skiprows=1, low_memory=False):
    cleaned_chunk = clean_data(chunk)
    cleaned_chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    i+=len(cleaned_chunk)
    print(f'Processed {i} rows')
