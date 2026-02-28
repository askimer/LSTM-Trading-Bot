#!/usr/bin/env python3
"""
Download 13 months (395 days) of BTC/USDT 1-minute klines from Binance
"""

import requests
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_process_data(start_date, end_date, symbol='BTCUSDT', interval='1m'):
    output_csv = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
    headers = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 
               'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 
               'Taker buy quote asset volume', 'Ignore']
    
    # Create directory
    os.makedirs('btc_usdt_data', exist_ok=True)
    os.makedirs('btc_usdt_training_data', exist_ok=True)
    
    # Initialize CSV with header
    all_data = []
    
    # Calculate total days for progress bar
    total_days = (end_date - start_date).days + 1
    
    current_date = start_date
    with tqdm(total=total_days, desc='Downloading') as pbar:
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            url = f'https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip'
            file_name = f'{date_str}.zip'
            file_path = os.path.join('btc_usdt_data', file_name)

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as file:
                    file.write(response.content)

                unzip_file(file_path, 'btc_usdt_data')
                os.remove(file_path)
                unzipped_file_path = f'btc_usdt_data/{symbol}-{interval}-{date_str}.csv'

                # Read the data
                daily_data = pd.read_csv(unzipped_file_path, header=None, names=headers)
                all_data.append(daily_data)

                # Delete the unzipped file
                os.remove(unzipped_file_path)

            except requests.RequestException as e:
                print(f'Error for {date_str}: {e}')

            current_date += timedelta(days=1)
            pbar.update(1)
    
    # Combine all data
    print('\nCombining all data...')
    full_data = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    print(f'Saving to {output_csv}...')
    full_data.to_csv(output_csv, index=False)
    
    print(f"\n✅ Complete!")
    print(f"📊 Total rows: {len(full_data):,}")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"💾 Saved to: {output_csv}")

if __name__ == '__main__':
    # 13 months = 395 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=395)
    
    print("=" * 70)
    print("📥 DOWNLOADING BTC/USDT 1-MINUTE K-LINES")
    print("=" * 70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Days: 395 (~13 months)")
    print("=" * 70)
    
    download_and_process_data(start_date, end_date)
