#!/bin/bash
# Auto-run feature engineering after download completes

set -e

echo "======================================================================"
echo "⏳ WAITING FOR DOWNLOAD TO COMPLETE..."
echo "======================================================================"

# Wait for download process to finish
while pgrep -f "download_13months.py" > /dev/null; do
    sleep 30
    echo "  Download in progress... ($(tail -1 download_data.log 2>/dev/null | grep -o '[0-9]*/396' || echo 'checking'))"
done

echo ""
echo "✅ DOWNLOAD COMPLETE!"
echo ""

# Check if download was successful
if [ ! -f "btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv" ]; then
    echo "❌ Downloaded file not found!"
    exit 1
fi

echo "======================================================================"
echo "🔧 STARTING FEATURE ENGINEERING..."
echo "======================================================================"

# Run feature engineering
source .venv/bin/activate
python process_features_v20.py

echo ""
echo "======================================================================"
echo "✅ ALL COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check processed data: btc_usdt_training_data/full_btc_usdt_data_feature_engineered_v20.csv"
echo "  2. Update train_v20_dqn.py to use new data file"
echo "  3. Continue/restart V20 training with fresh data"
echo ""
