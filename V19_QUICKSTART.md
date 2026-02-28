# 🚀 V19 Quick Start Guide

## ⚡ One-Command Training

```bash
python train_v19_dqn.py
```

That's it! Training will run for ~4-6 hours and save checkpoints to `./rl_checkpoints_v19_dqn_fixed/`

---

## 📋 Prerequisites Check

```bash
# 1. Verify data file exists
ls -lh ./btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv

# 2. Verify dependencies
python -c "import stable_baselines3; import pandas; import numpy; import gymnasium; print('✅ All dependencies OK')"

# 3. Verify V19 files exist
ls -lh train_v19_dqn.py eval_v19.py enhanced_trading_environment_v19.py
```

---

## 🎯 Training Workflow

### Step 1: Start Training
```bash
python train_v19_dqn.py
```

**What you'll see:**
```
======================================================================
🚀 DQN TRADING MODEL TRAINING v19 - CRITICAL FIXES APPLIED
======================================================================
Total timesteps: 1,000,000
Checkpoints: every 100,000 steps

V19 CRITICAL FIXES:
  P0 ✅ SaveBestCallback.best_model_path attribute added
  P0 ✅ Reward clipping: 50.0 → 200.0 (prevents information loss)
  ...

Creating environments...
Creating DQN model...
  buffer_size: 100000
  exploration: 1.0 → 0.05 (reduced for stability)
  ...

📚 STARTING DQN TRAINING v19...
----------------------------------------------------------------------

💾 Saving BEST model (reward=15.2341)...
✅ Best model saved!
💾 Checkpoint: 100,000 steps
```

### Step 2: Monitor Training (Optional)

**Option A: TensorBoard**
```bash
tensorboard --logdir ./logs_v19/
# Open browser: http://localhost:6006
```

**Option B: Watch console output**
- Look for increasing `reward` values
- Checkpoints saved every 100K steps

### Step 3: Evaluate Model

After training completes:
```bash
python eval_v19.py
```

**Expected output:**
```
======================================================================
📊 V19 DQN MODEL EVALUATION - CRITICAL FIXES
======================================================================

Ep1: return=+0.45%  trades=42 (L:23 S:19)  win=36%
Ep2: return=+0.12%  trades=48 (L:26 S:22)  win=33%
Ep3: return=+0.67%  trades=39 (L:21 S:18)  win=38%
Ep4: return=+0.28%  trades=45 (L:24 S:21)  win=35%

============================================================
SUMMARY
============================================================

📊 PERFORMANCE METRICS:
├─ Average Return:     +0.38%
├─ Average Win Rate:   35.5%
├─ Total Trades:       174 (43.5 per episode)
│  ├─ Long:            94 (54%)
│  └─ Short:           80 (46%)
└─ Win Rate (overall): 35.5%

🎯 V19 TARGETS:
├─ Return > -0.3%:     ✅ PASS
├─ Win Rate > 30%:     ✅ PASS
├─ Trades < 60:        ✅ PASS
└─ Short % > 20%:      ✅ PASS

✅ ALL V19 TARGETS ACHIEVED!
```

---

## 🔧 Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'stable_baselines3'"

**Solution:**
```bash
pip install stable-baselines3
```

---

### Issue 2: "FileNotFoundError: [Errno 2] No such file or directory: '...feature_engineered.csv'"

**Solution:**
```bash
# Run feature engineering first
python feature_engineer.py
```

---

### Issue 3: Training is slow (< 100 steps/sec)

**Solutions:**
1. Reduce `N_ENVS` from 4 to 2 in `train_v19_dqn.py`
2. Use SSD storage for data files
3. Close other CPU-intensive applications

---

### Issue 4: GPU out of memory

**Solution:** V19 uses CPU-only by default. If using GPU:
```python
# Add to train_v19_dqn.py after DQN creation
model = DQN(..., device='cpu')  # Force CPU
```

---

## 📊 Expected Training Timeline

| Timesteps | Time Elapsed | Expected Mean Reward |
|-----------|--------------|---------------------|
| 0 | 0 min | 0 (random actions) |
| 100K | ~40 min | 5-10 (learning starts) |
| 200K | ~80 min | 10-15 (improving) |
| 500K | ~200 min | 15-20 (stable) |
| 1M | ~400 min | 20-25 (converged) |

---

## 🎯 Success Metrics

After training, your model should achieve:

| Metric | Target | How to Check |
|--------|--------|--------------|
| Best Reward | > 15 | Console output during training |
| Return/Episode | > -0.3% | `eval_v19.py` |
| Win Rate | > 30% | `eval_v19.py` |
| Trades/Episode | < 60 | `eval_v19.py` |
| Short % | > 30% | `eval_v19.py` |

---

## 📁 Output Files

After successful training:

```
./rl_checkpoints_v19_dqn_fixed/
├── dqn_v19_100000_steps.zip      # Checkpoint at 100K steps
├── dqn_v19_200000_steps.zip      # Checkpoint at 200K steps
├── ...
├── dqn_v19_1000000_steps.zip     # Final checkpoint
└── dqn_v19_best.zip              # Best model (highest reward)

./logs_v19/
└── dqn_v19_fixed/
    └── events.out.tfevents.*     # TensorBoard logs
```

---

## 🚀 Next Steps

### After Successful Training:

1. **Evaluate:** `python eval_v19.py`
2. **Paper Trade:** Update model path in `paper_trade_hybrid.py`
3. **Deploy:** Use in live trading (with caution!)

### If Results Don't Meet Targets:

1. Check `V19_TRAINING_STATUS.md` for troubleshooting
2. Review TensorBoard logs
3. Adjust reward parameters in `enhanced_trading_environment_v19.py`

---

## 📞 Quick Reference

| Command | Purpose |
|---------|---------|
| `python train_v19_dqn.py` | Start training |
| `python eval_v19.py` | Evaluate model |
| `tensorboard --logdir ./logs_v19/` | Monitor training |
| `cat V19_SUMMARY.md` | Full documentation |
| `cat V19_TRAINING_STATUS.md` | Detailed fixes |

---

**Last Updated:** 2026-02-27  
**Version:** V19  
**Status:** Production Ready ✅
