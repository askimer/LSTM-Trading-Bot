#!/bin/bash
# V20 Project Cleanup Script
# Safely removes old, obsolete, and temporary files

set -e  # Exit on error

echo "======================================================================"
echo "🧹 V20 PROJECT CLEANUP"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
deleted=0
skipped=0

# Function to safely delete files
safe_delete() {
    local pattern=$1
    local description=$2
    
    if ls $pattern 1>/dev/null 2>&1; then
        count=$(ls -1 $pattern 2>/dev/null | wc -l)
        echo -e "${YELLOW}🗑️  Deleting: $description ($count files)${NC}"
        rm -rf $pattern 2>/dev/null || true
        deleted=$((deleted + count))
    else
        echo -e "${GREEN}✓${NC} Already clean: $description"
        skipped=$((skipped + 1))
    fi
}

cd /Volumes/Movies/PYTHON/RL-Algorithmic-Trading-Bot

echo "======================================================================"
echo "📝 STEP 1: Old Evaluation Scripts (v11-v18)"
echo "======================================================================"
safe_delete "eval_v11_improved.py" "eval_v11_improved.py"
safe_delete "eval_v11_improved_correct.py" "eval_v11_improved_correct.py"
safe_delete "eval_v16.py" "eval_v16.py"
safe_delete "eval_v17.py" "eval_v17.py"
safe_delete "eval_v18.py" "eval_v18.py"

echo ""
echo "======================================================================"
echo "📝 STEP 2: Old Training Scripts (v11-v18)"
echo "======================================================================"
safe_delete "train_v11_improved.py" "train_v11_improved.py"
safe_delete "train_v17_dqn.py" "train_v17_dqn.py"
safe_delete "train_v18_continue.py" "train_v18_continue.py"

echo ""
echo "======================================================================"
echo "📝 STEP 3: Old Environment Files"
echo "======================================================================"
echo -e "${YELLOW}⚠️  Keeping: enhanced_trading_environment_v19.py (reference)${NC}"
echo -e "${YELLOW}⚠️  Keeping: enhanced_trading_environment_v20.py (current)${NC}"
echo -e "${YELLOW}⚠️  Keeping: enhanced_trading_environment.py (original)${NC}"

echo ""
echo "======================================================================"
echo "📝 STEP 4: Old Documentation (V11-V18)"
echo "======================================================================"
safe_delete "HYBRID_MODE_FINAL_STATUS.md" "HYBRID_MODE_FINAL_STATUS.md"
safe_delete "HYBRID_V14_FINAL_STATUS.md" "HYBRID_V14_FINAL_STATUS.md"
safe_delete "HYBRID_V14_STATUS.md" "HYBRID_V14_STATUS.md"
safe_delete "V17_DQN_STATUS.md" "V17_DQN_STATUS.md"
safe_delete "V18_REWARD_FIXES.md" "V18_REWARD_FIXES.md"
safe_delete "V18_TRAINING_STATUS.md" "V18_TRAINING_STATUS.md"
safe_delete "V19_TRAINING_STATUS.md" "V19_TRAINING_STATUS.md"
safe_delete "V19_SUMMARY.md" "V19_SUMMARY.md"
safe_delete "V19.1_FIXES.md" "V19.1_FIXES.md"
safe_delete "V19.2_FIXES.md" "V19.2_FIXES.md"
safe_delete "V19.3_FIXES.md" "V19.3_FIXES.md"
safe_delete "V20_PLAN.md" "V20_PLAN.md (merged into V20_IMPROVEMENTS.md)"

echo ""
echo "======================================================================"
echo "📝 STEP 5: Old Log Files"
echo "======================================================================"
safe_delete "*.log" "All .log files"
safe_delete "v19*.log" "V19 training logs"
safe_delete "v20_training.log" "Current V20 log (keeping for monitoring)"

echo ""
echo "======================================================================"
echo "📝 STEP 6: Temporary & Test Files"
echo "======================================================================"
safe_delete "paper_trade_output.log" "paper_trade_output.log"
safe_delete "paper_trading_results.pkl" "paper_trading_results.pkl"
safe_delete "trading_results_live.pkl" "trading_results_live.pkl"
safe_delete "test_v19.3.py" "test_v19.3.py (replaced by trade.py)"

echo ""
echo "======================================================================"
echo "📝 STEP 7: Obsolete Checkpoint Directories"
echo "======================================================================"
echo -e "${RED}⚠️  SKIPPING: Checkpoint directories (manual review recommended)${NC}"
echo "   - rl_checkpoints_profitable/ (PPO models - obsolete)"
echo "   - rl_checkpoints_v17_dqn/ (V17 DQN models - obsolete)"
echo "   - rl_checkpoints_v18_dqn_improved/ (V18 DQN models - obsolete)"
echo ""
echo "   To delete later:"
echo "   rm -rf rl_checkpoints_profitable/"
echo "   rm -rf rl_checkpoints_v17_dqn/"
echo "   rm -rf rl_checkpoints_v18_dqn_improved/"

echo ""
echo "======================================================================"
echo "📝 STEP 8: Obsolete Python Scripts"
echo "======================================================================"
echo -e "${YELLOW}⚠️  Review these manually:${NC}"
echo "   - eval_model.py (generic - may still be useful)"
echo "   - paper_trade_test.py (test script)"
echo "   - paper_trade_v15.py (V15 specific)"
echo "   - paper_trade_hybrid.py (hybrid mode - may still be useful)"
echo "   - rule_based_entry_env.py (rule-based - reference)"
echo "   - trading_environment.py (original - reference)"
echo "   - rl_pipeline.py, rl_paper_trading.py, rl_live_trading.py (old pipeline)"
echo "   - run_pipeline.py (pipeline runner)"
echo "   - stability_callback.py, state_utils.py (utilities)"

echo ""
echo "======================================================================"
echo "📊 CLEANUP SUMMARY"
echo "======================================================================"
echo -e "${GREEN}✅ Files deleted: $deleted${NC}"
echo -e "${YELLOW}⚠️  Items skipped: $skipped${NC}"
echo ""
echo "======================================================================"
echo "📁 CURRENT ACTIVE FILES"
echo "======================================================================"
echo "✅ Environment:"
echo "   - enhanced_trading_environment_v20.py (CURRENT)"
echo "   - enhanced_trading_environment_v19.py (REFERENCE)"
echo ""
echo "✅ Training:"
echo "   - train_v20_dqn.py (CURRENT)"
echo "   - train_v19_dqn.py (REFERENCE)"
echo ""
echo "✅ Evaluation:"
echo "   - eval_v19.py (REFERENCE)"
echo "   - trade.py --paper (UNIFIED)"
echo ""
echo "✅ Trading:"
echo "   - trade.py (UNIFIED: --paper, --live)"
echo ""
echo "✅ Documentation:"
echo "   - V20_IMPROVEMENTS.md (CURRENT)"
echo "   - V19.3_TRADE_FIXES.md (REFERENCE)"
echo "   - README.md (GENERAL)"
echo ""
echo "======================================================================"
echo "✨ CLEANUP COMPLETE!"
echo "======================================================================"
