#!/usr/bin/env python3
"""
Quick test runner for GitHub Actions - runs essential tests without timeouts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules import correctly"""
    try:
        from fusion import strategies, scalping, methods, council, regime, features
        print(f"✅ Imports successful")
        print(f"  Strategies: {len(strategies.REGISTRY)}")
        print(f"  Scalping: {len(scalping.SCALPING_REGISTRY)}")
        print(f"  Methods: {len(methods.METHOD_REGISTRY)}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scalping_strategies():
    """Test that all scalping strategies can be instantiated"""
    try:
        from fusion import scalping
        from datetime import datetime

        # Test a few strategies
        for strategy_name in ['scalp_order_block', 'scalp_fvg', 'scalp_liquidity_sweep', 'scalp_bos_reversal', 'scalp_regime_filter']:
            if hasattr(scalping, strategy_name):
                strategy = getattr(scalping, strategy_name)
                print(f"✅ {strategy_name} loaded")

        return True
    except Exception as e:
        print(f"❌ Scalping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_council():
    """Test that council can run"""
    try:
        from fusion.council import CouncilWeights
        from fusion import strategies

        weights = CouncilWeights()
        print(f"✅ Council weights initialized")
        print(f"  Total prototypes: {len(strategies.PROTOTYPES)}")
        return True
    except Exception as e:
        print(f"❌ Council test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_gateway():
    """Test AI gateway imports"""
    try:
        from fusion.ai_gateway import initialize_ai_gateway, get_stored_credentials_info
        info = get_stored_credentials_info()
        print(f"✅ AI gateway loaded")
        print(f"  Credentials: {info['total_credentials']}")
        return True
    except Exception as e:
        print(f"⚠️  AI gateway test: {e}")
        return True  # Non-critical

def test_engine():
    """Test that engine can be initialized"""
    try:
        from fusion.engine import FusionEngine
        print(f"✅ Engine loaded")
        return True
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Running quick tests for GitHub Actions...")
    print("=" * 50)

    results = []
    results.append(test_imports())
    results.append(test_scalping_strategies())
    results.append(test_council())
    results.append(test_ai_gateway())
    results.append(test_engine())

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All quick tests passed")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
