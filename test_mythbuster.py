#!/usr/bin/env python3
"""
Simple test to validate MythBuster AI functionality
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import app
        print("✅ app.py imports successfully")
    except Exception as e:
        print(f"❌ Failed to import app.py: {e}")
        return False
        
    try:
        import config
        print("✅ config.py imports successfully")
    except Exception as e:
        print(f"❌ Failed to import config.py: {e}")
        return False
        
    try:
        import services
        print("✅ services.py imports successfully")
    except Exception as e:
        print(f"❌ Failed to import services.py: {e}")
        return False
        
    return True

def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    from utils import sanitize_input, validate_claim, is_vague_response
    
    # Test input sanitization
    dirty_input = "<script>alert('xss')</script>This is a test claim"
    clean_input = sanitize_input(dirty_input)
    assert "<script>" not in clean_input
    print("✅ Input sanitization works")
    
    # Test claim validation
    assert validate_claim("This is a valid claim") == True
    assert validate_claim("") == False  
    assert validate_claim("abc") == False
    print("✅ Claim validation works")
    
    # Test vague response detection
    assert is_vague_response("I don't know") == True
    assert is_vague_response("This claim is clearly false") == False
    print("✅ Vague response detection works")
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    from config import config
    
    # Check that config has required attributes
    assert hasattr(config, 'groq_api_key')
    assert hasattr(config, 'embedding_model_name')
    assert hasattr(config, 'vector_store_path')
    print("✅ Configuration loaded successfully")
    
    return True

def test_models():
    """Test data models."""
    print("\nTesting data models...")
    
    from models import MythResult, Verdict
    
    # Test verdict enum
    assert Verdict.BUSTED.value == "BUSTED"
    assert Verdict.CONFIRMED.value == "CONFIRMED"
    print("✅ Verdict enum works")
    
    # Test myth result model
    result = MythResult(
        claim="Test claim",
        verdict=Verdict.BUSTED,
        reasoning="Test reasoning",
        source="test"
    )
    assert result.claim == "Test claim"
    assert result.verdict == Verdict.BUSTED
    print("✅ MythResult model works")
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running MythBuster AI Tests...\n")
    
    tests = [
        test_imports,
        test_configuration, 
        test_models,
        test_utility_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MythBuster AI is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())