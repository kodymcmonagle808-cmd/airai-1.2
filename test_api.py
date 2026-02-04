#!/usr/bin/env python3
"""
AirAI 1.2 API Test Suite
Comprehensive testing for all API endpoints
"""

import requests
import json
import time
import sys

API_URL = "http://localhost:8000"
API_KEY = "AI-kokokwusu"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")

def print_pass(msg):
    print(f"{Colors.GREEN}  ✓ PASS:{Colors.END} {msg}")

def print_fail(msg):
    print(f"{Colors.RED}  ✗ FAIL:{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.YELLOW}  ℹ INFO:{Colors.END} {msg}")


def test_health_check():
    """Test health check endpoint"""
    print_test("Health Check (No Auth Required)")
    
    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        
        if response.status_code == 200:
            print_pass(f"Status code: {response.status_code}")
        else:
            print_fail(f"Status code: {response.status_code}")
            return False
        
        if data.get('status') == 'healthy':
            print_pass("Server is healthy")
        else:
            print_fail("Server health check failed")
            return False
        
        if data.get('model_loaded'):
            print_pass("Model is loaded")
        else:
            print_fail("Model not loaded")
            return False
        
        print_info(f"Version: {data.get('version')}")
        return True
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def test_get_info():
    """Test model info endpoint"""
    print_test("Get Model Info (Auth Required)")
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.get(f"{API_URL}/info", headers=headers)
        data = response.json()
        
        if response.status_code == 200:
            print_pass(f"Status code: {response.status_code}")
        else:
            print_fail(f"Status code: {response.status_code}")
            return False
        
        if 'parameters' in data:
            print_pass(f"Parameters: {data['parameters']:,}")
        
        if 'config' in data:
            print_pass("Config present")
            config = data['config']
            print_info(f"  Vocab size: {config.get('vocab_size')}")
            print_info(f"  Layers: {config.get('num_layers')}")
            print_info(f"  Heads: {config.get('num_heads')}")
        
        return True
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def test_auth_failure():
    """Test authentication failure"""
    print_test("Authentication Failure (Invalid Key)")
    
    headers = {"Authorization": "Bearer INVALID_KEY"}
    
    try:
        response = requests.get(f"{API_URL}/info", headers=headers)
        
        if response.status_code == 401:
            print_pass("Correctly rejected invalid key")
            return True
        else:
            print_fail(f"Expected 401, got {response.status_code}")
            return False
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def test_generate_basic():
    """Test basic text generation"""
    print_test("Text Generation - Basic")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": "hello",
        "max_length": 30,
        "temperature": 0.8,
        "top_k": 50
    }
    
    try:
        response = requests.post(
            f"{API_URL}/generate",
            headers=headers,
            json=payload
        )
        data = response.json()
        
        if response.status_code == 200:
            print_pass(f"Status code: {response.status_code}")
        else:
            print_fail(f"Status code: {response.status_code}")
            return False
        
        if data.get('status') == 'success':
            print_pass("Generation successful")
        else:
            print_fail("Generation failed")
            return False
        
        if 'generated_text' in data:
            print_pass("Generated text present")
            print_info(f"Output: '{data['generated_text']}'")
        
        return True
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def test_generate_parameters():
    """Test generation with different parameters"""
    print_test("Text Generation - Various Parameters")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    test_cases = [
        {"prompt": "hello", "max_length": 10, "temperature": 0.5},
        {"prompt": "the", "max_length": 20, "temperature": 1.0},
        {"prompt": "python", "max_length": 15, "temperature": 0.3},
    ]
    
    passed = 0
    for i, payload in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{API_URL}/generate",
                headers=headers,
                json=payload
            )
            data = response.json()
            
            if response.status_code == 200 and data.get('status') == 'success':
                print_pass(f"Test case {i}: '{payload['prompt']}' -> '{data.get('generated_text')}'")
                passed += 1
            else:
                print_fail(f"Test case {i} failed")
        
        except Exception as e:
            print_fail(f"Test case {i} exception: {e}")
    
    return passed == len(test_cases)


def test_train():
    """Test model training"""
    print_test("Model Training")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "texts": [
            "artificial intelligence is amazing",
            "machine learning transforms data",
            "neural networks learn patterns",
        ],
        "epochs": 3,
        "learning_rate": 0.0001
    }
    
    try:
        print_info("Starting training...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/train",
            headers=headers,
            json=payload
        )
        data = response.json()
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print_pass(f"Status code: {response.status_code}")
        else:
            print_fail(f"Status code: {response.status_code}")
            return False
        
        if data.get('status') == 'success':
            print_pass("Training completed")
            print_info(f"Samples: {data.get('samples_trained')}")
            print_info(f"Epochs: {data.get('epochs')}")
            print_info(f"Time: {elapsed:.2f}s")
        else:
            print_fail("Training failed")
            return False
        
        return True
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def test_invalid_params():
    """Test invalid parameter handling"""
    print_test("Invalid Parameters Handling")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    test_cases = [
        {"name": "Empty prompt", "payload": {"prompt": ""}},
        {"name": "Invalid max_length", "payload": {"prompt": "test", "max_length": -1}},
        {"name": "Invalid temperature", "payload": {"prompt": "test", "temperature": -1}},
        {"name": "Invalid top_k", "payload": {"prompt": "test", "top_k": 0}},
    ]
    
    passed = 0
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/generate",
                headers=headers,
                json=test_case["payload"]
            )
            
            if response.status_code == 400:
                print_pass(f"{test_case['name']}: Correctly rejected (400)")
                passed += 1
            else:
                print_fail(f"{test_case['name']}: Expected 400, got {response.status_code}")
        
        except Exception as e:
            print_fail(f"{test_case['name']}: Exception: {e}")
    
    return passed == len(test_cases)


def test_save_load():
    """Test model save and load"""
    print_test("Model Save and Load")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    save_path = "/tmp/test_airai_model.pkl"
    
    # Save model
    try:
        print_info("Saving model...")
        response = requests.post(
            f"{API_URL}/save",
            headers=headers,
            json={"filepath": save_path}
        )
        data = response.json()
        
        if response.status_code == 200 and data.get('status') == 'success':
            print_pass("Model saved successfully")
        else:
            print_fail("Failed to save model")
            return False
    
    except Exception as e:
        print_fail(f"Save exception: {e}")
        return False
    
    # Load model
    try:
        print_info("Loading model...")
        response = requests.post(
            f"{API_URL}/load",
            headers=headers,
            json={"filepath": save_path}
        )
        data = response.json()
        
        if response.status_code == 200 and data.get('status') == 'success':
            print_pass("Model loaded successfully")
            print_info(f"Parameters: {data.get('parameters'):,}")
            return True
        else:
            print_fail("Failed to load model")
            return False
    
    except Exception as e:
        print_fail(f"Load exception: {e}")
        return False


def test_api_documentation():
    """Test API documentation endpoint"""
    print_test("API Documentation")
    
    try:
        response = requests.get(f"{API_URL}/")
        data = response.json()
        
        if response.status_code == 200:
            print_pass(f"Status code: {response.status_code}")
        else:
            print_fail(f"Status code: {response.status_code}")
            return False
        
        if 'endpoints' in data:
            print_pass("Endpoints documentation present")
            print_info(f"API: {data.get('name')} v{data.get('version')}")
        
        return True
    
    except Exception as e:
        print_fail(f"Exception: {e}")
        return False


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 60)
    print("  AirAI 1.2 API Test Suite")
    print("=" * 60)
    print(f"\nAPI URL: {API_URL}")
    print(f"API Key: {API_KEY}")
    
    tests = [
        ("Health Check", test_health_check),
        ("API Documentation", test_api_documentation),
        ("Authentication Failure", test_auth_failure),
        ("Get Model Info", test_get_info),
        ("Basic Generation", test_generate_basic),
        ("Parameter Variations", test_generate_parameters),
        ("Invalid Parameters", test_invalid_params),
        ("Model Training", test_train),
        ("Save and Load", test_save_load),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_fail(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"{status}  {name}")
    
    print("\n" + "=" * 60)
    percentage = (passed / total * 100) if total > 0 else 0
    
    if passed == total:
        print(f"{Colors.GREEN}All tests passed! ({passed}/{total}){Colors.END}")
    else:
        print(f"{Colors.YELLOW}Tests passed: {passed}/{total} ({percentage:.1f}%){Colors.END}")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
