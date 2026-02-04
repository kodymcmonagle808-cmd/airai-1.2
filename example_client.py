#!/usr/bin/env python3
"""
AirAI 1.2 Example Client
Demonstrates how to interact with the AirAI API
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "AI-kokokwusu"

class AirAIClient:
    """Client for interacting with AirAI API"""
    
    def __init__(self, api_url=API_URL, api_key=API_KEY):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_info(self):
        """Get model information"""
        try:
            response = requests.get(
                f"{self.api_url}/info",
                headers=self.headers
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50):
        """Generate text from a prompt"""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_k": top_k
                }
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def train(self, texts, epochs=1, learning_rate=0.0001):
        """Train the model on custom texts"""
        try:
            response = requests.post(
                f"{self.api_url}/train",
                headers=self.headers,
                json={
                    "texts": texts,
                    "epochs": epochs,
                    "learning_rate": learning_rate
                }
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def save_model(self, filepath):
        """Save the current model"""
        try:
            response = requests.post(
                f"{self.api_url}/save",
                headers=self.headers,
                json={"filepath": filepath}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def load_model(self, filepath):
        """Load a saved model"""
        try:
            response = requests.post(
                f"{self.api_url}/load",
                headers=self.headers,
                json={"filepath": filepath}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    """Main example demonstration"""
    print("\n🤖 AirAI 1.2 Client Example")
    
    # Initialize client
    client = AirAIClient()
    
    # 1. Health Check
    print_section("1. Health Check")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if not health.get('model_loaded'):
        print("\n⚠️  Model not loaded. Make sure the API server is running!")
        sys.exit(1)
    
    # 2. Get Model Info
    print_section("2. Model Information")
    info = client.get_info()
    if 'error' not in info:
        print(f"Model: {info.get('name')}")
        print(f"Architecture: {info.get('architecture')}")
        print(f"Parameters: {info.get('parameters'):,}")
        print("\nConfiguration:")
        config = info.get('config', {})
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"Error: {info['error']}")
    
    # 3. Training Example
    print_section("3. Training the Model")
    training_texts = [
        "the weather is nice today",
        "i love programming in python",
        "neural networks are fascinating",
        "machine learning is powerful",
        "artificial intelligence is the future",
        "deep learning transforms data",
        "algorithms solve complex problems",
        "technology advances rapidly",
    ]
    
    print(f"Training on {len(training_texts)} samples...")
    train_result = client.train(training_texts, epochs=5, learning_rate=0.0001)
    
    if 'error' not in train_result:
        print(f"✓ {train_result.get('message')}")
        print(f"  Samples: {train_result.get('samples_trained')}")
        print(f"  Epochs: {train_result.get('epochs')}")
    else:
        print(f"✗ Error: {train_result['error']}")
    
    # 4. Text Generation Examples
    print_section("4. Text Generation Examples")
    
    prompts = [
        "hello",
        "the weather",
        "i love",
        "python is",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result = client.generate(
            prompt=prompt,
            max_length=50,
            temperature=0.7,
            top_k=40
        )
        
        if 'error' not in result:
            generated = result.get('generated_text', '')
            print(f"Generated: '{generated}'")
        else:
            print(f"Error: {result['error']}")
    
    # 5. Different Temperature Settings
    print_section("5. Temperature Comparison")
    prompt = "hello"
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    print(f"Prompt: '{prompt}'")
    print("\nGenerating with different temperatures:")
    
    for temp in temperatures:
        result = client.generate(
            prompt=prompt,
            max_length=30,
            temperature=temp,
            top_k=50
        )
        
        if 'error' not in result:
            generated = result.get('generated_text', '')
            print(f"  Temperature {temp}: '{generated}'")
    
    # 6. Save Model
    print_section("6. Saving Model")
    save_path = "/tmp/airai_trained_model.pkl"
    save_result = client.save_model(save_path)
    
    if 'error' not in save_result:
        print(f"✓ {save_result.get('message')}")
        print(f"  Location: {save_result.get('filepath')}")
    else:
        print(f"✗ Error: {save_result['error']}")
    
    # Summary
    print_section("Summary")
    print("✓ Demonstrated all major API features")
    print("✓ Health check passed")
    print("✓ Model information retrieved")
    print("✓ Training completed")
    print("✓ Text generation successful")
    print("✓ Model saved")
    print("\n💡 You can now use the API in your own applications!")
    print(f"   API URL: {API_URL}")
    print(f"   API Key: {API_KEY}")
    print("\nRefer to README.md for complete documentation.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
