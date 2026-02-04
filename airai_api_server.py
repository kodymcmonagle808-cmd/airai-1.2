from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import os
from urllib.parse import parse_qs, urlparse
import traceback

# Import the AirAI brain
sys.path.append(os.path.dirname(__file__))
from airai_brain import AirAIBrain

# API Key for authentication
VALID_API_KEY = "AI-kokokwusu"

# Global model instance
ai_model = None

class AirAIAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for AirAI API"""
    
    def _set_headers(self, status=200, content_type='application/json'):
        """Set response headers"""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def _authenticate(self):
        """Verify API key authentication"""
        auth_header = self.headers.get('Authorization', '')
        
        # Check for Bearer token
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return token == VALID_API_KEY
        
        # Check for API-Key header
        api_key = self.headers.get('API-Key', '')
        if api_key == VALID_API_KEY:
            return True
        
        return False
    
    def _send_json_response(self, data, status=200):
        """Send JSON response"""
        self._set_headers(status)
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error_response(self, message, status=400):
        """Send error response"""
        self._send_json_response({
            'error': message,
            'status': 'error'
        }, status)
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            if path == '/':
                # API documentation
                docs = {
                    'name': 'AirAI 1.2 API',
                    'version': '1.2.0',
                    'description': 'Complete neural network AI system built from scratch',
                    'authentication': 'Required - Use API key: AI-kokokwusu',
                    'endpoints': {
                        'GET /': 'API documentation',
                        'GET /health': 'Health check',
                        'GET /info': 'Model information',
                        'POST /generate': 'Generate text from prompt',
                        'POST /train': 'Train the model with custom data',
                        'POST /save': 'Save the current model',
                        'POST /load': 'Load a saved model',
                    },
                    'usage': {
                        'headers': {
                            'Authorization': 'Bearer AI-kokokwusu',
                            'OR': 'API-Key: AI-kokokwusu',
                            'Content-Type': 'application/json'
                        }
                    }
                }
                self._send_json_response(docs)
            
            elif path == '/health':
                # Health check endpoint
                self._send_json_response({
                    'status': 'healthy',
                    'model_loaded': ai_model is not None,
                    'version': '1.2.0'
                })
            
            elif path == '/info':
                # Authentication required
                if not self._authenticate():
                    self._send_error_response('Unauthorized - Invalid API key', 401)
                    return
                
                if ai_model is None:
                    self._send_error_response('Model not initialized', 500)
                    return
                
                # Model information
                info = {
                    'name': 'AirAI 1.2',
                    'architecture': 'Transformer-based Neural Network',
                    'parameters': ai_model._count_parameters(),
                    'config': {
                        'vocab_size': ai_model.vocab_size,
                        'embedding_dim': ai_model.embedding_dim,
                        'hidden_dim': ai_model.hidden_dim,
                        'num_layers': ai_model.num_layers,
                        'num_heads': ai_model.num_heads,
                        'max_seq_length': ai_model.max_seq_length,
                    },
                    'status': 'ready'
                }
                self._send_json_response(info)
            
            else:
                self._send_error_response(f'Endpoint not found: {path}', 404)
        
        except Exception as e:
            self._send_error_response(f'Server error: {str(e)}', 500)
            traceback.print_exc()
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Authentication required for all POST endpoints
            if not self._authenticate():
                self._send_error_response('Unauthorized - Invalid API key', 401)
                return
            
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_error_response('Invalid JSON in request body', 400)
                return
            
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            if path == '/generate':
                self._handle_generate(data)
            
            elif path == '/train':
                self._handle_train(data)
            
            elif path == '/save':
                self._handle_save(data)
            
            elif path == '/load':
                self._handle_load(data)
            
            else:
                self._send_error_response(f'Endpoint not found: {path}', 404)
        
        except Exception as e:
            self._send_error_response(f'Server error: {str(e)}', 500)
            traceback.print_exc()
    
    def _handle_generate(self, data):
        """Handle text generation request"""
        global ai_model
        
        if ai_model is None:
            self._send_error_response('Model not initialized', 500)
            return
        
        # Get parameters
        prompt = data.get('prompt', '')
        if not prompt:
            self._send_error_response('Missing required parameter: prompt', 400)
            return
        
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        
        # Validate parameters
        if not isinstance(max_length, int) or max_length < 1 or max_length > 1000:
            self._send_error_response('max_length must be between 1 and 1000', 400)
            return
        
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            self._send_error_response('temperature must be positive', 400)
            return
        
        if not isinstance(top_k, int) or top_k < 1:
            self._send_error_response('top_k must be positive integer', 400)
            return
        
        # Generate text
        try:
            generated_text = ai_model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            response = {
                'status': 'success',
                'prompt': prompt,
                'generated_text': generated_text,
                'parameters': {
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_k': top_k
                }
            }
            self._send_json_response(response)
        
        except Exception as e:
            self._send_error_response(f'Generation failed: {str(e)}', 500)
    
    def _handle_train(self, data):
        """Handle training request"""
        global ai_model
        
        if ai_model is None:
            self._send_error_response('Model not initialized', 500)
            return
        
        # Get parameters
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            self._send_error_response('Missing or invalid parameter: texts (must be array)', 400)
            return
        
        epochs = data.get('epochs', 1)
        learning_rate = data.get('learning_rate', 0.0001)
        
        # Validate parameters
        if not isinstance(epochs, int) or epochs < 1:
            self._send_error_response('epochs must be positive integer', 400)
            return
        
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            self._send_error_response('learning_rate must be positive', 400)
            return
        
        # Train model
        try:
            ai_model.train(
                texts=texts,
                epochs=epochs,
                learning_rate=learning_rate,
                verbose=False
            )
            
            response = {
                'status': 'success',
                'message': 'Training completed',
                'samples_trained': len(texts),
                'epochs': epochs
            }
            self._send_json_response(response)
        
        except Exception as e:
            self._send_error_response(f'Training failed: {str(e)}', 500)
    
    def _handle_save(self, data):
        """Handle model save request"""
        global ai_model
        
        if ai_model is None:
            self._send_error_response('Model not initialized', 500)
            return
        
        filepath = data.get('filepath', '/home/claude/airai_model.pkl')
        
        try:
            ai_model.save(filepath)
            response = {
                'status': 'success',
                'message': 'Model saved successfully',
                'filepath': filepath
            }
            self._send_json_response(response)
        
        except Exception as e:
            self._send_error_response(f'Save failed: {str(e)}', 500)
    
    def _handle_load(self, data):
        """Handle model load request"""
        global ai_model
        
        filepath = data.get('filepath', '/home/claude/airai_model.pkl')
        
        if not os.path.exists(filepath):
            self._send_error_response(f'Model file not found: {filepath}', 404)
            return
        
        try:
            ai_model = AirAIBrain.load(filepath)
            response = {
                'status': 'success',
                'message': 'Model loaded successfully',
                'filepath': filepath,
                'parameters': ai_model._count_parameters()
            }
            self._send_json_response(response)
        
        except Exception as e:
            self._send_error_response(f'Load failed: {str(e)}', 500)
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(host='0.0.0.0', port=8000):
    """Run the AirAI API server"""
    global ai_model
    
    print("=" * 60)
    print("AirAI 1.2 - Neural Network API Server")
    print("=" * 60)
    print("\nInitializing AirAI Brain...")
    
    # Initialize the model
    try:
        ai_model = AirAIBrain(
            vocab_size=5000,
            embedding_dim=256,
            hidden_dim=1024,
            num_layers=4,
            num_heads=8,
            max_seq_length=256
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return
    
    # Initial training with basic conversational data
    print("\nPerforming initial training...")
    initial_training_data = [
        "hello how are you",
        "i am doing well thank you",
        "what is your name",
        "my name is airai",
        "nice to meet you",
        "what can you do",
        "i can help you with various tasks",
        "that sounds great",
        "how does this work",
        "i use neural networks to process information",
    ]
    
    try:
        ai_model.train(initial_training_data, epochs=5, verbose=False)
        print("✓ Initial training completed")
    except Exception as e:
        print(f"✗ Training warning: {e}")
    
    print("\n" + "=" * 60)
    print("API INFORMATION")
    print("=" * 60)
    print(f"Server Address: http://{host}:{port}")
    print(f"API Key: {VALID_API_KEY}")
    print("\nAuthentication:")
    print(f"  Header: Authorization: Bearer {VALID_API_KEY}")
    print(f"  OR")
    print(f"  Header: API-Key: {VALID_API_KEY}")
    print("\nEndpoints:")
    print("  GET  /          - API documentation")
    print("  GET  /health    - Health check")
    print("  GET  /info      - Model information")
    print("  POST /generate  - Generate text")
    print("  POST /train     - Train model")
    print("  POST /save      - Save model")
    print("  POST /load      - Load model")
    print("=" * 60)
    
    # Start server
    try:
        server = HTTPServer((host, port), AirAIAPIHandler)
        print(f"\n🚀 Server started successfully!")
        print(f"🔗 Listening on {host}:{port}")
        print("\nPress Ctrl+C to stop the server\n")
        server.serve_forever()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        server.shutdown()
        print("✓ Server stopped")
    
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    run_server(host, port)
