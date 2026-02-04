# AirAI 1.2

A complete neural network AI system built entirely from scratch - no external API dependencies, no pre-trained models, just pure neural network architecture implemented in Python with NumPy.

## 🧠 Features

- **Transformer-based Architecture** - Multi-head attention, feed-forward networks, layer normalization
- **Built from Scratch** - All neural network components implemented without external AI libraries
- **REST API Server** - Complete HTTP API with authentication
- **Training Capabilities** - Train on custom datasets
- **Text Generation** - Generate text with configurable temperature and top-k sampling
- **Model Persistence** - Save and load trained models
- **No External Dependencies** - Only requires NumPy and Python standard library

## 📋 Architecture Details

- **Model Type**: Transformer-based Language Model
- **Components**:
  - Token embeddings (learnable)
  - Positional encodings (sinusoidal)
  - Multi-head self-attention layers
  - Position-wise feed-forward networks
  - Layer normalization
  - Output projection layer
- **Configurable Parameters**:
  - Vocabulary size
  - Embedding dimensions
  - Hidden dimensions
  - Number of layers
  - Number of attention heads
  - Maximum sequence length

## 🔑 Authentication

All API requests require authentication using the API key: **AI-kokokwusu**

Include the API key in request headers as:
- Authorization: Bearer AI-kokokwusu
- OR API-Key: AI-kokokwusu

## 📡 API Endpoints

### GET /

Get API documentation and available endpoints.

**Response:**
```json
{
  "name": "AirAI 1.2 API",
  "version": "1.2.0",
  "description": "Complete neural network AI system built from scratch",
  "endpoints": { ... }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.2.0"
}
```

### GET /info

Get model information (requires authentication).

**Headers:**
```
Authorization: Bearer AI-kokokwusu
```

**Response:**
```json
{
  "name": "AirAI 1.2",
  "architecture": "Transformer-based Neural Network",
  "parameters": 15728640,
  "config": {
    "vocab_size": 5000,
    "embedding_dim": 256,
    "hidden_dim": 1024,
    "num_layers": 4,
    "num_heads": 8,
    "max_seq_length": 256
  },
  "status": "ready"
}
```

### POST /generate

Generate text from a prompt (requires authentication).

**Headers:**
```
Authorization: Bearer AI-kokokwusu
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "hello how are",
  "max_length": 100,
  "temperature": 0.8,
  "top_k": 50
}
```

**Parameters:**
- `prompt` (required): Input text to generate from
- `max_length` (optional): Maximum tokens to generate (default: 100, max: 1000)
- `temperature` (optional): Sampling temperature (default: 0.8, higher = more random)
- `top_k` (optional): Top-k sampling parameter (default: 50)

**Response:**
```json
{
  "status": "success",
  "prompt": "hello how are",
  "generated_text": "hello how are you doing today",
  "parameters": {
    "max_length": 100,
    "temperature": 0.8,
    "top_k": 50
  }
}
```

### POST /train

Train the model on custom data (requires authentication).

**Headers:**
```
Authorization: Bearer AI-kokokwusu
Content-Type: application/json
```

**Request Body:**
```json
{
  "texts": [
    "hello world",
    "how are you",
    "i am fine thank you"
  ],
  "epochs": 5,
  "learning_rate": 0.0001
}
```

**Parameters:**
- `texts` (required): Array of training texts
- `epochs` (optional): Number of training epochs (default: 1)
- `learning_rate` (optional): Learning rate for training (default: 0.0001)

**Response:**
```json
{
  "status": "success",
  "message": "Training completed",
  "samples_trained": 3,
  "epochs": 5
}
```

### POST /save

Save the current model state (requires authentication).

**Headers:**
```
Authorization: Bearer AI-kokokwusu
Content-Type: application/json
```

**Request Body:**
```json
{
  "filepath": "/path/to/save/model.pkl"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model saved successfully",
  "filepath": "/path/to/save/model.pkl"
}
```

### POST /load

Load a previously saved model (requires authentication).

**Headers:**
```
Authorization: Bearer AI-kokokwusu
Content-Type: application/json
```

**Request Body:**
```json
{
  "filepath": "/path/to/model.pkl"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "filepath": "/path/to/model.pkl",
  "parameters": 15728640
}
```

## 🏗️ Direct Usage

The brain can be used directly without the API server. See example_client.py for complete examples.

## 🔧 Configuration

Server runs on 0.0.0.0:8000 by default.

Model configuration is adjustable in the code with parameters for:
- vocab_size
- embedding_dim
- hidden_dim
- num_layers
- num_heads
- max_seq_length

## 📊 Performance Notes

- **Parameter Count**: Default configuration has ~15.7M parameters
- **Memory**: Requires ~200MB RAM for default model
- **Speed**: Generation speed depends on sequence length and hardware
- **Training**: Basic implementation for demonstration; production would need optimized backpropagation

## 🔒 Security Notes

- The API key `AI-kokokwusu` is hardcoded for simplicity
- For production use, implement proper key management
- Consider adding rate limiting for production deployments
- Use HTTPS in production environments

## 📝 Technical Implementation Details

### Neural Network Components

1. **Embeddings**: Token and positional embeddings convert input tokens to vectors
2. **Self-Attention**: Multi-head attention mechanism to capture dependencies
3. **Feed-Forward**: Position-wise fully connected layers with GELU activation
4. **Layer Normalization**: Stabilizes training and improves convergence
5. **Residual Connections**: Helps with gradient flow in deep networks

### Training Process

- Uses cross-entropy loss
- Implements simplified gradient descent (full backpropagation recommended for production)
- Supports custom learning rates and epochs
- Trains on sequential prediction tasks

### Generation Strategy

- Uses causal masking to prevent looking ahead
- Top-k sampling for diverse outputs
- Temperature scaling for controlling randomness
- Supports various generation parameters

## 🤝 Contributing

Contributions are welcome! This is a educational/demonstration project showing how to build an AI from scratch.

## 📄 License

MIT License - feel free to use and modify as needed.

## 🎯 Use Cases

- Learning how neural networks work internally
- Building custom AI applications
- Educational purposes
- Experimentation with model architectures
- Lightweight AI inference

## ⚠️ Limitations

- Not pre-trained on large datasets (starts from scratch)
- Basic training implementation (simplified backpropagation)
- Limited vocabulary by default
- Not optimized for production-scale applications
- Requires training on your specific use case

## 🚀 Future Improvements

- Implement full backpropagation through time
- Add more sophisticated tokenization
- Support for fine-tuning
- Batch processing
- GPU acceleration support
- Model quantization
- Additional architectures (LSTM, GRU)

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**AirAI 1.2** - Built from scratch with ❤️ and NumPy
