import numpy as np
import json
import pickle
from typing import List, Dict, Tuple, Optional
import math

class AirAIBrain:
    """
    AirAI 1.2 - A complete neural network AI system built from scratch
    No external API dependencies or pre-trained models
    """
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 512, 
                 hidden_dim: int = 2048, num_layers: int = 6, num_heads: int = 8,
                 max_seq_length: int = 512):
        """Initialize the AirAI neural network architecture"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        # Initialize all weights and biases
        self._initialize_weights()
        
        # Tokenizer vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_default_vocab()
        
        print(f"AirAI 1.2 initialized with {self._count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize all neural network weights"""
        np.random.seed(42)
        
        # Token embeddings
        self.token_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.02
        
        # Positional embeddings
        self.positional_embeddings = self._create_positional_encoding()
        
        # Transformer layers
        self.layers = []
        for _ in range(self.num_layers):
            layer = {
                # Multi-head attention
                'q_weight': np.random.randn(self.embedding_dim, self.embedding_dim) * 0.02,
                'k_weight': np.random.randn(self.embedding_dim, self.embedding_dim) * 0.02,
                'v_weight': np.random.randn(self.embedding_dim, self.embedding_dim) * 0.02,
                'o_weight': np.random.randn(self.embedding_dim, self.embedding_dim) * 0.02,
                
                # Feed-forward network
                'ff1_weight': np.random.randn(self.embedding_dim, self.hidden_dim) * 0.02,
                'ff1_bias': np.zeros(self.hidden_dim),
                'ff2_weight': np.random.randn(self.hidden_dim, self.embedding_dim) * 0.02,
                'ff2_bias': np.zeros(self.embedding_dim),
                
                # Layer normalization
                'ln1_gamma': np.ones(self.embedding_dim),
                'ln1_beta': np.zeros(self.embedding_dim),
                'ln2_gamma': np.ones(self.embedding_dim),
                'ln2_beta': np.zeros(self.embedding_dim),
            }
            self.layers.append(layer)
        
        # Output layer
        self.output_weight = np.random.randn(self.embedding_dim, self.vocab_size) * 0.02
        self.output_bias = np.zeros(self.vocab_size)
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        position = np.arange(self.max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(np.log(10000.0) / self.embedding_dim))
        
        pos_encoding = np.zeros((self.max_seq_length, self.embedding_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def _build_default_vocab(self):
        """Build a default vocabulary with common tokens"""
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>']
        
        # Common characters and tokens
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'-\n")
        
        # Common words
        common_words = [
            "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "may", "might", "must",
            "i", "you", "he", "she", "it", "we", "they",
            "this", "that", "these", "those", "what", "which", "who",
            "and", "or", "but", "if", "because", "when", "where", "how",
            "not", "no", "yes", "hello", "hi", "thanks", "please",
            "what", "why", "how", "when", "where", "who",
        ]
        
        vocab = special_tokens + chars + common_words
        
        # Add more tokens to reach vocab size
        for i in range(len(vocab), min(self.vocab_size, 1000)):
            vocab.append(f"<TOKEN_{i}>")
        
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = []
        text = text.lower().strip()
        
        i = 0
        while i < len(text):
            # Try to match longest word first
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token_to_id:
                    tokens.append(self.token_to_id[substr])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Use character or unknown token
                char = text[i]
                tokens.append(self.token_to_id.get(char, self.token_to_id.get('<UNK>', 1)))
                i += 1
        
        return tokens[:self.max_seq_length]
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        text = ''.join(tokens)
        # Clean up special tokens
        for special in ['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>']:
            text = text.replace(special, '')
        return text.strip()
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                    eps: float = 1e-5) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, 
                                     v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Scaled dot-product attention mechanism"""
        d_k = q.shape[-1]
        scores = np.matmul(q, k.T) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        attention_weights = self._softmax(scores, axis=-1)
        return np.matmul(attention_weights, v)
    
    def _multi_head_attention(self, x: np.ndarray, layer: Dict, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Multi-head attention mechanism"""
        seq_len = x.shape[0]
        head_dim = self.embedding_dim // self.num_heads
        
        # Linear projections
        q = np.matmul(x, layer['q_weight'])
        k = np.matmul(x, layer['k_weight'])
        v = np.matmul(x, layer['v_weight'])
        
        # Reshape for multi-head attention
        q = q.reshape(seq_len, self.num_heads, head_dim).swapaxes(0, 1)
        k = k.reshape(seq_len, self.num_heads, head_dim).swapaxes(0, 1)
        v = v.reshape(seq_len, self.num_heads, head_dim).swapaxes(0, 1)
        
        # Apply attention for each head
        attn_outputs = []
        for i in range(self.num_heads):
            head_mask = mask if mask is not None else None
            attn = self._scaled_dot_product_attention(
                q[i], k[i], v[i], head_mask
            )
            attn_outputs.append(attn)
        
        # Concatenate heads
        attn_output = np.concatenate(attn_outputs, axis=-1)
        
        # Output projection
        return np.matmul(attn_output, layer['o_weight'])
    
    def _feed_forward(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """Position-wise feed-forward network"""
        hidden = self._gelu(np.matmul(x, layer['ff1_weight']) + layer['ff1_bias'])
        return np.matmul(hidden, layer['ff2_weight']) + layer['ff2_bias']
    
    def forward(self, token_ids: List[int], temperature: float = 1.0) -> np.ndarray:
        """Forward pass through the network"""
        seq_len = len(token_ids)
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        # Embedding layer
        x = self.token_embeddings[token_ids] + self.positional_embeddings[:seq_len]
        
        # Transformer layers
        for layer in self.layers:
            # Multi-head attention with residual connection
            attn_output = self._multi_head_attention(x, layer, mask)
            x = x + attn_output
            x = self._layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            
            # Feed-forward with residual connection
            ff_output = self._feed_forward(x, layer)
            x = x + ff_output
            x = self._layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
        
        # Output projection
        logits = np.matmul(x[-1], self.output_weight) + self.output_bias
        logits = logits / temperature
        
        return self._softmax(logits)
    
    def generate(self, prompt: str, max_length: int = 100, 
                temperature: float = 0.8, top_k: int = 50) -> str:
        """Generate text based on a prompt"""
        token_ids = self.tokenize(prompt)
        
        for _ in range(max_length):
            # Get predictions
            probs = self.forward(token_ids, temperature)
            
            # Top-k sampling
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            
            # Sample next token
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
            token_ids.append(next_token)
            
            # Stop if we hit end token or max length
            if next_token == self.token_to_id.get('<END>', -1):
                break
            
            if len(token_ids) >= self.max_seq_length:
                break
        
        return self.detokenize(token_ids)
    
    def train_step(self, input_ids: List[int], target_ids: List[int], 
                  learning_rate: float = 0.0001) -> float:
        """Single training step with backpropagation"""
        # Forward pass
        probs = self.forward(input_ids)
        
        # Compute loss (cross-entropy)
        target_id = target_ids[-1] if target_ids else 0
        loss = -np.log(probs[target_id] + 1e-10)
        
        # Simplified gradient update (for demonstration)
        # In production, you'd implement full backpropagation
        grad = probs.copy()
        grad[target_id] -= 1
        
        # Update output layer
        x_last = self.token_embeddings[input_ids[-1]] if input_ids else np.zeros(self.embedding_dim)
        self.output_weight -= learning_rate * np.outer(x_last, grad)
        self.output_bias -= learning_rate * grad
        
        return float(loss)
    
    def train(self, texts: List[str], epochs: int = 1, 
             learning_rate: float = 0.0001, verbose: bool = True):
        """Train the model on a list of texts"""
        total_loss = 0
        num_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for text in texts:
                token_ids = self.tokenize(text)
                
                # Train on sequences
                for i in range(1, len(token_ids)):
                    input_ids = token_ids[:i]
                    target_ids = token_ids[:i+1]
                    loss = self.train_step(input_ids, target_ids, learning_rate)
                    epoch_loss += loss
                    num_samples += 1
            
            avg_loss = epoch_loss / max(num_samples, 1)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def save(self, filepath: str):
        """Save the model to disk"""
        model_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'max_seq_length': self.max_seq_length,
            },
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_embeddings': self.token_embeddings,
            'positional_embeddings': self.positional_embeddings,
            'layers': self.layers,
            'output_weight': self.output_weight,
            'output_bias': self.output_bias,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        model = cls(**config)
        
        model.token_to_id = model_data['token_to_id']
        model.id_to_token = model_data['id_to_token']
        model.token_embeddings = model_data['token_embeddings']
        model.positional_embeddings = model_data['positional_embeddings']
        model.layers = model_data['layers']
        model.output_weight = model_data['output_weight']
        model.output_bias = model_data['output_bias']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        count = self.token_embeddings.size + self.positional_embeddings.size
        
        for layer in self.layers:
            for key, value in layer.items():
                if isinstance(value, np.ndarray):
                    count += value.size
        
        count += self.output_weight.size + self.output_bias.size
        return count


if __name__ == "__main__":
    # Example usage
    print("Initializing AirAI 1.2...")
    brain = AirAIBrain(
        vocab_size=5000,
        embedding_dim=256,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8
    )
    
    # Example training data
    training_texts = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is AirAI.",
        "Nice to meet you!",
    ]
    
    print("\nTraining AirAI...")
    brain.train(training_texts, epochs=10, verbose=True)
    
    # Generate text
    print("\n--- Testing Generation ---")
    prompt = "hello"
    print(f"Prompt: {prompt}")
    generated = brain.generate(prompt, max_length=50)
    print(f"Generated: {generated}")
    
    # Save model
    print("\nSaving model...")
    brain.save("/home/claude/airai_model.pkl")
    
    print("\nAirAI 1.2 Brain is ready!")
