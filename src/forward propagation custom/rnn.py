import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json
from typing import List, Tuple, Dict, Optional
import os

class CustomRNN:
    def __init__(self, model_path: str = None):

        self.model_path = model_path
        self.layer_configs = []
        self.weights = {}
        self.vocab_size = 10000
        self.max_length = 100
        self.embedding_dim = 128
        
        if model_path and os.path.exists(model_path):
            self.load_model_weights(model_path)
    
    def load_model_weights(self, model_path: str):

        print(f"Loading RNN model weights from: {model_path}")
        
        keras_model = keras.models.load_model(model_path)
        
        layer_idx = 0
        embedding_idx = 0
        rnn_idx = 0
        dense_idx = 0
        
        for layer in keras_model.layers:
            layer_config = {
                'type': layer.__class__.__name__,
                'name': layer.name,
                'config': layer.get_config()
            }
            
            if layer.get_weights():
                weights = layer.get_weights()
                
                if layer_config['type'] == 'Embedding':
                    self.weights[f'embedding_{embedding_idx}'] = {
                        'embeddings': weights[0]
                    }
                    embedding_idx += 1
                    
                elif layer_config['type'] == 'SimpleRNN':
                    self.weights[f'simple_rnn_{rnn_idx}'] = {
                        'kernel': weights[0],          
                        'recurrent_kernel': weights[1],
                        'bias': weights[2] if len(weights) > 2 else None
                    }
                    rnn_idx += 1
                    
                elif layer_config['type'] == 'Bidirectional':
                    forward_weights = weights[:len(weights)//2]
                    backward_weights = weights[len(weights)//2:]
                    
                    self.weights[f'bidirectional_rnn_{rnn_idx}'] = {
                        'forward': {
                            'kernel': forward_weights[0],
                            'recurrent_kernel': forward_weights[1],
                            'bias': forward_weights[2] if len(forward_weights) > 2 else None
                        },
                        'backward': {
                            'kernel': backward_weights[0],
                            'recurrent_kernel': backward_weights[1],
                            'bias': backward_weights[2] if len(backward_weights) > 2 else None
                        }
                    }
                    rnn_idx += 1
                    
                elif layer_config['type'] == 'Dense':
                    self.weights[f'dense_{dense_idx}'] = {
                        'kernel': weights[0],
                        'bias': weights[1] if len(weights) > 1 else None
                    }
                    dense_idx += 1
            
            self.layer_configs.append(layer_config)
            layer_idx += 1
        
        print(f"Loaded {len(self.layer_configs)} layers")
        print(f"Available weights: {list(self.weights.keys())}")
    
    def embedding_forward(self, input_sequences: np.ndarray, 
                         embeddings: np.ndarray) -> np.ndarray:
        batch_size, sequence_length = input_sequences.shape
        embedding_dim = embeddings.shape[1]
        
        output = np.zeros((batch_size, sequence_length, embedding_dim))
        
        for b in range(batch_size):
            for t in range(sequence_length):
                token_id = int(input_sequences[b, t])
                if 0 <= token_id < embeddings.shape[0]:
                    output[b, t, :] = embeddings[token_id]
        
        return output
    
    def simple_rnn_cell_forward(self, input_t: np.ndarray, hidden_prev: np.ndarray,
                               kernel: np.ndarray, recurrent_kernel: np.ndarray,
                               bias: np.ndarray = None) -> np.ndarray:
        hidden_new = np.dot(input_t, kernel) + np.dot(hidden_prev, recurrent_kernel)
        
        if bias is not None:
            hidden_new += bias
        
        hidden_new = np.tanh(hidden_new)
        
        return hidden_new
    
    def simple_rnn_forward(self, input_sequences: np.ndarray, kernel: np.ndarray,
                          recurrent_kernel: np.ndarray, bias: np.ndarray = None,
                          return_sequences: bool = False) -> np.ndarray:
        batch_size, sequence_length, input_dim = input_sequences.shape
        units = kernel.shape[1]
        
        hidden = np.zeros((batch_size, units))
        
        if return_sequences:
            all_hidden = np.zeros((batch_size, sequence_length, units))
        
        for t in range(sequence_length):
            input_t = input_sequences[:, t, :]
            hidden = self.simple_rnn_cell_forward(input_t, hidden, kernel, 
                                                 recurrent_kernel, bias)
            
            if return_sequences:
                all_hidden[:, t, :] = hidden
        
        if return_sequences:
            return all_hidden
        else:
            return hidden
    
    def bidirectional_rnn_forward(self, input_sequences: np.ndarray,
                                 forward_weights: Dict, backward_weights: Dict,
                                 return_sequences: bool = False) -> np.ndarray:
        batch_size, sequence_length, input_dim = input_sequences.shape
        
        forward_output = self.simple_rnn_forward(
            input_sequences,
            forward_weights['kernel'],
            forward_weights['recurrent_kernel'],
            forward_weights['bias'],
            return_sequences=return_sequences
        )
        
        reversed_sequences = input_sequences[:, ::-1, :]  
        backward_output = self.simple_rnn_forward(
            reversed_sequences,
            backward_weights['kernel'],
            backward_weights['recurrent_kernel'],
            backward_weights['bias'],
            return_sequences=return_sequences
        )
        
        if return_sequences:
            backward_output = backward_output[:, ::-1, :]
            output = np.concatenate([forward_output, backward_output], axis=2)
        else:
            output = np.concatenate([forward_output, backward_output], axis=1)
        
        return output
    
    def dense_forward(self, input_data: np.ndarray, kernel: np.ndarray,
                     bias: np.ndarray = None) -> np.ndarray:
        output = np.dot(input_data, kernel)
        
        if bias is not None:
            output += bias
        
        return output
    
    def softmax_activation(self, input_data: np.ndarray) -> np.ndarray:
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def forward(self, input_sequences: np.ndarray) -> np.ndarray:
        if not self.layer_configs:
            raise ValueError("No model loaded. Please load a model first.")
        
        x = input_sequences.copy()
        embedding_idx = 0
        rnn_idx = 0
        dense_idx = 0
        
        print(f"Input shape: {x.shape}")
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            config = layer_config['config']
            
            print(f"Layer {i}: {layer_type}")
            
            if layer_type == 'Embedding':
                weights = self.weights[f'embedding_{embedding_idx}']
                embeddings = weights['embeddings']
                
                x = self.embedding_forward(x, embeddings)
                embedding_idx += 1
                
            elif layer_type == 'SimpleRNN':
                weights = self.weights[f'simple_rnn_{rnn_idx}']
                kernel = weights['kernel']
                recurrent_kernel = weights['recurrent_kernel']
                bias = weights['bias']
                
                return_sequences = config.get('return_sequences', False)
                
                x = self.simple_rnn_forward(x, kernel, recurrent_kernel, bias, 
                                          return_sequences)
                rnn_idx += 1
                
            elif layer_type == 'Bidirectional':
                weights = self.weights[f'bidirectional_rnn_{rnn_idx}']
                forward_weights = weights['forward']
                backward_weights = weights['backward']
                
                wrapped_layer = config.get('layer', {})
                return_sequences = wrapped_layer.get('config', {}).get('return_sequences', False)
                
                x = self.bidirectional_rnn_forward(x, forward_weights, backward_weights,
                                                 return_sequences)
                rnn_idx += 1
                
            elif layer_type == 'Dense':
                weights = self.weights[f'dense_{dense_idx}']
                kernel = weights['kernel']
                bias = weights['bias']
                
                activation = config.get('activation', 'linear')
                
                x = self.dense_forward(x, kernel, bias)
                
                if activation == 'softmax':
                    x = self.softmax_activation(x)
                
                dense_idx += 1
                
            elif layer_type == 'Dropout':
                pass
            
            print(f"Output shape: {x.shape}")
        
        return x
    
    def predict(self, input_sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        num_samples = input_sequences.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        predictions = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = input_sequences[start_idx:end_idx]
            batch_predictions = self.forward(batch_data)
            
            predictions.append(batch_predictions)
        
        return np.vstack(predictions)
    
    def get_model_summary(self) -> str:
        if not self.layer_configs:
            return "No model loaded"
        
        summary = "Custom RNN Model Summary:\n"
        summary += "=" * 50 + "\n"
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            layer_name = layer_config['name']
            summary += f"Layer {i}: {layer_type} ({layer_name})\n"
            
            if layer_type == 'Embedding':
                config = layer_config['config']
                input_dim = config.get('input_dim', 'N/A')
                output_dim = config.get('output_dim', 'N/A')
                summary += f"  Vocab size: {input_dim}, Embedding dim: {output_dim}\n"
            
            elif layer_type == 'SimpleRNN':
                config = layer_config['config']
                units = config.get('units', 'N/A')
                return_sequences = config.get('return_sequences', False)
                summary += f"  Units: {units}, Return sequences: {return_sequences}\n"
            
            elif layer_type == 'Bidirectional':
                config = layer_config['config']
                wrapped_layer = config.get('layer', {})
                units = wrapped_layer.get('config', {}).get('units', 'N/A')
                summary += f"  Bidirectional RNN, Units: {units}\n"
            
            elif layer_type == 'Dense':
                config = layer_config['config']
                units = config.get('units', 'N/A')
                activation = config.get('activation', 'linear')
                summary += f"  Units: {units}, Activation: {activation}\n"
        
        summary += "=" * 50 + "\n"
        return summary

if __name__ == "__main__":
    model_path = "models/rnn_layers_2.h5"
    
    if os.path.exists(model_path):
        custom_rnn = CustomRNN(model_path)
        
        print(custom_rnn.get_model_summary())
        
        dummy_input = np.random.randint(0, 1000, size=(10, 100)).astype(np.int32)
        
        predictions = custom_rnn.predict(dummy_input)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample prediction: {predictions[0]}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using the pretrained model scripts.")