import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json
from typing import List, Tuple, Dict, Optional
import os

class CustomCNN:    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.layers = []
        self.layer_configs = []
        self.weights = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model_weights(model_path)
    
    def load_model_weights(self, model_path: str):
        print(f"Loading model weights from: {model_path}")
        
        keras_model = keras.models.load_model(model_path)
        
        layer_idx = 0
        conv_idx = 0
        dense_idx = 0
        
        for layer in keras_model.layers:
            layer_config = {
                'type': layer.__class__.__name__,
                'name': layer.name,
                'config': layer.get_config()
            }
            
            if layer.get_weights():
                weights = layer.get_weights()
                if layer_config['type'] == 'Conv2D':
                    self.weights[f'conv2d_{conv_idx}'] = {
                        'kernel': weights[0],
                        'bias': weights[1] if len(weights) > 1 else None
                    }
                    conv_idx += 1
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
    
    def conv2d_forward(self, input_data: np.ndarray, kernel: np.ndarray, 
                      bias: np.ndarray = None, stride: int = 1, 
                      padding: str = 'same') -> np.ndarray:
        
        batch_size, input_height, input_width, input_channels = input_data.shape
        kernel_height, kernel_width, _, output_channels = kernel.shape
        
        if padding == 'same':
            output_height = input_height // stride
            output_width = input_width // stride
            
            pad_height = max(0, ((output_height - 1) * stride + kernel_height - input_height))
            pad_width = max(0, ((output_width - 1) * stride + kernel_width - input_width))
            
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            input_data = np.pad(input_data, 
                              ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                              mode='constant', constant_values=0)
        else:
            output_height = (input_height - kernel_height) // stride + 1
            output_width = (input_width - kernel_width) // stride + 1
        
        output = np.zeros((batch_size, output_height, output_width, output_channels))
        
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * stride
                    h_end = h_start + kernel_height
                    w_start = w * stride
                    w_end = w_start + kernel_width
                    
                    window = input_data[b, h_start:h_end, w_start:w_end, :]
                    
                    for c in range(output_channels):
                        output[b, h, w, c] = np.sum(window * kernel[:, :, :, c])
                        
                        if bias is not None:
                            output[b, h, w, c] += bias[c]
        
        return output
    
    def max_pooling2d_forward(self, input_data: np.ndarray, 
                             pool_size: Tuple[int, int] = (2, 2),
                             stride: Tuple[int, int] = None) -> np.ndarray:
        if stride is None:
            stride = pool_size
        
        batch_size, input_height, input_width, channels = input_data.shape
        pool_height, pool_width = pool_size
        stride_height, stride_width = stride
        
        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1
        
        output = np.zeros((batch_size, output_height, output_width, channels))
        
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * stride_height
                    h_end = h_start + pool_height
                    w_start = w * stride_width
                    w_end = w_start + pool_width
                    
                    window = input_data[b, h_start:h_end, w_start:w_end, :]
                    output[b, h, w, :] = np.max(window, axis=(0, 1))
        
        return output
    
    def average_pooling2d_forward(self, input_data: np.ndarray,
                                 pool_size: Tuple[int, int] = (2, 2),
                                 stride: Tuple[int, int] = None) -> np.ndarray:
        if stride is None:
            stride = pool_size
        
        batch_size, input_height, input_width, channels = input_data.shape
        pool_height, pool_width = pool_size
        stride_height, stride_width = stride
        
        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1
        
        output = np.zeros((batch_size, output_height, output_width, channels))
        
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * stride_height
                    h_end = h_start + pool_height
                    w_start = w * stride_width
                    w_end = w_start + pool_width
                    
                    window = input_data[b, h_start:h_end, w_start:w_end, :]
                    output[b, h, w, :] = np.mean(window, axis=(0, 1))
        
        return output
    
    def global_average_pooling2d_forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.mean(input_data, axis=(1, 2))
    
    def dense_forward(self, input_data: np.ndarray, kernel: np.ndarray, 
                     bias: np.ndarray = None) -> np.ndarray:
        output = np.dot(input_data, kernel)
        
        if bias is not None:
            output += bias
        
        return output
    
    def relu_activation(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)
    
    def softmax_activation(self, input_data: np.ndarray) -> np.ndarray:
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if not self.layer_configs:
            raise ValueError("No model loaded. Please load a model first.")
        
        x = input_data.copy()
        conv_layer_idx = 0
        dense_layer_idx = 0
        
        print(f"Input shape: {x.shape}")
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            config = layer_config['config']
            
            print(f"Layer {i}: {layer_type}")
            
            if layer_type == 'Conv2D':
                weights = self.weights[f'conv2d_{conv_layer_idx}']
                kernel = weights['kernel']
                bias = weights['bias']
                
                strides = config.get('strides', [1, 1])
                padding = config.get('padding', 'same')
                activation = config.get('activation', 'linear')
                
                x = self.conv2d_forward(x, kernel, bias, stride=strides[0], padding=padding)
                
                if activation == 'relu':
                    x = self.relu_activation(x)
                
                conv_layer_idx += 1
                
            elif layer_type == 'MaxPooling2D':
                pool_size = config.get('pool_size', [2, 2])
                strides = config.get('strides', pool_size)
                x = self.max_pooling2d_forward(x, tuple(pool_size), tuple(strides))
                
            elif layer_type == 'AveragePooling2D':
                pool_size = config.get('pool_size', [2, 2])
                strides = config.get('strides', pool_size)
                x = self.average_pooling2d_forward(x, tuple(pool_size), tuple(strides))
                
            elif layer_type == 'GlobalAveragePooling2D':
                x = self.global_average_pooling2d_forward(x)
            
            elif layer_type == 'Flatten':
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)
            
            elif layer_type == 'Dense':
                weights = self.weights[f'dense_{dense_layer_idx}']
                kernel = weights['kernel']
                bias = weights['bias']
                
                activation = config.get('activation', 'linear')
                
                x = self.dense_forward(x, kernel, bias)
                
                if activation == 'relu':
                    x = self.relu_activation(x)
                elif activation == 'softmax':
                    x = self.softmax_activation(x)
                
                dense_layer_idx += 1
            
            elif layer_type == 'Dropout':
                pass
            
            print(f"Output shape: {x.shape}")
        
        return x
    
    def predict(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        num_samples = input_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        predictions = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = input_data[start_idx:end_idx]
            batch_predictions = self.forward(batch_data)
            
            predictions.append(batch_predictions)
        
        return np.vstack(predictions)
    
    def get_model_summary(self) -> str:
        if not self.layer_configs:
            return "No model loaded"
        
        summary = "Custom CNN Model Summary:\n"
        summary += "=" * 50 + "\n"
        
        for i, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config['type']
            layer_name = layer_config['name']
            summary += f"Layer {i}: {layer_type} ({layer_name})\n"
            
            if layer_type == 'Conv2D':
                config = layer_config['config']
                filters = config.get('filters', 'N/A')
                kernel_size = config.get('kernel_size', 'N/A')
                activation = config.get('activation', 'linear')
                summary += f"  Filters: {filters}, Kernel: {kernel_size}, Activation: {activation}\n"
            
            elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
                config = layer_config['config']
                pool_size = config.get('pool_size', 'N/A')
                summary += f"  Pool size: {pool_size}\n"
            
            elif layer_type == 'Dense':
                config = layer_config['config']
                units = config.get('units', 'N/A')
                activation = config.get('activation', 'linear')
                summary += f"  Units: {units}, Activation: {activation}\n"
        
        summary += "=" * 50 + "\n"
        return summary

if __name__ == "__main__":
    model_path = "models/cnn_layers_3.h5"
    
    if os.path.exists(model_path):
        custom_cnn = CustomCNN(model_path)
        
        print(custom_cnn.get_model_summary())
        
        dummy_input = np.random.random((10, 32, 32, 3)).astype(np.float32)
        
        predictions = custom_cnn.predict(dummy_input)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample prediction: {predictions[0]}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using the pretrained model scripts.")