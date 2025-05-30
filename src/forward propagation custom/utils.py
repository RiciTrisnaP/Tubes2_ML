import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import pickle

class ModelWeightLoader:
    @staticmethod
    def load_model_architecture(model_path: str) -> Dict:
        try:
            model = keras.models.load_model(model_path)
            
            architecture = {
                'layers': [],
                'input_shape': None,
                'output_shape': None
            }
            
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'index': i,
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'config': layer.get_config(),
                    'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None,
                    'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
                    'trainable_params': layer.count_params() if hasattr(layer, 'count_params') else 0
                }
                
                architecture['layers'].append(layer_info)
            
            # Set overall input and output shapes
            if model.layers:
                architecture['input_shape'] = model.layers[0].input_shape
                architecture['output_shape'] = model.layers[-1].output_shape
            
            return architecture
            
        except Exception as e:
            raise ValueError(f"Failed to load model architecture from {model_path}: {str(e)}")
    
    @staticmethod
    def extract_layer_weights(layer, layer_type: str) -> Optional[Dict]:
        weights = layer.get_weights()
        
        if not weights:
            return None
        
        if layer_type == 'Conv2D':
            return {
                'kernel': weights[0], 
                'bias': weights[1] if len(weights) > 1 else None
            }
        
        elif layer_type == 'Dense':
            return {
                'kernel': weights[0],  
                'bias': weights[1] if len(weights) > 1 else None
            }
        
        elif layer_type == 'Embedding':
            return {
                'embeddings': weights[0]  
            }
        
        elif layer_type == 'SimpleRNN':
            return {
                'kernel': weights[0],           
                'recurrent_kernel': weights[1], 
                'bias': weights[2] if len(weights) > 2 else None
            }
        
        elif layer_type == 'LSTM':
            return {
                'kernel': weights[0],          
                'recurrent_kernel': weights[1],
                'bias': weights[2] if len(weights) > 2 else None
            }
        
        elif layer_type == 'Bidirectional':
            num_weights = len(weights)
            forward_weights = weights[:num_weights//2]
            backward_weights = weights[num_weights//2:]
            
            return {
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
        
        else:
            return {'weights': weights}

class ActivationFunctions:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_values = np.exp(x_shifted)
        return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

class ConvolutionUtils:    
    @staticmethod
    def calculate_output_shape(input_shape: Tuple[int, int], 
                             kernel_size: Tuple[int, int],
                             stride: Tuple[int, int] = (1, 1),
                             padding: str = 'valid') -> Tuple[int, int]:

        input_height, input_width = input_shape
        kernel_height, kernel_width = kernel_size
        stride_height, stride_width = stride
        
        if padding == 'same':
            output_height = int(np.ceil(input_height / stride_height))
            output_width = int(np.ceil(input_width / stride_width))
        else:  
            output_height = int(np.ceil((input_height - kernel_height + 1) / stride_height))
            output_width = int(np.ceil((input_width - kernel_width + 1) / stride_width))
        
        return output_height, output_width
    
    @staticmethod
    def calculate_padding(input_shape: Tuple[int, int],
                         output_shape: Tuple[int, int],
                         kernel_size: Tuple[int, int],
                         stride: Tuple[int, int] = (1, 1)) -> Tuple[int, int, int, int]:

        input_height, input_width = input_shape
        output_height, output_width = output_shape
        kernel_height, kernel_width = kernel_size
        stride_height, stride_width = stride
        
        pad_height = max(0, (output_height - 1) * stride_height + kernel_height - input_height)
        pad_width = max(0, (output_width - 1) * stride_width + kernel_width - input_width)
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        return pad_top, pad_bottom, pad_left, pad_right

class TextProcessingUtils:

    @staticmethod
    def load_tokenizer_config(config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Tokenizer configuration not found: {config_path}")
    
    @staticmethod
    def load_vocabulary(vocab_path: str) -> List[str]:
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            return vocab
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    @staticmethod
    def pad_sequences(sequences: np.ndarray, maxlen: int, 
                     padding: str = 'pre', truncating: str = 'pre',
                     value: float = 0.0) -> np.ndarray:
        num_samples = len(sequences)
        result = np.full((num_samples, maxlen), value, dtype=sequences.dtype)
        
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            
            if len(seq) > maxlen:
                if truncating == 'pre':
                    seq = seq[-maxlen:]
                else:
                    seq = seq[:maxlen]
            
            if padding == 'pre':
                result[i, -len(seq):] = seq
            else:
                result[i, :len(seq)] = seq
        
        return result

class ValidationUtils:
 
    @staticmethod
    def compare_predictions(pred1: np.ndarray, pred2: np.ndarray, 
                          tolerance: float = 1e-5) -> Dict:

        if pred1.shape != pred2.shape:
            return {
                'shapes_match': False,
                'shape1': pred1.shape,
                'shape2': pred2.shape
            }
        
        abs_diff = np.abs(pred1 - pred2)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        close_elementwise = np.allclose(pred1, pred2, atol=tolerance)
        
        if pred1.ndim > 1 and pred1.shape[1] > 1:
            pred1_classes = np.argmax(pred1, axis=1)
            pred2_classes = np.argmax(pred2, axis=1)
            class_agreement = np.mean(pred1_classes == pred2_classes)
        else:
            class_agreement = None
        
        return {
            'shapes_match': True,
            'close_elementwise': close_elementwise,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'class_agreement': float(class_agreement) if class_agreement is not None else None,
            'tolerance_used': tolerance
        }
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred_classes, average='macro', zero_division=0
        )
        
        report = classification_report(y_true, y_pred_classes, output_dict=True, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'classification_report': report
        }

class BatchProcessor:
    
    @staticmethod
    def process_in_batches(data: np.ndarray, 
                          process_func: callable,
                          batch_size: int = 32,
                          **kwargs) -> np.ndarray:

        num_samples = data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = data[start_idx:end_idx]
            batch_result = process_func(batch_data, **kwargs)
            
            results.append(batch_result)
        
        return np.vstack(results) if results else np.array([])

class ConfigManager:
    
    @staticmethod
    def save_config(config: Dict, filepath: str):
        json_config = ConfigManager._convert_numpy_types(config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_config(filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def _convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ConfigManager._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ConfigManager._convert_numpy_types(item) for item in obj]
        else:
            return obj

class MemoryProfiler:
    @staticmethod
    def get_array_memory_mb(array: np.ndarray) -> float:
        return array.nbytes / (1024 * 1024)
    
    @staticmethod
    def profile_model_memory(model_weights: Dict) -> Dict:
        memory_info = {}
        total_memory = 0.0
        
        for layer_name, weights in model_weights.items():
            layer_memory = 0.0
            
            if isinstance(weights, dict):
                for weight_name, weight_array in weights.items():
                    if isinstance(weight_array, np.ndarray):
                        weight_memory = MemoryProfiler.get_array_memory_mb(weight_array)
                        layer_memory += weight_memory
            elif isinstance(weights, np.ndarray):
                layer_memory = MemoryProfiler.get_array_memory_mb(weights)
            
            memory_info[layer_name] = layer_memory
            total_memory += layer_memory
        
        memory_info['total_memory_mb'] = total_memory
        return memory_info

def test_utilities():
    print("Testing Custom Forward Propagation Utilities...")
    
    print("\n1. Testing Activation Functions:")
    x = np.array([-2, -1, 0, 1, 2])
    
    print(f"Input: {x}")
    print(f"ReLU: {ActivationFunctions.relu(x)}")
    print(f"Sigmoid: {ActivationFunctions.sigmoid(x)}")
    print(f"Tanh: {ActivationFunctions.tanh(x)}")
    
    x_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Softmax: {ActivationFunctions.softmax(x_2d)}")
    
    print("\n2. Testing Convolution Utilities:")
    input_shape = (32, 32)
    kernel_size = (3, 3)
    stride = (1, 1)
    
    output_shape_valid = ConvolutionUtils.calculate_output_shape(
        input_shape, kernel_size, stride, 'valid'
    )
    output_shape_same = ConvolutionUtils.calculate_output_shape(
        input_shape, kernel_size, stride, 'same'
    )
    
    print(f"Input shape: {input_shape}")
    print(f"Kernel size: {kernel_size}")
    print(f"Output shape (valid): {output_shape_valid}")
    print(f"Output shape (same): {output_shape_same}")
    
    print("\n3. Testing Validation Utilities:")
    pred1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    pred2 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]) + np.random.normal(0, 0.001, (2, 3))
    
    comparison = ValidationUtils.compare_predictions(pred1, pred2)
    print(f"Prediction comparison: {comparison}")
    
    print("\nUtility testing completed!")

if __name__ == "__main__":
    test_utilities()