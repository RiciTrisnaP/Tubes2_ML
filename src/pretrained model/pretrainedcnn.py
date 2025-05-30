import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import os
import json

class PretrainedCNN:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = x_train_full[:40000]
        y_train = y_train_full[:40000]
        x_val = x_train_full[40000:]
        y_val = y_train_full[40000:]
        
        y_train = y_train.flatten()
        y_val = y_val.flatten()
        y_test = y_test.flatten()
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def create_model_variant(self, conv_layers=3, filters_per_layer=[32, 64, 128], 
                           kernel_sizes=[3, 3, 3], pooling_type='max'):
        model = keras.Sequential([
            layers.Input(shape=self.input_shape)
        ])
        
        for i in range(conv_layers):
            model.add(layers.Conv2D(
                filters=filters_per_layer[i] if i < len(filters_per_layer) else filters_per_layer[-1],
                kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1],
                activation='relu',
                padding='same'
            ))
            
            if pooling_type == 'max':
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            else:
                model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        model.add(layers.GlobalAveragePooling2D())
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def compile_model(self, model):
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, model, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        f1_macro = f1_score(y_test, y_pred_classes, average='macro')
        
        report = classification_report(y_test, y_pred_classes, 
                                     target_names=[f'Class_{i}' for i in range(self.num_classes)])
        
        return f1_macro, report, y_pred_classes
    
    def run_experiments(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.load_and_preprocess_data()
        
        results = {}
        
        print("Starting CNN Experiments...")
        
        print("\n=== Experiment 1: Number of Convolutional Layers ===")
        conv_layer_variants = [2, 3, 4]
        
        for layers_count in conv_layer_variants:
            print(f"\nTraining CNN with {layers_count} convolutional layers...")
            
            model = self.create_model_variant(
                conv_layers=layers_count,
                filters_per_layer=[32, 64, 128, 256],
                kernel_sizes=[3, 3, 3, 3],
                pooling_type='max'
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/cnn_layers_{layers_count}.h5'
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            
            results[f'conv_layers_{layers_count}'] = {
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"Macro F1-Score: {f1_macro:.4f}")
        
        print("\n=== Experiment 2: Number of Filters per Layer ===")
        filter_variants = [
            [16, 32, 64],
            [32, 64, 128],
            [64, 128, 256]
        ]
        
        for i, filters in enumerate(filter_variants):
            print(f"\nTraining CNN with filters {filters}...")
            
            model = self.create_model_variant(
                conv_layers=3,
                filters_per_layer=filters,
                kernel_sizes=[3, 3, 3],
                pooling_type='max'
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/cnn_filters_{i+1}.h5'
            model.save(model_path)
            
            results[f'filters_variant_{i+1}'] = {
                'filters': filters,
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"Filters {filters} - Macro F1-Score: {f1_macro:.4f}")
        
        print("\n=== Experiment 3: Kernel Sizes ===")
        kernel_variants = [
            [3, 3, 3],
            [5, 5, 5],
            [3, 5, 7]
        ]
        
        for i, kernels in enumerate(kernel_variants):
            print(f"\nTraining CNN with kernel sizes {kernels}...")
            
            model = self.create_model_variant(
                conv_layers=3,
                filters_per_layer=[32, 64, 128],
                kernel_sizes=kernels,
                pooling_type='max'
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/cnn_kernels_{i+1}.h5'
            model.save(model_path)
            
            results[f'kernels_variant_{i+1}'] = {
                'kernels': kernels,
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"Kernels {kernels} - Macro F1-Score: {f1_macro:.4f}")
        
        print("\n=== Experiment 4: Pooling Type ===")
        pooling_types = ['max', 'average']
        
        for pooling in pooling_types:
            print(f"\nTraining CNN with {pooling} pooling...")
            
            model = self.create_model_variant(
                conv_layers=3,
                filters_per_layer=[32, 64, 128],
                kernel_sizes=[3, 3, 3],
                pooling_type=pooling
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/cnn_pooling_{pooling}.h5'
            model.save(model_path)
            
            results[f'pooling_{pooling}'] = {
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"{pooling} pooling - Macro F1-Score: {f1_macro:.4f}")
        
        with open('cnn_results.json', 'w') as f:
            json_results = {}
            for key, value in results.items():
                json_results[key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                    for k, v in value.items() if k != 'report'
                }
            json.dump(json_results, f, indent=2)
        
        return results
    
    def plot_training_curves(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN Training Curves Comparison')
        
        ax = axes[0, 0]
        for key in ['conv_layers_2', 'conv_layers_3', 'conv_layers_4']:
            if key in results:
                history = results[key]['history']
                ax.plot(history['loss'], label=f'{key} (train)')
                ax.plot(history['val_loss'], '--', label=f'{key} (val)')
        ax.set_title('Effect of Number of Layers')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        ax = axes[0, 1]
        for i in range(1, 4):
            key = f'filters_variant_{i}'
            if key in results:
                history = results[key]['history']
                filters = results[key]['filters']
                ax.plot(history['loss'], label=f'Filters {filters} (train)')
                ax.plot(history['val_loss'], '--', label=f'Filters {filters} (val)')
        ax.set_title('Effect of Number of Filters')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        ax = axes[1, 0]
        for i in range(1, 4):
            key = f'kernels_variant_{i}'
            if key in results:
                history = results[key]['history']
                kernels = results[key]['kernels']
                ax.plot(history['loss'], label=f'Kernels {kernels} (train)')
                ax.plot(history['val_loss'], '--', label=f'Kernels {kernels} (val)')
        ax.set_title('Effect of Kernel Sizes')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        ax = axes[1, 1]
        for pooling in ['max', 'average']:
            key = f'pooling_{pooling}'
            if key in results:
                history = results[key]['history']
                ax.plot(history['loss'], label=f'{pooling} pooling (train)')
                ax.plot(history['val_loss'], '--', label=f'{pooling} pooling (val)')
        ax.set_title('Effect of Pooling Type')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('cnn_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

if __name__ == "__main__":
    cnn = PretrainedCNN()
    results = cnn.run_experiments()
    cnn.plot_training_curves(results)
    
    print("\n=== Hasil CNN ===")
    for exp_name, exp_results in results.items():
        print(f"{exp_name}: F1-Score = {exp_results['f1_macro']:.4f}")