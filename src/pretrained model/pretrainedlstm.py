import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import json
import re

class PretrainedLSTM:
    def __init__(self, vocab_size=10000, max_length=100, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_classes = 3  
        self.label_encoder = LabelEncoder()
        self.text_vectorizer = None
        
    def load_and_preprocess_data(self):
        train_df = pd.read_csv('../data/train.csv')
        val_df = pd.read_csv('../data/valid.csv')
        test_df = pd.read_csv('../data/test.csv')
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        
        train_texts = [preprocess_text(text) for text in train_df['text']]
        val_texts = [preprocess_text(text) for text in val_df['text']]
        test_texts = [preprocess_text(text) for text in test_df['text']]
        
        all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])
        self.label_encoder.fit(all_labels)
        
        train_labels = self.label_encoder.transform(train_df['label'])
        val_labels = self.label_encoder.transform(val_df['label'])
        test_labels = self.label_encoder.transform(test_df['label'])
        
        print(f"Label classes: {self.label_encoder.classes_}")
        print(f"Label distribution in train: {np.bincount(train_labels)}")
        
        self.text_vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_length,
            output_mode='int'
        )
        
        self.text_vectorizer.adapt(train_texts)

        train_sequences = self.text_vectorizer(train_texts)
        val_sequences = self.text_vectorizer(val_texts)
        test_sequences = self.text_vectorizer(test_texts)
        
        return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)
    
    def create_model_variant(self, lstm_layers=2, units_per_layer=[64, 32], 
                           bidirectional=True, dropout_rate=0.5):
        
        model = keras.Sequential()
        
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True
        ))
        
        for i in range(lstm_layers):
            units = units_per_layer[i] if i < len(units_per_layer) else units_per_layer[-1]
            return_sequences = i < lstm_layers - 1 
            
            lstm_layer = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate
            )
            
            if bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer)
            
            model.add(lstm_layer)

        model.add(layers.Dropout(dropout_rate))
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
                                     target_names=self.label_encoder.classes_)
        
        return f1_macro, report, y_pred_classes
    
    def run_experiments(self):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.load_and_preprocess_data()
        
        results = {}
        
        print("Starting LSTM Experiments...")

        print("\n=== Experiment 1: Number of LSTM Layers ===")
        layer_variants = [1, 2, 3]
        
        for layers_count in layer_variants:
            print(f"\nTraining LSTM with {layers_count} layers...")
            
            units_per_layer = [64] * layers_count
            
            model = self.create_model_variant(
                lstm_layers=layers_count,
                units_per_layer=units_per_layer,
                bidirectional=True,
                dropout_rate=0.5
            )
            model = self.compile_model(model)
            
            print(model.summary())
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/lstm_layers_{layers_count}.h5'
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            
            results[f'lstm_layers_{layers_count}'] = {
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"Macro F1-Score: {f1_macro:.4f}")
        
        print("\n=== Experiment 2: Number of Units per Layer ===")
        units_variants = [
            [32, 16],
            [64, 32],
            [128, 64]
        ]
        
        for i, units in enumerate(units_variants):
            print(f"\nTraining LSTM with units {units}...")
            
            model = self.create_model_variant(
                lstm_layers=2,
                units_per_layer=units,
                bidirectional=True,
                dropout_rate=0.5
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/lstm_units_{i+1}.h5'
            model.save(model_path)
            
            results[f'units_variant_{i+1}'] = {
                'units': units,
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"Units {units} - Macro F1-Score: {f1_macro:.4f}")

        print("\n=== Experiment 3: LSTM Direction ===")
        direction_variants = [False, True] 
        direction_names = ['Unidirectional', 'Bidirectional']
        
        for bidirectional, name in zip(direction_variants, direction_names):
            print(f"\nTraining {name} LSTM...")
            
            model = self.create_model_variant(
                lstm_layers=2,
                units_per_layer=[64, 32],
                bidirectional=bidirectional,
                dropout_rate=0.5
            )
            model = self.compile_model(model)
            
            history = self.train_model(model, x_train, y_train, x_val, y_val, epochs=30)
            f1_macro, report, y_pred = self.evaluate_model(model, x_test, y_test)
            
            model_path = f'models/lstm_direction_{name.lower()}.h5'
            model.save(model_path)
            
            results[f'direction_{name.lower()}'] = {
                'f1_macro': f1_macro,
                'history': history.history,
                'model_path': model_path,
                'report': report
            }
            
            print(f"{name} LSTM - Macro F1-Score: {f1_macro:.4f}")
        
        with open('lstm_results.json', 'w') as f:
            json_results = {}
            for key, value in results.items():
                json_results[key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                    for k, v in value.items() if k != 'report'
                }
            json.dump(json_results, f, indent=2)
        
        return results
    
    def plot_training_curves(self, results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('LSTM Training Curves Comparison')

        ax = axes[0]
        for layers_count in [1, 2, 3]:
            key = f'lstm_layers_{layers_count}'
            if key in results:
                history = results[key]['history']
                ax.plot(history['loss'], label=f'{layers_count} layers (train)')
                ax.plot(history['val_loss'], '--', label=f'{layers_count} layers (val)')
        ax.set_title('Effect of Number of Layers')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        ax = axes[1]
        for i in range(1, 4):
            key = f'units_variant_{i}'
            if key in results:
                history = results[key]['history']
                units = results[key]['units']
                ax.plot(history['loss'], label=f'Units {units} (train)')
                ax.plot(history['val_loss'], '--', label=f'Units {units} (val)')
        ax.set_title('Effect of Number of Units')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        ax = axes[2]
        for direction in ['unidirectional', 'bidirectional']:
            key = f'direction_{direction}'
            if key in results:
                history = results[key]['history']
                ax.plot(history['loss'], label=f'{direction} (train)')
                ax.plot(history['val_loss'], '--', label=f'{direction} (val)')
        ax.set_title('Effect of Direction')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_vectorizer(self):
        if self.text_vectorizer is not None:
            vocab = self.text_vectorizer.get_vocabulary()
            with open('text_vectorizer_vocab_lstm.json', 'w') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            config = {
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'embedding_dim': self.embedding_dim
            }
            with open('text_vectorizer_config_lstm.json', 'w') as f:
                json.dump(config, f, indent=2)

if __name__ == "__main__":
    lstm = PretrainedLSTM()
    results = lstm.run_experiments()
    lstm.plot_training_curves(results)
    lstm.save_vectorizer()
    
    print("\n=== LSTM Experiments Summary ===")
    for exp_name, exp_results in results.items():
        print(f"{exp_name}: F1-Score = {exp_results['f1_macro']:.4f}")