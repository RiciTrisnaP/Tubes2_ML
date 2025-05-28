import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, 
    TextVectorization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class LSTMSentimentAnalyzer:
    def __init__(self, vocab_size=10000, max_length=100, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.histories = {}
        
    def load_data(self, train_path, valid_path, test_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        
        # Load CSV files
        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Train data shape: {self.train_df.shape}")
        print(f"Validation data shape: {self.valid_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        # Display basic info about the dataset
        print(f"\nSentiment distribution in training data:")
        print(self.train_df['label'].value_counts())
        
        return self.train_df, self.valid_df, self.test_df
    
    def setup_text_vectorization(self):
        """Setup TextVectorization layer for tokenization"""
        print("Setting up text vectorization...")
        
        # Create TextVectorization layer
        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_length,
            output_mode='int'
        )
        
        # Adapt the vectorizer on training data
        text_data = self.train_df['text'].values
        self.vectorizer.adapt(text_data)
        
        print(f"Vocabulary size: {len(self.vectorizer.get_vocabulary())}")
        
    def preprocess_data(self):
        """Preprocess text data and encode labels"""
        print("Preprocessing data...")
        
        # Vectorize text data
        X_train = self.vectorizer(self.train_df['text'].values)
        X_valid = self.vectorizer(self.valid_df['text'].values)
        X_test = self.vectorizer(self.test_df['text'].values)
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(self.train_df['label'])
        y_valid = self.label_encoder.transform(self.valid_df['label'])
        y_test = self.label_encoder.transform(self.test_df['label'])
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    
    def create_lstm_model(self, num_layers=1, units_per_layer=[128], 
                         use_bidirectional=True, dropout_rate=0.3):
        """Create LSTM model with specified architecture"""
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length
        ))
        
        # LSTM layers
        for i, units in enumerate(units_per_layer):
            # Return sequences for all LSTM layers except the last one
            return_sequences = (i < len(units_per_layer) - 1) or (len(units_per_layer) == 1 and num_layers > 1)
            
            if use_bidirectional:
                model.add(Bidirectional(LSTM(
                    units, 
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate
                )))
            else:
                model.add(LSTM(
                    units, 
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate
                ))
            
            # Add dropout layer after each LSTM layer
            model.add(Dropout(dropout_rate))
        
        # Dense output layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_valid, y_valid, 
                   model_name, epochs=50, batch_size=32):
        """Train the model with early stopping"""
        print(f"\nTraining model: {model_name}")
        print(f"Model architecture:")
        model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            f'best_model_{model_name}.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and calculate metrics"""
        print(f"\nEvaluating model: {model_name}")
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'macro_f1': macro_f1,
            'predictions': y_pred
        }
    
    def plot_training_history(self, model_names, figsize=(15, 10)):
        """Plot training and validation loss/accuracy"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot training loss
        axes[0, 0].set_title('Training Loss')
        for name in model_names:
            if name in self.histories:
                axes[0, 0].plot(self.histories[name].history['loss'], label=name)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot validation loss
        axes[0, 1].set_title('Validation Loss')
        for name in model_names:
            if name in self.histories:
                axes[0, 1].plot(self.histories[name].history['val_loss'], label=name)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot training accuracy
        axes[1, 0].set_title('Training Accuracy')
        for name in model_names:
            if name in self.histories:
                axes[1, 0].plot(self.histories[name].history['accuracy'], label=name)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot validation accuracy
        axes[1, 1].set_title('Validation Accuracy')
        for name in model_names:
            if name in self.histories:
                axes[1, 1].plot(self.histories[name].history['val_accuracy'], label=name)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, results_dict):
        """Compare model performance"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(results_dict.keys()),
            'Test Accuracy': [results_dict[model]['test_accuracy'] for model in results_dict.keys()],
            'Macro F1-Score': [results_dict[model]['macro_f1'] for model in results_dict.keys()],
            'Test Loss': [results_dict[model]['test_loss'] for model in results_dict.keys()]
        })
        
        print(comparison_df.round(4))
        
        # Best model based on macro F1-score
        best_model = comparison_df.loc[comparison_df['Macro F1-Score'].idxmax(), 'Model']
        print(f"\nBest model based on Macro F1-Score: {best_model}")
        
        return comparison_df

def main():
    # Initialize analyzer
    analyzer = LSTMSentimentAnalyzer(vocab_size=10000, max_length=100, embedding_dim=128)
    
    # Load data (adjust paths as needed)
    train_df, valid_df, test_df = analyzer.load_data('src/data/train.csv', 'src/data/valid.csv', 'src/data/test.csv')
    
    # Setup text vectorization
    analyzer.setup_text_vectorization()
    
    # Preprocess data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = analyzer.preprocess_data()
    
    results = {}
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: PENGARUH JUMLAH LAYER LSTM")
    print("="*60)
    
    # Experiment 1: Effect of number of LSTM layers
    layer_configs = [
        {"name": "1_layer", "units": [128], "description": "1 LSTM Layer"},
        {"name": "2_layers", "units": [128, 64], "description": "2 LSTM Layers"},
        {"name": "3_layers", "units": [128, 64, 32], "description": "3 LSTM Layers"}
    ]
    
    layer_results = {}
    for config in layer_configs:
        model = analyzer.create_lstm_model(
            num_layers=len(config["units"]),
            units_per_layer=config["units"],
            use_bidirectional=True,
            dropout_rate=0.3
        )
        
        model, history = analyzer.train_model(
            model, X_train, y_train, X_valid, y_valid,
            config["name"], epochs=30
        )
        
        result = analyzer.evaluate_model(model, X_test, y_test, config["name"])
        layer_results[config["description"]] = result
    
    # Plot comparison for layer experiment
    analyzer.plot_training_history([config["name"] for config in layer_configs])
    layer_comparison = analyzer.compare_models(layer_results)
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: PENGARUH BANYAK CELL LSTM PER LAYER")
    print("="*60)
    
    # Experiment 2: Effect of number of LSTM cells per layer
    cell_configs = [
        {"name": "small_cells", "units": [64], "description": "64 cells"},
        {"name": "medium_cells", "units": [128], "description": "128 cells"},
        {"name": "large_cells", "units": [256], "description": "256 cells"}
    ]
    
    cell_results = {}
    for config in cell_configs:
        model = analyzer.create_lstm_model(
            num_layers=1,
            units_per_layer=config["units"],
            use_bidirectional=True,
            dropout_rate=0.3
        )
        
        model, history = analyzer.train_model(
            model, X_train, y_train, X_valid, y_valid,
            config["name"], epochs=30
        )
        
        result = analyzer.evaluate_model(model, X_test, y_test, config["name"])
        cell_results[config["description"]] = result
    
    # Plot comparison for cell experiment
    analyzer.plot_training_history([config["name"] for config in cell_configs])
    cell_comparison = analyzer.compare_models(cell_results)
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: PENGARUH JENIS LAYER LSTM (BIDIRECTIONAL VS UNIDIRECTIONAL)")
    print("="*60)
    
    # Experiment 3: Effect of LSTM direction
    direction_configs = [
        {"name": "unidirectional", "bidirectional": False, "description": "Unidirectional LSTM"},
        {"name": "bidirectional", "bidirectional": True, "description": "Bidirectional LSTM"}
    ]
    
    direction_results = {}
    for config in direction_configs:
        model = analyzer.create_lstm_model(
            num_layers=1,
            units_per_layer=[128],
            use_bidirectional=config["bidirectional"],
            dropout_rate=0.3
        )
        
        model, history = analyzer.train_model(
            model, X_train, y_train, X_valid, y_valid,
            config["name"], epochs=30
        )
        
        result = analyzer.evaluate_model(model, X_test, y_test, config["name"])
        direction_results[config["description"]] = result
    
    # Plot comparison for direction experiment
    analyzer.plot_training_history([config["name"] for config in direction_configs])
    direction_comparison = analyzer.compare_models(direction_results)
    
    # Final summary
    print("\n" + "="*80)
    print("KESIMPULAN ANALISIS")
    print("="*80)
    
    print("\n1. PENGARUH JUMLAH LAYER LSTM:")
    best_layers = layer_comparison.loc[layer_comparison['Macro F1-Score'].idxmax(), 'Model']
    print(f"   - Model terbaik: {best_layers}")
    print("   - Analisis: Menambah layer LSTM dapat meningkatkan kemampuan model untuk")
    print("     menangkap pola yang lebih kompleks, namun juga meningkatkan risiko overfitting.")
    
    print("\n2. PENGARUH BANYAK CELL LSTM PER LAYER:")
    best_cells = cell_comparison.loc[cell_comparison['Macro F1-Score'].idxmax(), 'Model']
    print(f"   - Model terbaik: {best_cells}")
    print("   - Analisis: Jumlah cell yang lebih banyak memberikan kapasitas model yang lebih besar")
    print("     untuk mempelajari representasi yang kompleks, namun perlu diseimbangkan dengan")
    print("     risiko overfitting dan computational cost.")
    
    print("\n3. PENGARUH JENIS LAYER LSTM (BIDIRECTIONAL VS UNIDIRECTIONAL):")
    best_direction = direction_comparison.loc[direction_comparison['Macro F1-Score'].idxmax(), 'Model']
    print(f"   - Model terbaik: {best_direction}")
    print("   - Analisis: Bidirectional LSTM umumnya memberikan performa yang lebih baik")
    print("     karena dapat memproses informasi dari kedua arah (forward dan backward),")
    print("     memberikan konteks yang lebih lengkap untuk prediksi.")
    
    # Save best model weights
    print(f"\nModel weights telah disimpan sebagai file .h5 untuk setiap konfigurasi yang diuji.")
    
    return analyzer, layer_comparison, cell_comparison, direction_comparison

if __name__ == "__main__":
    analyzer, layer_comp, cell_comp, direction_comp = main()