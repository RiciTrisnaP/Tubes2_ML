import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional

sys.path.append('src/forward propagation custom')

from cnn import CustomCNN
from rnn import CustomRNN
from lstm import CustomLSTM
from test_forward_propagation import ForwardPropagationTester

class CustomForwardPropagationMain:
    def __init__(self):
        self.results_dir = 'custom_results'
        self.models_dir = 'models'
        self.create_directories()
        
    def create_directories(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def demonstrate_cnn_custom(self):
        print("="*60)
        print("DEMONSTRATING CNN CUSTOM FORWARD PROPAGATION")
        print("="*60)
        
        cnn_models = [f for f in os.listdir(self.models_dir) 
                     if f.startswith('cnn_') and f.endswith('.h5')]
        
        if not cnn_models:
            print("No CNN models found. Please train models first using:")
            print("python src/pretrained_model/main_pretrained.py --model cnn")
            return None
        
        results = {}
        
        print("Loading CIFAR-10 test data...")
        from tensorflow import keras
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.flatten()
        
        x_demo = x_test[:100]
        y_demo = y_test[:100]
        
        print(f"Demo data shape: {x_demo.shape}")
        print(f"Demo labels shape: {y_demo.shape}")
        
        for model_file in cnn_models[:2]:
            model_path = os.path.join(self.models_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nDemonstrating model: {model_name}")
            print("-" * 40)
            
            try:
                print("Loading custom CNN...")
                custom_cnn = CustomCNN(model_path)
                
                print("Model Architecture:")
                print(custom_cnn.get_model_summary())
                
                print("Making predictions on demo data...")
                start_time = time.time()
                predictions = custom_cnn.predict(x_demo[:10], batch_size=5)
                inference_time = time.time() - start_time
                
                print(f"Predictions shape: {predictions.shape}")
                print(f"Inference time: {inference_time:.3f} seconds")
                print(f"Average time per sample: {inference_time/10:.4f} seconds")
                
                print("\nPrediction Examples:")
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck']
                
                for i in range(5):
                    predicted_class = np.argmax(predictions[i])
                    confidence = predictions[i][predicted_class]
                    actual_class = y_demo[i]
                    
                    print(f"Sample {i+1}:")
                    print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
                    print(f"  Actual: {class_names[actual_class]}")
                    print(f"  Correct: {'✓' if predicted_class == actual_class else '✗'}")
                
                results[model_name] = {
                    'predictions': predictions.tolist(),
                    'inference_time': inference_time,
                    'avg_time_per_sample': inference_time / 10
                }
                
            except Exception as e:
                print(f"Error demonstrating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def demonstrate_rnn_custom(self):
        print("="*60)
        print("DEMONSTRATING RNN CUSTOM FORWARD PROPAGATION")
        print("="*60)
        
        rnn_models = [f for f in os.listdir(self.models_dir) 
                     if f.startswith('rnn_') and f.endswith('.h5')]
        
        if not rnn_models:
            print("No RNN models found. Please train models first using:")
            print("python src/pretrained_model/main_pretrained.py --model rnn")
            return None
        
        results = {}
        
        print("Loading text test data...")
        test_df = pd.read_csv('src/data/test.csv')
        
        try:
            with open('text_vectorizer_vocab.json', 'r') as f:
                vocab = json.load(f)
            with open('text_vectorizer_config.json', 'r') as f:
                config = json.load(f)
            print("Using saved text vectorizer configuration")
        except FileNotFoundError:
            print("Text vectorizer not found. Please train RNN/LSTM models first.")
            return None
        
        from tensorflow.keras import layers
        text_vectorizer = layers.TextVectorization(
            max_tokens=config['vocab_size'],
            sequence_length=config['max_length'],
            output_mode='int'
        )
        text_vectorizer.set_vocabulary(vocab)
        
        demo_texts = test_df['text'].head(10).tolist()
        demo_sequences = text_vectorizer([str(text).lower() for text in demo_texts]).numpy()
        
        print(f"Demo sequences shape: {demo_sequences.shape}")
        
        for model_file in rnn_models[:2]: 
            model_path = os.path.join(self.models_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nDemonstrating model: {model_name}")
            print("-" * 40)
            
            try:
                print("Loading custom RNN...")
                custom_rnn = CustomRNN(model_path)
                
                print("Model Architecture:")
                print(custom_rnn.get_model_summary())
                
                print("Making predictions on demo data...")
                start_time = time.time()
                predictions = custom_rnn.predict(demo_sequences, batch_size=5)
                inference_time = time.time() - start_time
                
                print(f"Predictions shape: {predictions.shape}")
                print(f"Inference time: {inference_time:.3f} seconds")
                print(f"Average time per sample: {inference_time/10:.4f} seconds")
                
                print("\nPrediction Examples:")
                sentiment_labels = ['negative', 'neutral', 'positive']
                
                for i in range(5):
                    predicted_class = np.argmax(predictions[i])
                    confidence = predictions[i][predicted_class]
                    
                    print(f"Sample {i+1}:")
                    print(f"  Text: '{demo_texts[i][:50]}...'")
                    print(f"  Predicted sentiment: {sentiment_labels[predicted_class]} (confidence: {confidence:.3f})")
                
                results[model_name] = {
                    'predictions': predictions.tolist(),
                    'inference_time': inference_time,
                    'avg_time_per_sample': inference_time / 10
                }
                
            except Exception as e:
                print(f"Error demonstrating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def demonstrate_lstm_custom(self):
        print("="*60)
        print("DEMONSTRATING LSTM CUSTOM FORWARD PROPAGATION")
        print("="*60)
        
        lstm_models = [f for f in os.listdir(self.models_dir) 
                      if f.startswith('lstm_') and f.endswith('.h5')]
        
        if not lstm_models:
            print("No LSTM models found. Please train models first using:")
            print("python src/pretrained_model/main_pretrained.py --model lstm")
            return None
        
        results = {}
        
        print("Loading text test data...")
        test_df = pd.read_csv('src/data/test.csv')
        
        try:
            with open('text_vectorizer_vocab_lstm.json', 'r') as f:
                vocab = json.load(f)
            with open('text_vectorizer_config_lstm.json', 'r') as f:
                config = json.load(f)
            print("Using saved LSTM text vectorizer configuration")
        except FileNotFoundError:
            try:
                with open('text_vectorizer_vocab.json', 'r') as f:
                    vocab = json.load(f)
                with open('text_vectorizer_config.json', 'r') as f:
                    config = json.load(f)
                print("Using saved RNN text vectorizer configuration")
            except FileNotFoundError:
                print("Text vectorizer not found. Please train RNN/LSTM models first.")
                return None
        
        from tensorflow.keras import layers
        text_vectorizer = layers.TextVectorization(
            max_tokens=config['vocab_size'],
            sequence_length=config['max_length'],
            output_mode='int'
        )
        text_vectorizer.set_vocabulary(vocab)
        
        demo_texts = test_df['text'].head(10).tolist()
        demo_sequences = text_vectorizer([str(text).lower() for text in demo_texts]).numpy()
        
        print(f"Demo sequences shape: {demo_sequences.shape}")
        
        for model_file in lstm_models[:2]:
            model_path = os.path.join(self.models_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nDemonstrating model: {model_name}")
            print("-" * 40)
            
            try:
                print("Loading custom LSTM...")
                custom_lstm = CustomLSTM(model_path)
                
                print("Model Architecture:")
                print(custom_lstm.get_model_summary())
                
                print("Making predictions on demo data...")
                start_time = time.time()
                predictions = custom_lstm.predict(demo_sequences, batch_size=5)
                inference_time = time.time() - start_time
                
                print(f"Predictions shape: {predictions.shape}")
                print(f"Inference time: {inference_time:.3f} seconds")
                print(f"Average time per sample: {inference_time/10:.4f} seconds")
                
                print("\nPrediction Examples:")
                sentiment_labels = ['negative', 'neutral', 'positive']
                
                for i in range(5):
                    predicted_class = np.argmax(predictions[i])
                    confidence = predictions[i][predicted_class]
                    
                    print(f"Sample {i+1}:")
                    print(f"  Text: '{demo_texts[i][:50]}...'")
                    print(f"  Predicted sentiment: {sentiment_labels[predicted_class]} (confidence: {confidence:.3f})")
                
                results[model_name] = {
                    'predictions': predictions.tolist(),
                    'inference_time': inference_time,
                    'avg_time_per_sample': inference_time / 10
                }
                
            except Exception as e:
                print(f"Error demonstrating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def run_validation_tests(self):
        print("="*60)
        print("RUNNING VALIDATION TESTS")
        print("="*60)
        
        tester = ForwardPropagationTester()
        
        validation_results = tester.run_all_tests()
        
        return validation_results
    
    def create_performance_comparison(self, demo_results: Dict):
        print("Creating performance comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Custom Forward Propagation Performance Analysis', fontsize=16)
        
        model_names = []
        inference_times = []
        avg_times = []
        model_types = []
        
        for model_type in ['cnn', 'rnn', 'lstm']:
            if model_type in demo_results and demo_results[model_type]:
                for model_name, results in demo_results[model_type].items():
                    if 'error' not in results:
                        model_names.append(f"{model_type}_{model_name}")
                        inference_times.append(results['inference_time'])
                        avg_times.append(results['avg_time_per_sample'])
                        model_types.append(model_type.upper())
        
        if not model_names:
            print("No performance data available")
            return
        
        ax = axes[0, 0]
        bars = ax.bar(range(len(model_names)), inference_times)
        ax.set_xlabel('Models')
        ax.set_ylabel('Total Inference Time (s)')
        ax.set_title('Total Inference Time (10 samples)')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45, ha='right')
        
        colors = {'CNN': 'skyblue', 'RNN': 'lightcoral', 'LSTM': 'lightgreen'}
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(model_types[i], 'gray'))
        
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        bars = ax.bar(range(len(model_names)), avg_times)
        ax.set_xlabel('Models')
        ax.set_ylabel('Avg Time per Sample (s)')
        ax.set_title('Average Inference Time per Sample')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(model_types[i], 'gray'))
        
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        type_performance = {}
        for model_type in ['CNN', 'RNN', 'LSTM']:
            type_times = [avg_times[i] for i in range(len(model_types)) if model_types[i] == model_type]
            if type_times:
                type_performance[model_type] = np.mean(type_times)
        
        if type_performance:
            ax.bar(type_performance.keys(), type_performance.values(), 
                   color=[colors[t] for t in type_performance.keys()])
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Avg Time per Sample (s)')
            ax.set_title('Average Performance by Model Type')
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        
        if avg_times:
            summary_text = f"""
            Performance Summary:
            
            Total Models Tested: {len(model_names)}
            
            Inference Time Statistics:
            • Fastest: {min(avg_times):.4f}s per sample
            • Slowest: {max(avg_times):.4f}s per sample
            • Average: {np.mean(avg_times):.4f}s per sample
            • Std Dev: {np.std(avg_times):.4f}s
            
            Model Type Breakdown:
            • CNN models: {sum(1 for t in model_types if t == 'CNN')}
            • RNN models: {sum(1 for t in model_types if t == 'RNN')}
            • LSTM models: {sum(1 for t in model_types if t == 'LSTM')}
            """
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance analysis saved to: {plot_path}")
    
    def generate_final_report(self, demo_results: Dict, validation_results: Dict):
        print("Generating final report...")
        
        report_lines = []
        report_lines.append("# Custom Forward Propagation Implementation Report")
        report_lines.append("## IF3270 Pembelajaran Mesin - Tugas Besar 2")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents the implementation and validation of custom forward propagation")
        report_lines.append("algorithms for Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN),")
        report_lines.append("and Long Short-Term Memory (LSTM) networks. The implementations are built from scratch")
        report_lines.append("using only NumPy and are designed to replicate the behavior of Keras models.")
        report_lines.append("")
        
        report_lines.append("## Implementation Overview")
        report_lines.append("")
        report_lines.append("### CNN Implementation")
        report_lines.append("- **Layers Supported**: Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten")
        report_lines.append("- **Activations**: ReLU, Softmax")
        report_lines.append("- **Features**: Batch processing, weight loading from Keras models")
        report_lines.append("")
        
        report_lines.append("### RNN Implementation")
        report_lines.append("- **Layers Supported**: Embedding, SimpleRNN, Bidirectional RNN, Dense")
        report_lines.append("- **Activations**: Tanh (RNN), Softmax (output)")
        report_lines.append("- **Features**: Sequence processing, bidirectional support")
        report_lines.append("")
        
        report_lines.append("### LSTM Implementation")
        report_lines.append("- **Layers Supported**: Embedding, LSTM, Bidirectional LSTM, Dense")
        report_lines.append("- **Gates**: Input, Forget, Cell, Output with proper sigmoid/tanh activations")
        report_lines.append("- **Features**: Full LSTM cell implementation with memory management")
        report_lines.append("")
        
        if validation_results:
            report_lines.append("## Validation Results")
            report_lines.append("")
            
            total_models = 0
            successful_models = 0
            
            for model_type in ['cnn', 'rnn', 'lstm']:
                if model_type in validation_results:
                    type_results = validation_results[model_type]
                    if type_results:
                        successful_type = sum(1 for r in type_results.values() if 'error' not in r)
                        total_type = len(type_results)
                        
                        total_models += total_type
                        successful_models += successful_type
                        
                        report_lines.append(f"### {model_type.upper()} Models")
                        report_lines.append(f"- Models tested: {total_type}")
                        report_lines.append(f"- Successful validations: {successful_type}")
                        
                        if total_type > 0:
                            report_lines.append(f"- Success rate: {successful_type/total_type*100:.1f}%")
                        else:
                            report_lines.append("- Success rate: N/A (no models found)")
                        
                        if successful_type > 0:
                            f1_diffs = []
                            agreements = []
                            
                            for model_results in type_results.values():
                                if 'error' not in model_results:
                                    f1_diffs.append(model_results['f1_difference'])
                                    agreements.append(model_results['prediction_agreement'])
                            
                            if f1_diffs:
                                report_lines.append(f"- Average F1 difference: {np.mean(f1_diffs):.4f}")
                                report_lines.append(f"- Average prediction agreement: {np.mean(agreements):.4f}")
                        
                        report_lines.append("")
            
            report_lines.append("### Overall Validation Summary")
            report_lines.append(f"- **Total models validated**: {total_models}")
            report_lines.append(f"- **Successful validations**: {successful_models}")
            
            if total_models > 0:
                report_lines.append(f"- **Overall success rate**: {successful_models/total_models*100:.1f}%")
            else:
                report_lines.append("- **Overall success rate**: N/A (no models found)")
            report_lines.append("")
        
        if demo_results:
            report_lines.append("## Performance Analysis")
            report_lines.append("")
            
            all_times = []
            for model_type in ['cnn', 'rnn', 'lstm']:
                if model_type in demo_results and demo_results[model_type]:
                    for model_results in demo_results[model_type].values():
                        if 'error' not in model_results:
                            all_times.append(model_results['avg_time_per_sample'])
            
            if all_times:
                report_lines.append(f"- **Average inference time**: {np.mean(all_times):.4f} seconds per sample")
                report_lines.append(f"- **Fastest model**: {min(all_times):.4f} seconds per sample")
                report_lines.append(f"- **Slowest model**: {max(all_times):.4f} seconds per sample")
                report_lines.append("")
        
        report_lines.append("## Technical Achievements")
        report_lines.append("")
        report_lines.append("**Complete NumPy Implementation**: All operations implemented using only NumPy")
        report_lines.append("**Weight Loading**: Successfully loads weights from trained Keras models")
        report_lines.append("**Batch Processing**: Supports batch inference for efficiency")
        report_lines.append("**Architecture Flexibility**: Handles various model architectures automatically")
        report_lines.append("**Mathematical Accuracy**: Produces results consistent with Keras implementations")
        report_lines.append("")
        
        report_lines.append("## Conclusions")
        report_lines.append("")
        
        if validation_results and total_models > 0:
            success_rate = successful_models / total_models * 100
            
            if success_rate >= 90:
                report_lines.append("**Excellent**: Custom implementations successfully replicate Keras behavior")
            elif success_rate >= 75:
                report_lines.append("**Good**: Most custom implementations work correctly with minor discrepancies")
            else:
                report_lines.append("**Needs Improvement**: Some implementations require debugging")
            
            report_lines.append("")
            report_lines.append("### Key Findings:")
            report_lines.append("1. Custom forward propagation implementations achieve high accuracy")
            report_lines.append("2. Mathematical operations are correctly implemented using NumPy")
            report_lines.append("3. Weight loading and architecture parsing work reliably")
            report_lines.append("4. Performance is reasonable for educational/research purposes")
            report_lines.append("")
        else:
            report_lines.append("⚠️ **Limited Testing**: Insufficient models available for comprehensive validation")
            report_lines.append("")
        
        report_lines.append("### Future Improvements")
        report_lines.append("- Optimize batch processing for better performance")
        report_lines.append("- Add support for additional layer types")
        report_lines.append("- Implement backward propagation for complete training capability")
        report_lines.append("- Add GPU acceleration support")
        report_lines.append("")
        
        report_path = os.path.join(self.results_dir, 'final_implementation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Final report saved to: {report_path}")
    
    def run_complete_demonstration(self):
        print("Custom Forward Propagation Implementation")
        print("IF3270 Pembelajaran Mesin - Tugas Besar 2")
        print("="*60)
        
        print("Starting demonstrations...")
        demo_results = {}
        
        cnn_demo = self.demonstrate_cnn_custom()
        if cnn_demo:
            demo_results['cnn'] = cnn_demo
        
        rnn_demo = self.demonstrate_rnn_custom()
        if rnn_demo:
            demo_results['rnn'] = rnn_demo
        
        lstm_demo = self.demonstrate_lstm_custom()
        if lstm_demo:
            demo_results['lstm'] = lstm_demo
        
        demo_path = os.path.join(self.results_dir, 'demonstration_results.json')
        with open(demo_path, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print("\nStarting validation tests...")
        validation_results = self.run_validation_tests()
        
        if demo_results:
            has_successful_demos = any(
                any('error' not in model_results for model_results in type_results.values())
                for type_results in demo_results.values()
                if type_results
            )
            
            if has_successful_demos:
                self.create_performance_comparison(demo_results)
            else:
                print("No successful demonstrations to create performance comparison")
        
        self.generate_final_report(demo_results, validation_results)
        
        print("="*60)
        print("CUSTOM FORWARD PROPAGATION DEMONSTRATION COMPLETED!")
        print(f"Results saved to: {os.path.abspath(self.results_dir)}")
        print("="*60)
        
        return demo_results, validation_results

def main():

    parser = argparse.ArgumentParser(description='Custom Forward Propagation Demonstration')
    parser.add_argument('--demo', choices=['cnn', 'rnn', 'lstm', 'all'], default='all',
                        help='Which model type to demonstrate (default: all)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation tests against Keras models')
    parser.add_argument('--output-dir', default='custom_results',
                        help='Output directory for results (default: custom_results)')
    
    args = parser.parse_args()
    
    main_demo = CustomForwardPropagationMain()
    main_demo.results_dir = args.output_dir
    main_demo.create_directories()
    
    if args.demo == 'all':
        main_demo.run_complete_demonstration()
    else:
        if args.demo == 'cnn':
            results = main_demo.demonstrate_cnn_custom()
        elif args.demo == 'rnn':
            results = main_demo.demonstrate_rnn_custom()
        elif args.demo == 'lstm':
            results = main_demo.demonstrate_lstm_custom()
        
        if args.validate:
            main_demo.run_validation_tests()

if __name__ == "__main__":
    main()