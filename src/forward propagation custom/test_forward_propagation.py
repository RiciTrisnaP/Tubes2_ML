import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

sys.path.append('src/forward propagation custom')

from cnn import CustomCNN
from rnn import CustomRNN  
from lstm import CustomLSTM

class ForwardPropagationTester:
    def __init__(self):
        self.results = {}
        self.test_results_dir = "test_results"
        os.makedirs(self.test_results_dir, exist_ok=True)
    
    def load_cifar10_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading CIFAR-10 test data...")
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.flatten()
        
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return x_test, y_test
    
    def load_text_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print("Loading text test data...")
        
        train_df = pd.read_csv('src/data/train.csv')
        test_df = pd.read_csv('src/data/test.csv')
        
        label_encoder = LabelEncoder()
        all_labels = pd.concat([train_df['label'], test_df['label']])
        label_encoder.fit(all_labels)
        
        y_test = label_encoder.transform(test_df['label'])
        
        try:
            with open('text_vectorizer_vocab.json', 'r') as f:
                vocab = json.load(f)
            
            with open('text_vectorizer_config.json', 'r') as f:
                config = json.load(f)
            
            print("Using saved text vectorizer configuration")
            
            text_vectorizer = keras.layers.TextVectorization(
                max_tokens=config['vocab_size'],
                output_sequence_length=config['max_length'],
                output_mode='int'
            )
            
            text_vectorizer.set_vocabulary(vocab)
            
        except FileNotFoundError:
            print("Saved vectorizer not found, creating new one...")
            text_vectorizer = keras.layers.TextVectorization(
                max_tokens=10000,
                output_sequence_length=100,
                output_mode='int'
            )
            
            train_texts = [str(text).lower() for text in train_df['text']]
            text_vectorizer.adapt(train_texts)
        
        test_texts = [str(text).lower() for text in test_df['text']]
        x_test_sequences = text_vectorizer(test_texts).numpy()
        
        print(f"Test sequences shape: {x_test_sequences.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Label classes: {label_encoder.classes_}")
        
        return x_test_sequences, y_test, label_encoder
    
    def test_cnn_models(self, num_samples: int = 1000) -> Dict:
        print("="*60)
        print("TESTING CNN MODELS")
        print("="*60)
        
        x_test, y_test = self.load_cifar10_test_data()
        
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
        
        cnn_results = {}
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            print(f"Models directory '{model_dir}' not found.")
            return {}
            
        cnn_models = [f for f in os.listdir(model_dir) if f.startswith('cnn_') and f.endswith('.h5')]
        
        if not cnn_models:
            print("No CNN models found. Please train models first.")
            return {}
        
        for model_file in cnn_models:
            model_path = os.path.join(model_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nTesting model: {model_name}")
            print("-" * 40)
            
            try:
                keras_model = keras.models.load_model(model_path)
                
                print("Getting Keras predictions...")
                start_time = time.time()
                keras_predictions = keras_model.predict(x_test, batch_size=32, verbose=0)
                keras_time = time.time() - start_time
                keras_pred_classes = np.argmax(keras_predictions, axis=1)
                
                keras_f1 = f1_score(y_test, keras_pred_classes, average='macro')
                
                print("Testing custom CNN implementation...")
                custom_cnn = CustomCNN(model_path)
                
                start_time = time.time()
                custom_predictions = custom_cnn.predict(x_test, batch_size=32)
                custom_time = time.time() - start_time
                custom_pred_classes = np.argmax(custom_predictions, axis=1)
                
                custom_f1 = f1_score(y_test, custom_pred_classes, average='macro')
                
                prediction_diff = np.abs(keras_predictions - custom_predictions)
                max_diff = np.max(prediction_diff)
                mean_diff = np.mean(prediction_diff)
                
                agreement = np.mean(keras_pred_classes == custom_pred_classes)
                
                cnn_results[model_name] = {
                    'keras_f1': keras_f1,
                    'custom_f1': custom_f1,
                    'f1_difference': abs(keras_f1 - custom_f1),
                    'max_prediction_diff': max_diff,
                    'mean_prediction_diff': mean_diff,
                    'prediction_agreement': agreement,
                    'keras_time': keras_time,
                    'custom_time': custom_time,
                    'speedup': keras_time / custom_time if custom_time > 0 else 0
                }
                
                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Custom F1-Score: {custom_f1:.4f}")
                print(f"F1 Difference: {abs(keras_f1 - custom_f1):.4f}")
                print(f"Prediction Agreement: {agreement:.4f}")
                print(f"Max Prediction Diff: {max_diff:.6f}")
                print(f"Mean Prediction Diff: {mean_diff:.6f}")
                print(f"Keras Time: {keras_time:.2f}s")
                print(f"Custom Time: {custom_time:.2f}s")
                
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                cnn_results[model_name] = {'error': str(e)}
        
        return cnn_results
    
    def test_rnn_models(self, num_samples: int = 500) -> Dict:
        print("="*60)
        print("TESTING RNN MODELS")
        print("="*60)
        
        try:
            x_test, y_test, label_encoder = self.load_text_test_data()
        except Exception as e:
            print(f"Error loading text data: {str(e)}")
            return {}
        
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
        
        rnn_results = {}
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            print(f"Models directory '{model_dir}' not found.")
            return {}
            
        rnn_models = [f for f in os.listdir(model_dir) if f.startswith('rnn_') and f.endswith('.h5')]
        
        if not rnn_models:
            print("No RNN models found. Please train models first.")
            return {}
        
        for model_file in rnn_models:
            model_path = os.path.join(model_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nTesting model: {model_name}")
            print("-" * 40)
            
            try:
                keras_model = keras.models.load_model(model_path)
                
                print("Getting Keras predictions...")
                start_time = time.time()
                keras_predictions = keras_model.predict(x_test, batch_size=32, verbose=0)
                keras_time = time.time() - start_time
                keras_pred_classes = np.argmax(keras_predictions, axis=1)
                
                keras_f1 = f1_score(y_test, keras_pred_classes, average='macro')
                
                print("Testing custom RNN implementation...")
                custom_rnn = CustomRNN(model_path)
                
                start_time = time.time()
                custom_predictions = custom_rnn.predict(x_test, batch_size=32)
                custom_time = time.time() - start_time
                custom_pred_classes = np.argmax(custom_predictions, axis=1)
                
                custom_f1 = f1_score(y_test, custom_pred_classes, average='macro')
                
                prediction_diff = np.abs(keras_predictions - custom_predictions)
                max_diff = np.max(prediction_diff)
                mean_diff = np.mean(prediction_diff)
                
                agreement = np.mean(keras_pred_classes == custom_pred_classes)
                
                rnn_results[model_name] = {
                    'keras_f1': keras_f1,
                    'custom_f1': custom_f1,
                    'f1_difference': abs(keras_f1 - custom_f1),
                    'max_prediction_diff': max_diff,
                    'mean_prediction_diff': mean_diff,
                    'prediction_agreement': agreement,
                    'keras_time': keras_time,
                    'custom_time': custom_time,
                    'speedup': keras_time / custom_time if custom_time > 0 else 0
                }
                
                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Custom F1-Score: {custom_f1:.4f}")
                print(f"F1 Difference: {abs(keras_f1 - custom_f1):.4f}")
                print(f"Prediction Agreement: {agreement:.4f}")
                print(f"Max Prediction Diff: {max_diff:.6f}")
                print(f"Mean Prediction Diff: {mean_diff:.6f}")
                print(f"Keras Time: {keras_time:.2f}s")
                print(f"Custom Time: {custom_time:.2f}s")
                
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                rnn_results[model_name] = {'error': str(e)}
        
        return rnn_results
    
    def test_lstm_models(self, num_samples: int = 500) -> Dict:
        print("="*60)
        print("TESTING LSTM MODELS")
        print("="*60)
        
        try:
            x_test, y_test, label_encoder = self.load_text_test_data()
        except Exception as e:
            print(f"Error loading text data: {str(e)}")
            return {}
        
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
        
        lstm_results = {}
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            print(f"Models directory '{model_dir}' not found.")
            return {}
            
        lstm_models = [f for f in os.listdir(model_dir) if f.startswith('lstm_') and f.endswith('.h5')]
        
        if not lstm_models:
            print("No LSTM models found. Please train models first.")
            return {}
        
        for model_file in lstm_models:
            model_path = os.path.join(model_dir, model_file)
            model_name = model_file.replace('.h5', '')
            
            print(f"\nTesting model: {model_name}")
            print("-" * 40)
            
            try:
                keras_model = keras.models.load_model(model_path)
                
                print("Getting Keras predictions...")
                start_time = time.time()
                keras_predictions = keras_model.predict(x_test, batch_size=32, verbose=0)
                keras_time = time.time() - start_time
                keras_pred_classes = np.argmax(keras_predictions, axis=1)
                
                keras_f1 = f1_score(y_test, keras_pred_classes, average='macro')
                
                print("Testing custom LSTM implementation...")
                custom_lstm = CustomLSTM(model_path)
                
                start_time = time.time()
                custom_predictions = custom_lstm.predict(x_test, batch_size=32)
                custom_time = time.time() - start_time
                custom_pred_classes = np.argmax(custom_predictions, axis=1)
                
                custom_f1 = f1_score(y_test, custom_pred_classes, average='macro')
                
                prediction_diff = np.abs(keras_predictions - custom_predictions)
                max_diff = np.max(prediction_diff)
                mean_diff = np.mean(prediction_diff)
                
                agreement = np.mean(keras_pred_classes == custom_pred_classes)
                
                # Store results
                lstm_results[model_name] = {
                    'keras_f1': keras_f1,
                    'custom_f1': custom_f1,
                    'f1_difference': abs(keras_f1 - custom_f1),
                    'max_prediction_diff': max_diff,
                    'mean_prediction_diff': mean_diff,
                    'prediction_agreement': agreement,
                    'keras_time': keras_time,
                    'custom_time': custom_time,
                    'speedup': keras_time / custom_time if custom_time > 0 else 0
                }
                
                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Custom F1-Score: {custom_f1:.4f}")
                print(f"F1 Difference: {abs(keras_f1 - custom_f1):.4f}")
                print(f"Prediction Agreement: {agreement:.4f}")
                print(f"Max Prediction Diff: {max_diff:.6f}")
                print(f"Mean Prediction Diff: {mean_diff:.6f}")
                print(f"Keras Time: {keras_time:.2f}s")
                print(f"Custom Time: {custom_time:.2f}s")
                
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                lstm_results[model_name] = {'error': str(e)}
        
        return lstm_results
    
    def create_comparison_plots(self, results: Dict):
        print("Creating comparison plots...")
        
        valid_results_exist = False
        for model_type in ['cnn', 'rnn', 'lstm']:
            if model_type in results and results[model_type]:
                for model_results in results[model_type].values():
                    if 'error' not in model_results:
                        valid_results_exist = True
                        break
                if valid_results_exist:
                    break
        
        if not valid_results_exist:
            print("No valid results to plot")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Custom vs Keras Implementation Comparison', fontsize=16)
        
        model_names = []
        keras_f1_scores = []
        custom_f1_scores = []
        f1_differences = []
        prediction_agreements = []
        time_comparisons = []
        
        for model_type in ['cnn', 'rnn', 'lstm']:
            if model_type in results and results[model_type]:
                for model_name, model_results in results[model_type].items():
                    if 'error' not in model_results:
                        model_names.append(f"{model_type}_{model_name}")
                        keras_f1_scores.append(model_results['keras_f1'])
                        custom_f1_scores.append(model_results['custom_f1'])
                        f1_differences.append(model_results['f1_difference'])
                        prediction_agreements.append(model_results['prediction_agreement'])
                        time_comparisons.append(model_results['custom_time'] / model_results['keras_time'])
        
        if not model_names:
            print("No valid results to plot")
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No valid test results available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            plot_path = os.path.join(self.test_results_dir, 'comparison_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, keras_f1_scores, width, label='Keras', alpha=0.8)
        ax.bar(x + width/2, custom_f1_scores, width, label='Custom', alpha=0.8)
        ax.set_xlabel('Models')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        bars = ax.bar(model_names, f1_differences)
        ax.set_xlabel('Models')
        ax.set_ylabel('F1-Score Difference')
        ax.set_title('F1-Score Difference (|Keras - Custom|)')
        ax.tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars):
            if f1_differences[i] < 0.01:
                bar.set_color('green')
            elif f1_differences[i] < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        bars = ax.bar(model_names, prediction_agreements)
        ax.set_xlabel('Models')
        ax.set_ylabel('Prediction Agreement')
        ax.set_title('Prediction Agreement Rate')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
        
        for i, bar in enumerate(bars):
            if prediction_agreements[i] > 0.95:
                bar.set_color('green')
            elif prediction_agreements[i] > 0.90:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        bars = ax.bar(model_names, time_comparisons)
        ax.set_xlabel('Models')
        ax.set_ylabel('Time Ratio (Custom/Keras)')
        ax.set_title('Inference Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.scatter(keras_f1_scores, custom_f1_scores, alpha=0.7, s=100)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')
        ax.set_xlabel('Keras F1-Score')
        ax.set_ylabel('Custom F1-Score')
        ax.set_title('Custom vs Keras F1-Score Scatter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, name in enumerate(model_names):
            ax.annotate(name.split('_')[-1], 
                       (keras_f1_scores[i], custom_f1_scores[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax = axes[1, 2]
        ax.axis('off')
        
        mean_f1_diff = np.mean(f1_differences)
        max_f1_diff = np.max(f1_differences)
        mean_agreement = np.mean(prediction_agreements)
        min_agreement = np.min(prediction_agreements)
        
        summary_text = f"""
        Summary Statistics:
        
        Mean F1 Difference: {mean_f1_diff:.4f}
        Max F1 Difference: {max_f1_diff:.4f}
        
        Mean Prediction Agreement: {mean_agreement:.4f}
        Min Prediction Agreement: {min_agreement:.4f}
        
        Models Tested: {len(model_names)}
        
        Success Criteria:
        ✓ F1 Difference < 0.01: {'PASS' if mean_f1_diff < 0.01 else 'FAIL'}
        ✓ Agreement > 0.95: {'PASS' if mean_agreement > 0.95 else 'FAIL'}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.test_results_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plots saved to: {plot_path}")
    
    def generate_test_report(self, results: Dict):
        print("Generating test report...")
        
        report_lines = []
        report_lines.append("# Custom Forward Propagation Test Report")
        report_lines.append("## IF3270 Pembelajaran Mesin - Tugas Besar 2")
        report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_models_tested = 0
        successful_tests = 0
        
        for model_type in ['cnn', 'rnn', 'lstm']:
            if model_type in results and results[model_type]:
                for model_name, model_results in results[model_type].items():
                    total_models_tested += 1
                    if 'error' not in model_results:
                        successful_tests += 1
        
        report_lines.append(f"## Test Summary")
        report_lines.append(f"- Total models tested: {total_models_tested}")
        report_lines.append(f"- Successful tests: {successful_tests}")
        
        if total_models_tested > 0:
            report_lines.append(f"- Success rate: {successful_tests/total_models_tested*100:.1f}%")
        else:
            report_lines.append("- Success rate: N/A (no models found)")
        report_lines.append("")
        
        for model_type in ['CNN', 'RNN', 'LSTM']:
            model_key = model_type.lower()
            
            if model_key in results and results[model_key]:
                report_lines.append(f"## {model_type} Models")
                report_lines.append("")
                
                for model_name, model_results in results[model_key].items():
                    report_lines.append(f"### {model_name}")
                    
                    if 'error' in model_results:
                        report_lines.append(f"**Error**: {model_results['error']}")
                    else:
                        f1_diff = model_results['f1_difference']
                        agreement = model_results['prediction_agreement']
                        
                        if f1_diff < 0.01 and agreement > 0.95:
                            status = "**PASS**"
                        elif f1_diff < 0.05 and agreement > 0.90:
                            status = "**WARNING**"
                        else:
                            status = "**FAIL**"
                        
                        report_lines.append(f"**Status**: {status}")
                        report_lines.append(f"- Keras F1-Score: {model_results['keras_f1']:.4f}")
                        report_lines.append(f"- Custom F1-Score: {model_results['custom_f1']:.4f}")
                        report_lines.append(f"- F1 Difference: {f1_diff:.4f}")
                        report_lines.append(f"- Prediction Agreement: {agreement:.4f}")
                        report_lines.append(f"- Max Prediction Diff: {model_results['max_prediction_diff']:.6f}")
                        report_lines.append(f"- Mean Prediction Diff: {model_results['mean_prediction_diff']:.6f}")
                        report_lines.append(f"- Keras Time: {model_results['keras_time']:.2f}s")
                        report_lines.append(f"- Custom Time: {model_results['custom_time']:.2f}s")
                        report_lines.append(f"- Speed Ratio: {model_results['custom_time']/model_results['keras_time']:.2f}x")
                    
                    report_lines.append("")
            else:
                report_lines.append(f"## {model_type} Models")
                report_lines.append("No models found for testing.")
                report_lines.append("")
        
        report_lines.append("## Conclusions")
        report_lines.append("")
        
        if total_models_tested == 0:
            report_lines.append("**No models available for testing**")
            report_lines.append("")
            report_lines.append("To run tests, please first train models using:")
            report_lines.append("```")
            report_lines.append("python src/pretrained_model/main_pretrained.py --model all")
            report_lines.append("```")
        elif successful_tests == total_models_tested:
            report_lines.append("**All models tested successfully!**")
        else:
            report_lines.append(f"**{total_models_tested - successful_tests} models failed testing**")
        
        if successful_tests > 0:
            report_lines.append("")
            report_lines.append("### Key Findings:")
            
            all_f1_diffs = []
            all_agreements = []
            
            for model_type in ['cnn', 'rnn', 'lstm']:
                if model_type in results and results[model_type]:
                    for model_name, model_results in results[model_type].items():
                        if 'error' not in model_results:
                            all_f1_diffs.append(model_results['f1_difference'])
                            all_agreements.append(model_results['prediction_agreement'])
            
            if all_f1_diffs:
                report_lines.append(f"- Average F1-Score difference: {np.mean(all_f1_diffs):.4f}")
                report_lines.append(f"- Maximum F1-Score difference: {np.max(all_f1_diffs):.4f}")
                report_lines.append(f"- Average prediction agreement: {np.mean(all_agreements):.4f}")
                report_lines.append(f"- Minimum prediction agreement: {np.min(all_agreements):.4f}")
                
                if np.mean(all_f1_diffs) < 0.01:
                    report_lines.append("- Custom implementations are **highly accurate**")
                elif np.mean(all_f1_diffs) < 0.05:
                    report_lines.append("- Custom implementations have **acceptable accuracy**")
                else:
                    report_lines.append("- Custom implementations need **improvement**")
        
        report_path = os.path.join(self.test_results_dir, 'test_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Test report saved to: {report_path}")
        
    def run_all_tests(self) -> Dict:
        print("Starting comprehensive forward propagation testing...")
        print(f"Test results will be saved to: {os.path.abspath(self.test_results_dir)}")
        
        all_results = {}
        
        try:
            cnn_results = self.test_cnn_models()
            all_results['cnn'] = cnn_results
        except Exception as e:
            print(f"Error testing CNN models: {str(e)}")
            all_results['cnn'] = {}
        
        try:
            rnn_results = self.test_rnn_models()
            all_results['rnn'] = rnn_results
        except Exception as e:
            print(f"Error testing RNN models: {str(e)}")
            all_results['rnn'] = {}
        
        try:
            lstm_results = self.test_lstm_models()
            all_results['lstm'] = lstm_results
        except Exception as e:
            print(f"Error testing LSTM models: {str(e)}")
            all_results['lstm'] = {}
        
        results_path = os.path.join(self.test_results_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json_results = {}
            for model_type, type_results in all_results.items():
                json_results[model_type] = {}
                if type_results:  # Check if not empty
                    for model_name, model_results in type_results.items():
                        if isinstance(model_results, dict):
                            json_results[model_type][model_name] = {
                                k: (float(v) if isinstance(v, np.floating) else v)
                                for k, v in model_results.items()
                            }
                        else:
                            json_results[model_type][model_name] = model_results
            
            json.dump(json_results, f, indent=2)
        
        self.create_comparison_plots(all_results)
        self.generate_test_report(all_results)
        
        print("="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)
        
        return all_results

def main():
    print("Forward Propagation Testing Suite")
    print("IF3270 Pembelajaran Mesin - Tugas Besar 2")
    print("="*60)
    
    tester = ForwardPropagationTester()
    
    results = tester.run_all_tests()
    
    print("\nTest Summary:")
    for model_type, type_results in results.items():
        if type_results:
            print(f"{model_type.upper()}: {len(type_results)} models tested")
            
            successful = sum(1 for r in type_results.values() if isinstance(r, dict) and 'error' not in r)
            print(f"  Successful: {successful}/{len(type_results)}")
        else:
            print(f"{model_type.upper()}: 0 models found")

if __name__ == "__main__":
    main()