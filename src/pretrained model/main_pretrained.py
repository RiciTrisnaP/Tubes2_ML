import sys
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append('src/pretrained model')

from pretrainedcnn import PretrainedCNN
from pretrainedrnn import PretrainedRNN
from pretrainedlstm import PretrainedLSTM

class MainPretrained:
    def __init__(self):
        self.results_dir = 'results'
        self.models_dir = 'models'
        self.create_directories()
        
    def create_directories(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def run_cnn_experiments(self):
        print("="*60)
        print("STARTING CNN EXPERIMENTS ON CIFAR-10")
        print("="*60)
        
        try:
            cnn = PretrainedCNN()
            cnn_results = cnn.run_experiments()
            cnn.plot_training_curves(cnn_results)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.results_dir, f'cnn_results_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json_results = {}
                for key, value in cnn_results.items():
                    json_results[key] = {
                        k: (v.tolist() if hasattr(v, 'tolist') else v) 
                        for k, v in value.items() if k != 'report'
                    }
                json.dump(json_results, f, indent=2)
            
            print(f"\nCNN results saved to: {results_file}")
            return cnn_results
            
        except Exception as e:
            print(f"Error in CNN experiments: {str(e)}")
            return None
    
    def run_rnn_experiments(self):
        print("="*60)
        print("STARTING RNN EXPERIMENTS ON TEXT CLASSIFICATION")
        print("="*60)
        
        try:
            rnn = PretrainedRNN()
            rnn_results = rnn.run_experiments()
            rnn.plot_training_curves(rnn_results)
            rnn.save_vectorizer()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.results_dir, f'rnn_results_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json_results = {}
                for key, value in rnn_results.items():
                    json_results[key] = {
                        k: (v.tolist() if hasattr(v, 'tolist') else v) 
                        for k, v in value.items() if k != 'report'
                    }
                json.dump(json_results, f, indent=2)
            
            print(f"\nRNN results saved to: {results_file}")
            return rnn_results
            
        except Exception as e:
            print(f"Error in RNN experiments: {str(e)}")
            return None
    
    def run_lstm_experiments(self):
        print("="*60)
        print("STARTING LSTM EXPERIMENTS ON TEXT CLASSIFICATION")
        print("="*60)
        
        try:
            lstm = PretrainedLSTM()
            lstm_results = lstm.run_experiments()
            lstm.plot_training_curves(lstm_results)
            lstm.save_vectorizer()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.results_dir, f'lstm_results_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json_results = {}
                for key, value in lstm_results.items():
                    json_results[key] = {
                        k: (v.tolist() if hasattr(v, 'tolist') else v) 
                        for k, v in value.items() if k != 'report'
                    }
                json.dump(json_results, f, indent=2)
            
            print(f"\nLSTM results saved to: {results_file}")
            return lstm_results
            
        except Exception as e:
            print(f"Error in LSTM experiments: {str(e)}")
            return None
    
    def compare_results(self, cnn_results, rnn_results, lstm_results):
        print("="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        if cnn_results:
            for exp_name, exp_data in cnn_results.items():
                comparison_data.append({
                    'Model': 'CNN',
                    'Experiment': exp_name,
                    'F1_Score': exp_data['f1_macro'],
                    'Task': 'Image Classification (CIFAR-10)'
                })
        
        if rnn_results:
            for exp_name, exp_data in rnn_results.items():
                comparison_data.append({
                    'Model': 'RNN',
                    'Experiment': exp_name,
                    'F1_Score': exp_data['f1_macro'],
                    'Task': 'Text Classification (Sentiment)'
                })
        
        if lstm_results:
            for exp_name, exp_data in lstm_results.items():
                comparison_data.append({
                    'Model': 'LSTM',
                    'Experiment': exp_name,
                    'F1_Score': exp_data['f1_macro'],
                    'Task': 'Text Classification (Sentiment)'
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            comparison_file = os.path.join(self.results_dir, 'model_comparison.csv')
            df_comparison.to_csv(comparison_file, index=False)
            
            print("\nF1-Score Summary by Model:")
            print("-" * 40)
            print(df_comparison.groupby('Model')['F1_Score'].agg(['mean', 'std', 'min', 'max']).round(4))
            
            self.plot_model_comparison(df_comparison)
            
            return df_comparison
        
        return None
    
    def plot_model_comparison(self, df_comparison):
        plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        ax1 = axes[0, 0]
        df_comparison.boxplot(column='F1_Score', by='Model', ax=ax1)
        ax1.set_title('F1-Score Distribution by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('F1-Score')
        
        ax2 = axes[0, 1]
        best_scores = df_comparison.groupby('Model')['F1_Score'].max()
        best_scores.plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Best F1-Score per Model')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=0)
        
        ax3 = axes[1, 0]
        text_models = df_comparison[df_comparison['Model'].isin(['RNN', 'LSTM'])]
        if not text_models.empty:
            sns.barplot(data=text_models, x='Experiment', y='F1_Score', hue='Model', ax=ax3)
            ax3.set_title('RNN vs LSTM Performance')
            ax3.tick_params(axis='x', rotation=45)
        
        ax4 = axes[1, 1]
        model_means = df_comparison.groupby('Model')['F1_Score'].mean()
        model_stds = df_comparison.groupby('Model')['F1_Score'].std()
        
        x_pos = range(len(model_means))
        ax4.bar(x_pos, model_means.values, yerr=model_stds.values, 
                capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax4.set_title('Average F1-Score with Standard Deviation')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('F1-Score')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_means.index)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plots saved to: {plot_file}")
    
    def generate_report(self, cnn_results, rnn_results, lstm_results):
        print("="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_lines = []
        report_lines.append("# IF3270 Pembelajaran Mesin - Tugas Besar 2")
        report_lines.append("## Pretrained Models Experiment Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if cnn_results:
            report_lines.append("## CNN Experiments (CIFAR-10 Image Classification)")
            report_lines.append("")
            report_lines.append("### Results Summary:")
            for exp_name, exp_data in cnn_results.items():
                report_lines.append(f"- **{exp_name}**: F1-Score = {exp_data['f1_macro']:.4f}")
            report_lines.append("")
            
            best_cnn = max(cnn_results.items(), key=lambda x: x[1]['f1_macro'])
            report_lines.append(f"**Best CNN Model**: {best_cnn[0]} with F1-Score = {best_cnn[1]['f1_macro']:.4f}")
            report_lines.append("")
        
        if rnn_results:
            report_lines.append("## RNN Experiments (Text Sentiment Classification)")
            report_lines.append("")
            report_lines.append("### Results Summary:")
            for exp_name, exp_data in rnn_results.items():
                report_lines.append(f"- **{exp_name}**: F1-Score = {exp_data['f1_macro']:.4f}")
            report_lines.append("")
            
            best_rnn = max(rnn_results.items(), key=lambda x: x[1]['f1_macro'])
            report_lines.append(f"**Best RNN Model**: {best_rnn[0]} with F1-Score = {best_rnn[1]['f1_macro']:.4f}")
            report_lines.append("")
        
        if lstm_results:
            report_lines.append("## LSTM Experiments (Text Sentiment Classification)")
            report_lines.append("")
            report_lines.append("### Results Summary:")
            for exp_name, exp_data in lstm_results.items():
                report_lines.append(f"- **{exp_name}**: F1-Score = {exp_data['f1_macro']:.4f}")
            report_lines.append("")
            
            best_lstm = max(lstm_results.items(), key=lambda x: x[1]['f1_macro'])
            report_lines.append(f"**Best LSTM Model**: {best_lstm[0]} with F1-Score = {best_lstm[1]['f1_macro']:.4f}")
            report_lines.append("")
        
        if cnn_results and rnn_results and lstm_results:
            report_lines.append("## Overall Comparison")
            report_lines.append("")
            
            if rnn_results and lstm_results:
                avg_rnn = sum(exp['f1_macro'] for exp in rnn_results.values()) / len(rnn_results)
                avg_lstm = sum(exp['f1_macro'] for exp in lstm_results.values()) / len(lstm_results)
                
                report_lines.append(f"- **Average RNN F1-Score**: {avg_rnn:.4f}")
                report_lines.append(f"- **Average LSTM F1-Score**: {avg_lstm:.4f}")
                
                if avg_lstm > avg_rnn:
                    report_lines.append("- **Winner for Text Classification**: LSTM")
                else:
                    report_lines.append("- **Winner for Text Classification**: RNN")
                report_lines.append("")
        
        report_file = os.path.join(self.results_dir, 'experiment_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Report saved to: {report_file}")
        
        print("\n" + "\n".join(report_lines))
    
    def run_all_experiments(self):
        print("Starting all pretrained model experiments...")
        print(f"Results will be saved to: {os.path.abspath(self.results_dir)}")
        print(f"Models will be saved to: {os.path.abspath(self.models_dir)}")
        print()
        
        cnn_results = self.run_cnn_experiments()
        rnn_results = self.run_rnn_experiments()
        lstm_results = self.run_lstm_experiments()
        
        comparison_df = self.compare_results(cnn_results, rnn_results, lstm_results)
        
        self.generate_report(cnn_results, rnn_results, lstm_results)
        
        print("="*60)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*60)
        
        return {
            'cnn': cnn_results,
            'rnn': rnn_results,
            'lstm': lstm_results,
            'comparison': comparison_df
        }

def main():
    parser = argparse.ArgumentParser(description='Run pretrained model experiments')
    parser.add_argument('--model', choices=['cnn', 'rnn', 'lstm', 'all'], default='all',
                        help='Which model to run (default: all)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    main_exp = MainPretrained()
    main_exp.results_dir = args.output_dir
    main_exp.create_directories()
    
    if args.model == 'all':
        main_exp.run_all_experiments()
    elif args.model == 'cnn':
        main_exp.run_cnn_experiments()
    elif args.model == 'rnn':
        main_exp.run_rnn_experiments()
    elif args.model == 'lstm':
        main_exp.run_lstm_experiments()

if __name__ == "__main__":
    main()