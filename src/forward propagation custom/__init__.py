__version__ = "1.0.0"
__author__ = "004 026 112"
__email__ = "13522XXX@itb.ac.id"
__description__ = "Custom Forward Propagation Implementation for CNN, RNN, and LSTM"

try:
    from .cnn import CustomCNN
    from .rnn import CustomRNN  
    from .lstm import CustomLSTM
    
    from .utils import (
        ModelWeightLoader,
        ActivationFunctions,
        ConvolutionUtils,
        TextProcessingUtils,
        ValidationUtils,
        BatchProcessor,
        ConfigManager,
        MemoryProfiler
    )
    
    from .test_forward_propagation import ForwardPropagationTester
    
    from .main_custom import CustomForwardPropagationMain
    
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    print("This might be due to missing dependencies. Please check requirements.txt")

__all__ = [
    'CustomCNN',
    'CustomRNN', 
    'CustomLSTM',
    
    'ModelWeightLoader',
    'ActivationFunctions',
    'ConvolutionUtils',
    'TextProcessingUtils',
    'ValidationUtils',
    'BatchProcessor',
    'ConfigManager',
    'MemoryProfiler',
    
    'ForwardPropagationTester',
    'CustomForwardPropagationMain',
    
    '__version__',
    '__author__',
    '__description__',
]

def get_version():
    return __version__

def get_available_models():
    return ['CustomCNN', 'CustomRNN', 'CustomLSTM']

def check_dependencies():
    dependencies = {}
    
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        dependencies['numpy'] = None
    
    try:
        import tensorflow
        dependencies['tensorflow'] = tensorflow.__version__
    except ImportError:
        dependencies['tensorflow'] = None
    
    try:
        import pandas
        dependencies['pandas'] = pandas.__version__
    except ImportError:
        dependencies['pandas'] = None
    
    try:
        import sklearn
        dependencies['scikit-learn'] = sklearn.__version__
    except ImportError:
        dependencies['scikit-learn'] = None
    
    try:
        import matplotlib
        dependencies['matplotlib'] = matplotlib.__version__
    except ImportError:
        dependencies['matplotlib'] = None
    
    return dependencies

def print_package_info():
    print("="*60)
    print(f"Custom Forward Propagation Package v{__version__}")
    print("IF3270 Pembelajaran Mesin - Tugas Besar 2")
    print("="*60)
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print()
    
    print("Available Models:")
    for model in get_available_models():
        print(f"  ✓ {model}")
    print()
    
    print("Dependency Status:")
    deps = check_dependencies()
    for dep_name, version in deps.items():
        status = f"✓ {version}" if version else "✗ Not found"
        print(f"  {dep_name}: {status}")
    print()
    
    if all(deps.values()):
        print("All dependencies are satisfied!")
    else:
        missing = [name for name, version in deps.items() if not version]
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install missing dependencies using: pip install -r requirements.txt")
    
    print("="*60)

def create_model(model_type: str, model_path: str = None):
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return CustomCNN(model_path)
    elif model_type == 'rnn':
        return CustomRNN(model_path)
    elif model_type == 'lstm':
        return CustomLSTM(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Available types: {get_available_models()}")

def run_demo(model_type: str = 'all', validate: bool = False):
    try:
        demo = CustomForwardPropagationMain()
        
        if model_type.lower() == 'all':
            demo.run_complete_demonstration()
        else:
            if model_type.lower() == 'cnn':
                demo.demonstrate_cnn_custom()
            elif model_type.lower() == 'rnn':
                demo.demonstrate_rnn_custom() 
            elif model_type.lower() == 'lstm':
                demo.demonstrate_lstm_custom()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if validate:
                demo.run_validation_tests()
    
    except Exception as e:
        print(f"Error running demonstration: {str(e)}")
        print("Please make sure all dependencies are installed and models are trained.")

def run_tests():
    try:
        tester = ForwardPropagationTester()
        results = tester.run_all_tests()
        return results
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return None

if __name__ != "__main__":
    print(f"Custom Forward Propagation Package v{__version__} loaded successfully!")
    print("Use help(forward_propagation_custom) for more information.")

def _example_usage():
    example_code = ""
    return example_code

def help():
    print_package_info()
    print("\nQuick Start:")
    print(_example_usage())
    print("\nFor detailed documentation, see README.md")

__doc__ = {__version__}