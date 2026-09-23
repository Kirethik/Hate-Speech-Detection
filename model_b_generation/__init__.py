import warnings

# Suppress the PyNVML future warning from PyTorch
warnings.filterwarnings(
    "ignore", 
    message="The pynvml package is deprecated", 
    category=FutureWarning, 
    module="torch.cuda"
)
