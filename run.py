import torch.nn as nn
from src import run_ae
config = {"learning_rate": 1e-2, "batch_size": 128, "max_epochs": 10, 
          "layers": 4,
          "data_dir": "~/data", "weight_decay": 0, "log_interval": 50, "gpus": 1}

run_ae.main(config)