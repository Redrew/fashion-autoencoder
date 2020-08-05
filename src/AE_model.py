import torch.nn as nn
from .fcstack import FCStack

DEFAULT_MODEL_CONFIG = {
    "layers": 6,
    "hidden_size": 256,
    "embedding_size": 32,
    "activation_fn": "nn.Tanh"
}

def fill_defaults(model_config):
    for v, k in DEFAULT_MODEL_CONFIG.items():
        if v not in model_config:
            model_config[v] = k
    return model_config

class AE_Model(nn.Module):
    def __init__(self, shape, model_config):
        super().__init__()
        self.config = fill_defaults(model_config)
        layers = self.config.get("layers")
        hidden_size = self.config.get("hidden_size")
        embedding_size = self.config.get("embedding_size")
        activation_fn = self.config.get("activation_fn")
        initializer = self.config.get("initializer")
        if isinstance(activation_fn, str): activation_fn = eval(activation_fn)
        if isinstance(initializer, str): initializer = eval(initializer)

        assert layers % 2 == 0 and layers >= 4
        self.encoder = FCStack(shape, embedding_size, layers//2, hidden_size, 
                               activation_fn=activation_fn, initializer=initializer)
        self.decoder = FCStack(embedding_size, shape, layers//2, hidden_size, 
                               activation_fn=activation_fn, initializer=initializer)

    def forward(self, x):
        encoding = self.encoder(x)
        output = self.decoder(encoding)
        return output