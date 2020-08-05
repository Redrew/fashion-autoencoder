import torch.nn as nn

class FC(nn.Module):
    def __init__(self, in_size, out_size, initializer=None, bias_init=0, activation_fn=None):
        super().__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer:
            initializer(linear.weight)
        nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class FCStack(nn.Module):
    def __init__(self, in_size, out_size, num_layers, hidden_size=None, initializer=None, bias_init=0, activation_fn=None, last_activation=None):
        super().__init__()
        last_size = in_size
        layers = [] 
        for _ in range(num_layers-1):
            layers.append(FC(last_size, hidden_size, initializer, bias_init, activation_fn))
            last_size = hidden_size
        layers.append(FC(last_size, out_size, initializer, bias_init, last_activation))
        self._model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._model(x)