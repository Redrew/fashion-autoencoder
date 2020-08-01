import torch as th
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
import torchvision.utils as vutils

from typing import Any

from utils import Config

DEFAULT_MODEL_CONFIG = {
    "layers": 6,
    "hidden_size": 256,
    "embedding_size": 32,
    "activation_fn": nn.Tanh
}

def fill_defaults(model_config):
    for v, k in DEFAULT_MODEL_CONFIG.items():
        if v not in model_config:
            model_config[v] = k
    return model_config

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

class VAE_Model(nn.Module):
    def __init__(self, shape, model_config):
        super().__init__()
        self.config = fill_defaults(model_config)
        layers = self.config.get("layers")
        hidden_size = self.config.get("hidden_size")
        embedding_size = self.config.get("embedding_size")
        activation_fn = self.config.get("activation_fn")

        assert layers % 2 == 0 and layers >= 4
        self.encoder = FCStack(shape, embedding_size, layers//2, hidden_size, activation_fn=activation_fn)
        self.decoder = FCStack(embedding_size, shape, layers//2, hidden_size, activation_fn=activation_fn)

    def forward(self, x):
        encoding = self.encoder(x)
        output = self.decoder(encoding)
        return output
        
class VAE_Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = Config(config)
        self.model = VAE_Model(784, config)

        self.train_size = len(self.train_dataloader().dataset)
    
    def forward(self, image):
        image_shape = image.shape
        x = image.flatten(1)
        output = self.model(x)
        loss = self._loss(output, x)
        l1_dif = self._l1_dif(output, x)
        l2_dif = self._l2_dif(output, x)
        matching_output = output.reshape(image_shape)
        return loss, l1_dif, l2_dif, matching_output

    def training_step(self, batch, batch_nb):
        x = batch
        loss, l1_dif, l2_dif, output = self.forward(x)

        input_output = vutils.make_grid(th.cat([x[0:2], output[0:2]]))
        log = {"loss/train": loss, "l1/train": l1_dif, 
            "l2/train": l2_dif, "input-output/train": input_output, 
            **self.model.state_dict()}
        return {'loss': loss, 'log': log}
    
    def on_train_end(self, logs):
        past_io = [log["input-output/train"] for log in logs]
        video = th.stack(past_io).unsqueeze(0)
        log = {"video-io/train": video}
        return log

    def _l1_dif(self, x, y):
        return th.mean(th.abs(x - y))
    
    def _l2_dif(self, x, y):
        return th.mean(th.square(x - y))

    def _loss(self, output, target):
        return self._l1_dif(output, target)
    
    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = th.optim.Adam(parameters, lr=self.config.learning_rate,
                                  weight_decay=self.config.weight_decay)
        scheduler = {'scheduler': th.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.learning_rate,
                                                                      steps_per_epoch=self.train_size//self.config.batch_size,
                                                                      epochs=self.config.max_epochs),
                     'interval': 'step', 'name': 'learning_rate'}
        
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.ToTensor()])
        labelled_dataset = torchvision.datasets.FashionMNIST(self.config.data_dir, transform=transform_train, download=True)
        class DropLabel:
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, index):
                data, label = self.dataset[index]
                return data
        dataset = DropLabel(labelled_dataset)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        return self.train_dataloader()

class Trainer:
    def __init__(self, config, logger=None):
        self.config = config
        self.device = "cuda" if self.gpus > 0 else "cpu"
        self.logger = logger
        self.past_logs = []
    
    def __getattribute__(self, name: str) -> Any:
        if name != "config" and name in self.config:
            return self.config[name]
        else:
            return super().__getattribute__(name)
    
    def call_method_all(self, iterable, name):
        for obj in iterable:
            method = getattr(obj, name)
            method()

    def fit(self, module):
        self.past_logs.clear()

        train_loader = module.train_dataloader()
        optimizers, schedulers = module.configure_optimizers()
        module = module.to(self.device)

        update_step = 0
        for epoch in range(self.max_epochs):
            for batch_nb, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                result = {}

                self.call_method_all(optimizers, "zero_grad")
                module_result = module.training_step(batch, batch_nb)
                result.update(module_result)
                loss = result["loss"]
                loss.backward()
                self.call_method_all(optimizers, "step")
                
                for scheduler_dict in schedulers:
                    scheduler = scheduler_dict["scheduler"]
                    scheduler.step()
                    result["learning_rate/train"] = scheduler.get_last_lr()[0]


                if batch_nb % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_nb * len(batch), len(train_loader.dataset),
                        100. * batch_nb / len(train_loader), loss.item()))
                    self.log_metrics(result["log"], update_step)
                update_step += 1
        
        if self.logger:
            self.logger.flush()
        
        if hasattr(module, "on_train_end"):
            log = module.on_train_end(self.past_logs)
            self.log_metrics(log, -1)
    
    def normalize_tensors(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, th.Tensor):
                v = v.detach()
                if v.shape.numel() == 1:
                    v = v.item()
                metrics[k] = v
        return metrics

    def log_metrics(self, metrics, step):
        self.normalize_tensors(metrics)
        if step >= 0: self.past_logs.append(metrics)

        if not self.logger: return
        for k, v in metrics.items():
            if "video" in k:
                self.logger.add_video(k, vid_tensor=v, fps=1)
                continue
            if hasattr(v, "shape"):
                if v.shape[0] == 3:
                    self.logger.add_image(k, v, step)
                else:
                    self.logger.add_histogram(k, v, step)
            if np.isscalar(v):
                self.logger.add_scalar(k, v, step)
    
    def test(self, module):
        test_loader = module.test_loader()

config = {"learning_rate": 3e-4, "batch_size": 128, "max_epochs": 20, "data_dir": "~/data", "weight_decay": 0, "log_interval": 50, "gpus": 1}

logger = SummaryWriter(comment="VAE")
trainer = Trainer(config, logger=logger)
classifier = VAE_Module(config)
trainer.fit(classifier)