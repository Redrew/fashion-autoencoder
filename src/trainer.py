import torch as th
import numpy as np
from typing import Any

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
        self.log_hparams(module)

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
    
    def log_hparams(self, module):
        hparams = {}
        hparams.update(module.config)
        self.logger.add_hparams(hparams, {"dummy": 0})
    
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
