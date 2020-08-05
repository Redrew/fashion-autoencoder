import torch as th
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule

from .utils import Config
from .AE_model import AE_Model

class AE_Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = Config()
        self.config.update(config)
        self.model = AE_Model(784, config)

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
    
    # def on_train_end(self, logs):
    #     past_io = [log["input-output/train"] for log in logs]
    #     video = th.stack(past_io).unsqueeze(0)
    #     log = {"video-io/train": video}
    #     return log

    def _l1_dif(self, x, y):
        return th.mean(th.abs(x - y))
    
    def _l2_dif(self, x, y):
        return th.mean(th.square(x - y))

    def _loss(self, output, target):
        return self._l1_dif(output, target)
    
    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = th.optim.SGD(parameters, lr=self.config.learning_rate,
                                  weight_decay=self.config.weight_decay, momentum=0.9, nesterov=True)
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
    
    # def test_dataloader(self):
    #     return self.train_dataloader()