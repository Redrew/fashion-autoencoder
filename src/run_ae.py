from tensorboardX import SummaryWriter

from .AE_module import AE_Module
from .trainer import Trainer

def main(config):
    logger = SummaryWriter(comment="VAE")
    trainer = Trainer(config, logger=logger)
    classifier = AE_Module(config)
    trainer.fit(classifier)