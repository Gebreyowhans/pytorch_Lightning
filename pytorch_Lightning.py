import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from datamodule import CIFAR10DataModule

L.seed_everything(42)


class CIFAR10CNN(L.LightningModule):
    def __init__(self, model_params):
        super(CIFAR10CNN, self).__init__()

        self.model_params = model_params
        self.learning_rate = self.model_params['learning_rate']
        self.factor = self.model_params['scheduler']['factor']
        self.patience = self.model_params['scheduler']['patience']
        self.optimizer_name = self.model_params['optimizer']
        self.mode = self.model_params['scheduler'].get('mode', 'min')

        # verbose = self.model_params['scheduler'].get('verbose', False)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Writing a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        # Log the loss at each training step and epoch, create a progress bar
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        self.log("train_acc", acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):

        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.mode, factor=self.factor, patience=self.patience)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def main():

    # Load the configuration file
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters
    model_params = config['model']
    training_params = config['training']
    callback_params = config['callbacks']
    logger_params = config['logger']
    data_params = config['data']

    num_epochs = training_params['num_epochs']
    batch_size = training_params['batch_size']

    # Data module with transforms
    train_transforms = transforms.Compose([
        eval(transform) for transform in data_params['train_transforms']
    ])
    test_transforms = transforms.Compose([
        eval(transform) for transform in data_params['test_transforms']
    ])

    # dataset module
    data_module = CIFAR10DataModule(
        batch_size=batch_size,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        data_dir=data_params['data_dir'],
        num_workers=data_params['num_workers'],
        # pin_memory=data_params['pin_memory']
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(**callback_params['checkpoint'])
    early_stopping = EarlyStopping(**callback_params['early_stopping'])

    # Initialize the logger
    logger = TensorBoardLogger(
        save_dir=logger_params['save_dir'],
        name=logger_params['name']
    )

    # Initialize the model
    model = CIFAR10CNN(model_params)

    # Trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision=training_params['precision']
        # learning_rate=learning_rate,
        # accumulate_grad_batches=training_params['gradient_accumulation'],
        # fast_dev_run=training_params['fast_dev_run'],
        # overfit_batches=training_params['overfit_batches']
    )
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
# %% [markdown]
