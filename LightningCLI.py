from pytorch_lightning.cli import LightningCLI
from datamodule import CIFAR10DataModule
from pytorch_lightning import CIFAR10CNN


# Initialize the CLI
# if __name__ == "__main__":
#     cli = LightningCLI(model, CIFAR10DataModule)

# python main.py --trainer.max_epochs=10 --data.batch_size=32 --data.data_dir=/path/to/data
