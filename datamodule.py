
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import lightning as L


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, train_transforms, test_transforms, data_dir, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.data_dir = data_dir
        self.num_workers = num_workers
        # self.pin_memory = pin_memory

        # self.transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])

        # self.transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=self.train_transforms)
            self.val_dataset = datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.test_transforms)

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
