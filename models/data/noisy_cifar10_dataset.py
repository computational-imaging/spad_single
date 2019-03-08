import torch
import torchvision
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, random_split

from sacred import Ingredient

noisy_cifar10_ingredient = Ingredient("data_config")

@noisy_cifar10_ingredient.config
def cfg():
    data_name = "cifar10"
    data_root = "./cifar10"
    sigma = 0.05
    seed = 1337

@noisy_cifar10_ingredient.capture
def load_data(data_root, sigma, seed):
    """
    Load CIFAR10 with a random data split
    (well, not so random if the random seed has been set)
    :param data_root: Location to save the dataset
    :param sigma: standard deviation of the noise to add to it.
    :return:
    """
    torch.manual_seed(seed)
    full_train = NoisyCIFAR10Dataset(data_root, sigma, train=True)
    # 90-10 split for train-val
    train_size = int(0.9*len(full_train))
    val_size = len(full_train) - train_size
    train, val = random_split(full_train, [train_size, val_size])
    test = NoisyCIFAR10Dataset(data_root, sigma, train=False)
    return train, val, test


class NoisyCIFAR10Dataset(Dataset):
    '''Dataset class that adds noise to cifar10-images'''
    def __init__(self,
                 data_root,
                 sigma,
                 train):
        super(NoisyCIFAR10Dataset, self).__init__()

        self.sigma = sigma

        self.transforms = Compose([
            ToTensor()
        ])

        self.cifar10 = torchvision.datasets.CIFAR10(root=data_root,
                                                    train=train,
                                                    download=True,
                                                    transform=self.transforms)

    def __len__(self):
        return len(self.cifar10)

    def add_noise(self, img):
        return img + torch.randn_like(img) * self.sigma

    def __getitem__(self, idx):
        '''Returns tuple of (model_input, ground_truth)'''
        img, _ = self.cifar10[idx]

        img = (img - 0.5) * 2
        noisy_img = self.add_noise(img)

        return noisy_img, img
