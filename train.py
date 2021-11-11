import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from helpers import set_up_hyperparams
from model import Model


def main():
    H, logger = set_up_hyperparams()
    model = Model(H).cuda()
    model = torch.nn.DataParallel(model)
    train_data = ImageFolder(H.data_root, transforms.ToTensor())
    for data_train in DataLoader(train_data, batch_size=100):
        with torch.no_grad():
            res = model(data_train)
            print(res)
        break

if __name__ == "__main__":
    main()
