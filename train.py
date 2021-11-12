import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from helpers import set_up_hyperparams, generate_for_NN
from model import Model
from sampler import Sampler


def train(H, model, train_data, logger):
    sampler = Sampler(H, H.n_split)
    sampler.imle_sample(train_data, model, force_update=True, factor=H.force_factor)
    # save_latents_latest(H, split_ind, sampler.selected_latents)
    sampler.calc_dists_existing(train_data, model)

    for ind, x in enumerate(DataLoader(train_data, batch_size=8)):
        generate_for_NN(model.module, sampler, x[0], sampler.selected_latents[0: 8], x[0].shape,
                        f'{H.save_dir}/NN-samples-{ind}.png', logger)
        if ind > 5:
            break


def main():
    H, logger = set_up_hyperparams()
    model = Model(H).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    train_data = ImageFolder(H.data_root, transforms.ToTensor())
    from torchvision.utils import save_image

    for data_train in DataLoader(train_data, batch_size=H.n_split):
        print(data_train[0].shape)
        train_data = TensorDataset((data_train[0] - 0.5) * 2)
        # save_image(data_train[0][0], 'file.png')
        break
    train(H, model, train_data, logger)


if __name__ == "__main__":
    main()
