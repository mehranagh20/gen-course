import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from helpers import set_up_hyperparams, generate_for_NN, ZippedDataset, linear_warmup
from model import Model
from sampler import Sampler
import time


def training_step(targets, latents, model, optimizer, loss_fn):
    t0 = time.time()
    model.zero_grad()
    print(targets.shape, latents.shape)
    px_z = model(latents)
    loss = loss_fn(px_z, targets)
    loss.backward()
    optimizer.step()
    t1 = time.time()
    return t1 - t0, loss


def train(H, model, train_data, logger):
    epoch, iter_num = 0, 0
    sampler = Sampler(H, H.n_split)
    for to_vis in DataLoader(train_data, batch_size=8):
        break

    optimizer = AdamW(model.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))

    iter_times = []
    losses = []

    while True:
        for cur_epoch in range(epoch, epoch + H.imle_staleness):
            sampler.imle_sample(train_data, model, force_update=True, factor=H.force_factor)
            # save_latents_latest(H, split_ind, sampler.selected_latents)
            # sampler.calc_dists_existing(train_data, model)

            generate_for_NN(model.module, sampler, to_vis[0], sampler.selected_latents[0: 8], to_vis[0].shape,
                            f'{H.save_dir}/NN-samples-{iter_num}.png', logger)

            comb_dataset = ZippedDataset(train_data, TensorDataset(sampler.selected_latents))
            data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, pin_memory=True, shuffle=True)
            for ind, batch in enumerate(data_loader):
                x = batch[0][0].cuda()
                latents = batch[1][0]
                iter_time, loss = training_step(x[0], latents, model, optimizer, sampler.calc_loss)
                scheduler.step(cur_epoch)

                iter_times.append(iter_time)
                losses.append(loss)

                if iter_num % H.iters_per_print == 0:
                    logger(model=H.desc, type='train_loss', latest=loss, lr=scheduler.get_last_lr()[0], epoch=cur_epoch,
                           step=iter_num, average_time=np.mean(iter_times), loss=np.mean)

                iter_num = iter_num + 1


def main():
    H, logger = set_up_hyperparams()
    model = Model(H).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    train_data = ImageFolder(H.data_root, transforms.ToTensor())
    n_split = H.n_split
    if n_split == -1:
        n_split = len(train_data)

    for data_train in DataLoader(train_data, batch_size=H.n_split):
        train_data = TensorDataset((data_train[0] - 0.5) * 2)
        break

    train(H, model, train_data, logger)


if __name__ == "__main__":
    main()
