import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from helpers import set_up_hyperparams, generate_for_NN, ZippedDataset, linear_warmup, generate_images
from model import Model
from sampler import Sampler
import time


def training_step(targets, latents, model, optimizer, loss_fn):
    t0 = time.time()
    model.zero_grad()
    px_z = model(latents)
    loss = loss_fn(px_z, targets)
    loss.backward()
    optimizer.step()
    t1 = time.time()
    return t1 - t0, loss.item()


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
        sampler.calc_dists_existing(train_data, model)
        sampler.imle_sample(train_data, model, force_update=True, factor=H.force_factor)
        # save_latents_latest(H, split_ind, sampler.selected_latents)
        generate_for_NN(H, model.module, sampler, to_vis[0], sampler.selected_latents[0: 8], to_vis[0].shape,
                        f'{H.save_dir}/NN-samples-{iter_num}.png', logger)


        for epoch in range(epoch, epoch + H.imle_staleness):
            comb_dataset = ZippedDataset(train_data, TensorDataset(sampler.selected_latents))
            data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, pin_memory=True, shuffle=True)
            for ind, batch in enumerate(data_loader):
                x = batch[0][0].cuda(device=H.devices[0])
                latents = batch[1][0]
                iter_time, loss = training_step(x, latents, model, optimizer, sampler.calc_loss)
                scheduler.step()
                iter_num = iter_num + 1

                iter_times.append(iter_time)
                losses.append(loss)
                if len(iter_times) > 1000:
                    iter_times = iter_times[1:]
                    losses = losses[1:]

                if iter_num % H.iters_per_print == 0:
                    logger(model=H.desc, type='train_loss', latest=loss, lr=scheduler.get_last_lr()[0], epoch=epoch,
                           step=iter_num, average_time=np.mean(iter_times), loss=np.mean(losses))
                if iter_num % H.iters_per_images == 0:
                    generate_images(H, model, sampler, to_vis[0],
                                            sampler.selected_latents[0: H.num_images_visualize],
                                            to_vis[0].shape, f'{H.save_dir}/samples-{iter_num}.png', logger)


def main():
    H, logger = set_up_hyperparams()
    model = Model(H).cuda(device=H.devices[0])
    model = torch.nn.DataParallel(model, device_ids=H.devices)
    train_data = ImageFolder(H.data_root, transforms.ToTensor())
    n_split = H.n_split
    if n_split == -1:
        n_split = len(train_data)

    for data_train in DataLoader(train_data, batch_size=n_split):
        train_data = TensorDataset((data_train[0] - 0.5) * 2)
        break

    train(H, model, train_data, logger)


if __name__ == "__main__":
    main()
