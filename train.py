import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from helpers import set_up_hyperparams, generate_for_NN, ZippedDataset, linear_warmup, generate_images, save_model, \
    restore_params, unconditional_images_fix_first, unconditional_images_fix_second, unconditional_images_zero_second, \
    restore_log, make_gif
from model import Model
from sampler import Sampler


def training_step(targets, latents, model, optimizer, loss_fn):
    t0 = time.time()
    model.zero_grad()
    px_z = model(latents)
    loss = loss_fn(px_z, targets)
    loss.backward()
    optimizer.step()
    t1 = time.time()
    return t1 - t0, loss.item()


def train(H, model, train_data, logger, sampler):
    epoch, iter_num = restore_log(H.restore_log_path)
    print('epoch: {}, iter_num: {}'.format(epoch, iter_num))
    starting_epoch = epoch
    for to_vis in DataLoader(train_data, batch_size=8):
        break

    optimizer = AdamW(model.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))

    iter_times = []
    losses = []

    while True:
        sampler.calc_dists_existing(train_data, model)
        sampler.first_phase(train_data, model, force_update=True, factor=H.force_factor)
        sampler.second_phase(train_data, model, 2)
        # save_latents_latest(H, split_ind, sampler.selected_latents)
        generate_for_NN(H, model.module, sampler, to_vis[0], sampler.selected_latents[0: 8], to_vis[0].shape,
                        f'{H.save_dir}/NN-samples-{iter_num}.png', logger)
        if epoch == starting_epoch:
            if H.restore_latent_path:
                print('restoring latent codes')
                sampler.selected_latents[:] = torch.load(H.restore_latent_path)[:]
            elif epoch == 0:
                print('random latents')
                sampler.selected_latents.normal_()

        # print('yo', sampler.selected_latents.shape)
        # for i in range(10):
        #     cur = sampler.selected_latents[i]
        #     print('min: {}, max: {}, mean: {}, std: {}'.format(torch.min(cur), torch.max(cur), torch.mean(cur), torch.std(cur)))
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

                if iter_num % H.iters_per_save == 0:
                    fp = os.path.join(H.save_dir, 'latest')
                    logger(f'Saving model@ {iter_num} to {fp}')
                    save_model(fp, model, sampler.selected_latents, optimizer, H)

                if iter_num % H.iters_per_ckpt == 0:
                    save_model(os.path.join(H.save_dir, f'iter-{iter_num}'), model, sampler.selected_latents, optimizer,
                               H)


def main():
    H, logger = set_up_hyperparams()
    model = Model(H).cuda(device=H.devices[0])
    train_data1 = ImageFolder(H.data_root, transforms.ToTensor())
    train_data2 = ImageFolder(H.data_root2, transforms.ToTensor())
    train_data = None

    n_split = H.n_split
    if n_split == -1:
        n_split = len(train_data)
    for first in DataLoader(train_data1, batch_size=n_split // 2):
        break
    for second in DataLoader(train_data2, batch_size=n_split // 2):
        break
    train_data = ConcatDataset([TensorDataset((first[0] - 0.5) * 2), TensorDataset((second[0] - 0.5) * 2)])
    for train_data in DataLoader(train_data, batch_size=len(train_data), shuffle=True):
        train_data = TensorDataset(train_data[0])
        break

    sampler = Sampler(H, H.n_split)

    restore_params(model, H.restore_path, map_cpu=True)
    if H.test_eval:
        for to_vis in DataLoader(train_data, batch_size=12):
            break
        model = torch.nn.DataParallel(model, device_ids=H.devices)

        make_gif(H, model, sampler, f'{H.save_dir}/gif.png', logger)
        unconditional_images_fix_first(H, model, sampler, to_vis[0].shape, f'{H.save_dir}/first-{H.fname}', logger)
        unconditional_images_fix_second(H, model, sampler, to_vis[0].shape, f'{H.save_dir}/second-{H.fname}', logger)
        unconditional_images_zero_second(H, model, sampler, to_vis[0].shape, f'{H.save_dir}/zero-secodn-{H.fname}',
                                         logger)
        unconditional_images_zero_second(H, model, sampler, to_vis[0].shape, f'{H.save_dir}/zero-first-{H.fname}',
                                         logger)

    else:
        model = torch.nn.DataParallel(model, device_ids=H.devices)
        train(H, model, train_data, logger, sampler)


if __name__ == "__main__":
    main()
