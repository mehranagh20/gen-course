import argparse
import json
import os
import time

import imageio
import numpy as np
import torch
import torch.utils.data as data

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'

    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):
        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                val = f'{val:.5f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)

        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)

    return log


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    os.makedirs(H.save_dir, exist_ok=True)
    H.logdir = os.path.join(H.save_dir, 'log')


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    # setup_mpi(H)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    H.devices = [int(x) for x in H.devices.split(',')]
    return H, logprint


def generate_for_NN(H, model, sampler, orig, initial, shape, fname, logprint):
    initial = initial[:shape[0]].cuda(device=H.devices[0])
    nns = sampler.sample(initial, model)
    batches = [sampler.sample_from_out(orig), nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def generate_images(H, model, sampler, orig, initial, shape, fname, logprint):
    initial = initial[:shape[0]].cuda(device=H.devices[0])
    nns = sampler.sample(initial, model)
    batches = [sampler.sample_from_out(orig), nns]

    temp_latent = torch.randn([shape[0], H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    for i in range(H.num_temperatures_visualize):
        temp_latent.normal_()
        batches.append(sampler.sample(temp_latent, model))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)

def unconditional_images_fix_first(H, model, sampler, shape, fname, logprint):
    batches = []
    temp_latent = torch.randn([shape[0], H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    for i in range(H.num_temperatures_visualize):
        temp_latent.normal_()
        temp_latent[:, :H.latent_dim//2] = temp_latent[1, :H.latent_dim//2]
        batches.append(sampler.sample(temp_latent, model))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)

def unconditional_images_fix_second(H, model, sampler, shape, fname, logprint):
    batches = []
    temp_latent = torch.randn([shape[0], H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    for i in range(H.num_temperatures_visualize):
        temp_latent.normal_()
        temp_latent[:, H.latent_dim//2:] = temp_latent[1, H.latent_dim//2:]
        batches.append(sampler.sample(temp_latent, model))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)

def unconditional_images_zero_second(H, model, sampler, shape, fname, logprint):
    batches = []
    temp_latent = torch.randn([shape[0], H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    for i in range(H.num_temperatures_visualize):
        temp_latent.normal_()
        temp_latent[:, H.latent_dim//2:] = min(1, i/10)
        batches.append(sampler.sample(temp_latent, model))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)

def unconditional_images_zero_first(H, model, sampler, shape, fname, logprint):
    batches = []
    temp_latent = torch.randn([shape[0], H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    for i in range(H.num_temperatures_visualize):
        temp_latent.normal_()
        temp_latent[:, :H.latent_dim//2] = min(1, i/10)
        batches.append(sampler.sample(temp_latent, model))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose(
        [0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)

def restore_params(model, path, map_ddp=True, map_cpu=False):
    if not path:
        return
    print('restoring the model')
    state_dict = torch.load(path, map_location='cpu' if map_cpu else None)
    if map_ddp:
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict)


def save_model(path, model, latents, optimizer, H):
    torch.save(model.state_dict(), f'{path}-model.th')
    torch.save(optimizer.state_dict(), f'{path}-opt.th')
    torch.save(latents, f'{path}-latents.th')
    # from_log = os.path.join(H.save_dir, 'log.jsonl')
    # to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    # subprocess.check_output(['cp', from_log, to_log])


class ZippedDataset(data.Dataset):

    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        # print(index, [len(x) for x in self.datasets])
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return len(self.datasets[0])


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f

def restore_log(path):
    if not path:
        return 0, 0
    loaded = [json.loads(l) for l in open(path)]
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return starting_epoch, iterate


def make_gif(H, model, sampler, fname, logprint):
    result = []
    lat1 = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    lat2 = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    # lat2[:, :H.latent_dim//2] = lat1[:, :H.latent_dim//2][:]
    num = 400
    for i in range(num):
        lat1 = torch.lerp(lat1, lat2, 1/400)
        # temp_latent[:, :H.latent_dim//2] = min(1, i/10)
        result.append(sampler.sample(lat1, model).squeeze())
    # lat1 = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    lat2 = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda(device=H.devices[0])
    # lat2[:, H.latent_dim//2:] = lat1[:, H.latent_dim//2:][:]
    num = 400
    for i in range(num):
        lat1 = torch.lerp(lat1, lat2, 1/400)
        # temp_latent[:, :H.latent_dim//2] = min(1, i/10)
        result.append(sampler.sample(lat1, model).squeeze())
    logprint(f'printing gif to {fname}')
    imageio.mimwrite(fname, result, fps=50)