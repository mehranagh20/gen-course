import imageio

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
import argparse
import os
import torch
import numpy as np
import time
import json


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
    return H, logprint


def generate_for_NN(model, sampler, orig, initial, shape, fname, logprint):
    initial = initial[:shape[0]].cuda()
    nns = sampler.sample(initial, model)
    batches = [sampler.sample_from_out(orig), nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, shape[0], shape[2], shape[2], 3)).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[2], shape[0] * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
