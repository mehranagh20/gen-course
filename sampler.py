import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from LPNet import LPNet
from dciknn_cuda.dciknn_cuda import DCI


class Sampler:
    def __init__(self, H, sz):
        self.l2_loss = torch.nn.MSELoss(reduce=False).cuda(device=H.devices[0])
        self.H = H
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_dists = torch.empty([sz], dtype=torch.float32).cuda(device=H.devices[0])
        self.selected_latents_future = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_dists_future = torch.empty([sz], dtype=torch.float32).cuda(device=H.devices[0])
        self.selected_dists_future[:] = np.inf
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32).cuda(device=H.devices[0])
        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net).cuda(device=H.devices[0])

        fake = torch.zeros(1, 3, H.image_size, H.image_size).cuda(device=H.devices[0])
        out, shapes = self.lpips_net(fake)
        dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
        if H.proj_proportion:
            sm = sum([dim.shape[1] for dim in out])
            dims = [int(feat.shape[1] * (H.proj_dim / sm)) for feat in out]
        print(dims)
        for ind, feat in enumerate(out):
            print(feat.shape)
            self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind]), p=2, dim=1).cuda(device=H.devices[0]))

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum(dims)], dtype=torch.float32).cuda(device=H.devices[0])
        self.dataset_proj = torch.empty([sz, sum(dims)], dtype=torch.float32)

    def get_projected(self, inp):
        out, _ = self.lpips_net(inp.cuda(device=self.H.devices[0]))
        gen_feat = []
        for i in range(len(out)):
            gen_feat.append(torch.mm(out[i], self.projections[i]))
        return torch.cat(gen_feat, dim=1)

    def init_projection(self):
        for proj_mat in self.projections:
            proj_mat[:] = F.normalize(torch.randn(proj_mat.shape), p=2, dim=1)

    def sample(self, latents, model):
        with torch.no_grad():
            px_z = model(latents).permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def sample_from_out(self, px_z):
        with torch.no_grad():
            px_z = px_z.permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def calc_loss(self, inp, tar, use_mean=True, l2_coef=None, lpips_coef=None):
        inp_feat, inp_shape = self.lpips_net(inp)
        tar_feat, _ = self.lpips_net(tar)
        res = 0

        if l2_coef is None:
            l2_coef = self.H.l2_coef
        if lpips_coef is None:
            lpips_coef = self.H.lpips_coef

        for i, g_feat in enumerate(inp_feat):
            res += torch.sum((g_feat - tar_feat[i]) ** 2, dim=1) / (inp_shape[i] ** 2)
        if use_mean:
            return lpips_coef * res.mean() + l2_coef * self.l2_loss(inp, tar).mean()
        else:
            return lpips_coef * res + l2_coef * torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])

    def calc_dists_existing(self, dataset, model, dists=None, latents=None):
        if dists is None:
            dists = self.selected_dists
        if latents is None:
            latents = self.selected_latents
        self.init_projection()
        for ind, x in enumerate(DataLoader(dataset, batch_size=self.H.n_batch)):
            if ind % 1000 == 0:
                print('finished updating dists for', ind * self.H.n_batch)
            batch_slice = slice(ind * self.H.n_batch, (ind + 1) * self.H.n_batch)
            cur_latents = latents[batch_slice].cuda(device=self.H.devices[0])
            self.dataset_proj[batch_slice] = self.get_projected(x[0])
            with torch.no_grad():
                out = model(cur_latents)
                dist = self.calc_loss(x[0].cuda(device=self.H.devices[0]), out, use_mean=False)
                dists[batch_slice] = torch.squeeze(dist)

    def first_phase(self, dataset, model, force_update=False, factor=-1, update_projection=False):
        if force_update or update_projection:
            self.init_projection()
            for ind, x in enumerate(DataLoader(dataset, batch_size=self.H.imle_batch)):
                if ind % 100 == 0:
                    print(ind)
                batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x[0].shape[0])
                self.dataset_proj[batch_slice] = self.get_projected(x[0])

        imle_pool_size = int(len(dataset) * self.H.imle_factor)
        if factor != -1:
            imle_pool_size = int(len(dataset) * factor)

        t1 = time.time()
        # if self.H.reinitialize_nn:
        #     self.selected_dists[:] = np.inf
        self.selected_dists_tmp[:] = self.selected_dists[:]
        for i in range(imle_pool_size // self.H.imle_db_size):
            self.temp_latent_rnds.normal_()
            self.temp_latent_rnds[:, self.H.latent_dim//2:].zero_()
            for j in range(self.H.imle_db_size // self.H.n_batch):
                batch_slice = slice(j * self.H.n_batch, (j + 1) * self.H.n_batch)
                cur_latents = self.temp_latent_rnds[batch_slice].cuda(device=self.H.devices[0])
                with torch.no_grad():
                    self.temp_samples[batch_slice] = model(cur_latents)
                    # self.temp_samples[batch_slice] = torch.from_numpy(self.sample(cur_latents, model))
                    self.temp_samples_proj[batch_slice] = self.get_projected(self.temp_samples[batch_slice])

            if not model.module.dci_db:
                model.module.dci_db = DCI(self.temp_samples_proj.shape[1], num_comp_indices=self.H.num_comp_indices,
                                        num_simp_indices=self.H.num_simp_indices)
            model.module.dci_db.add(self.temp_samples_proj)
            if force_update:
                print('samples generated')

            t0 = time.time()
            for ind, y in enumerate(DataLoader(dataset, batch_size=self.H.imle_batch)):
                # t2 = time.time()
                x = self.dataset_proj[ind * self.H.imle_batch:(ind + 1) * self.H.imle_batch].cuda(device=self.H.devices[0])
                cur_batch_data_flat = x.float().cuda(device=self.H.devices[0])
                nearest_indices, _ = model.module.dci_db.query(cur_batch_data_flat, num_neighbours=1)
                nearest_indices = nearest_indices.long()[:, 0]

                batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x.size()[0])
                actual_selected_dists = self.calc_loss(y[0].cuda(device=self.H.devices[0]), self.temp_samples[nearest_indices].cuda(device=self.H.devices[0]), use_mean=False, lpips_coef=1., l2_coef=0.)
                #actual_selected_dists = self.calc_loss(y[0].cuda(device=self.H.devices[0]), self.temp_samples[nearest_indices].cuda(device=self.H.devices[0]), use_mean=False)
                actual_selected_dists = torch.squeeze(actual_selected_dists)

                to_update = torch.nonzero(actual_selected_dists < self.selected_dists[batch_slice], as_tuple=False)
                self.selected_dists[ind * self.H.imle_batch + to_update] = actual_selected_dists[to_update]
                self.selected_latents[ind * self.H.imle_batch + to_update] = self.temp_latent_rnds[nearest_indices[
                    to_update]] + self.H.imle_perturb_coef * torch.randn(self.selected_latents[ind * self.H.imle_batch + to_update].shape)

                to_update = torch.nonzero(actual_selected_dists < self.selected_dists_future[batch_slice],
                                          as_tuple=False)
                self.selected_dists_future[ind * self.H.imle_batch + to_update] = actual_selected_dists[to_update]
                self.selected_latents_future[ind * self.H.imle_batch + to_update] = self.temp_latent_rnds[
                    nearest_indices[to_update]]

                del cur_batch_data_flat
            model.module.dci_db.clear()

            if i % 100 == 0:
                print("NN calculated for {} - {}".format((i + 1) * self.H.imle_db_size, time.time() - t0))

        if force_update:
            self.selected_dists[:] = self.selected_dists_future[:]
            self.selected_latents[:] = self.selected_latents_future[:] + self.H.imle_perturb_coef * torch.randn(self.selected_latents.shape)
            self.selected_dists_future[:] = np.inf
            self.selected_latents_future.normal_()
            self.calc_dists_existing(dataset, model)

        # adding perturbation
        # latents += H.imle_perturb_coef * torch.randn(latents.shape)
        changed = torch.sum(self.selected_dists_tmp != self.selected_dists).item()
        print("Samples and NN are calculated, time: {}, mean: {} # changed: {}, {}%".format(time.time() - t1,
                                                                                            self.selected_dists.mean(),
                                                                                            changed, (changed / len(
                dataset)) * 100))

    def second_phase(self, dataset, model, factor):
        t1 = time.time()
        tmp_latents = torch.zeros(factor, self.H.latent_dim)
        for ind, y in enumerate(DataLoader(dataset, batch_size=1)):
            y = y[0]
            tmp_latents.normal_()
            tmp_latents[:, :self.H.latent_dim//2] = self.selected_latents[ind, :self.H.latent_dim//2][:].reshape(1, self.H.latent_dim//2)
            with torch.no_grad():
                for i in range(factor // self.H.n_batch):
                    batch_slice = slice(i * self.H.n_batch, (i + 1) * self.H.n_batch)
                    cur_latents = tmp_latents[batch_slice].cuda(device=self.H.devices[0])
                    self.temp_samples[batch_slice] = model(cur_latents)

            flatten = self.temp_samples[:factor].reshape(factor, -1).cuda(device=self.H.devices[0])
            dci = DCI(flatten.shape[1], num_comp_indices=self.H.num_comp_indices, num_simp_indices=self.H.num_simp_indices)
            dci.add(flatten)

            nearest_indices, _ = dci.query(y.float().cuda(device=self.H.devices[0]).reshape(1, -1), num_neighbours=1)
            nearest_indices = nearest_indices.long()[:, 0]
            self.selected_latents[ind][:] = tmp_latents[nearest_indices][:]

