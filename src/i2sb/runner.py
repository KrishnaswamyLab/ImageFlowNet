# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug


def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys(
        ) and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (torch.linspace(linear_start**0.5,
                            linear_end**0.5,
                            n_timestep,
                            dtype=torch.float64)**2)
    return betas.numpy()


def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()


class Runner(object):

    def __init__(self, opt, log, save_opt=True):
        super(Runner, self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval,
                                   linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate(
            [betas[:opt.interval // 2],
             np.flip(betas[:opt.interval // 2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        assert opt.time_range[0] > 0
        interp_levels = torch.linspace(
            1e-4, 1.0, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log,
                               interp_levels=interp_levels,
                               use_fp16=opt.use_fp16,
                               cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(),
                                            decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    # def compute_label(self, step, x0, xt):
    #     """ Eq 12 """
    #     std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
    #     label = (xt - x0) / std_fwd
    #     return label.detach()

    def compute_label(self,
                      images,
                      step,
                      total_steps,
                      interp_method: str = 'linear_pair'):
        """ Use interpolation of the images! """
        if interp_method == 'linear_pair':
            x_target = images[-1][None, ...]
            x_source = images[0][None, ...]
            label = torch.lerp(x_target, x_source,
                               (step / torch.tensor(total_steps)).to(
                                   x_source.device))

        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        # """ Given network output, recover x0. This should be the inverse of Eq 12 """
        # std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        # pred_x0 = xt - std_fwd * net_out
        # NOTE: edited and removed the inverse process above.
        pred_x0 = net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader):
        # In the Retina Dataset (return_format == 'array'),
        # the dataloader returns 2 objects:
        #   - the image array: [1, N, C, H, W], where N is number of images.
        #   - the timestamps: [1, N]
        img_array, time_array = next(loader)

        assert img_array.shape[0] == 1
        assert time_array.shape[0] == 1

        images = img_array[0, ...].float().detach().to(opt.device)
        times = time_array[0, ...].float().detach().to(opt.device)

        x_source = images[0][None, ...]
        cond = x_source.detach() if opt.cond_x1 else None

        return images, times, cond

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader = util.setup_loader(val_dataset, opt.microbatch)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for it_inner in range(n_inner_loop):
                # ===== sample boundary pair =====
                images, times, cond = self.sample_batch(opt, train_loader)

                # ===== compute loss =====
                total_steps = int(
                    opt.interval * (times[-1] - times[0]) /
                    (opt.time_range[1] - opt.time_range[0]))

                step = torch.randint(0, total_steps, (1, ))

                # x0 is "target", x1 is "source".
                x0 = images[-1][None, ...]
                x1 = images[0][None, ...]

                # Compute the analytic posterior.
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)

                # Interpolated image as pseudo-GT.
                pseudo_label = self.compute_label(images, step, total_steps,
                                                  opt.interp)

                pred = net(xt, step, cond=cond)
                assert xt.shape == pseudo_label.shape == pred.shape

                if it % 100 == 0 and it_inner == 0:
                    sampled_time = (1 - step.item() / opt.interval) * (times[-1] - times[0]) + times[0]
                    from matplotlib import pyplot as plt
                    fig = plt.figure(figsize=(25, 6))
                    ax = fig.add_subplot(1, 5, 1)
                    ax.imshow(((x1 + 1) / 2).cpu().detach().numpy().reshape(
                        3, 256, 256).transpose(1, 2, 0))
                    ax.set_title('Source t = %s' % times[0].item())
                    ax.set_axis_off()
                    ax = fig.add_subplot(1, 5, 2)
                    ax.imshow(((x0 + 1) / 2).cpu().detach().numpy().reshape(
                        3, 256, 256).transpose(1, 2, 0))
                    ax.set_title('Target t = %s' % times[-1].item())
                    ax.set_axis_off()
                    ax = fig.add_subplot(1, 5, 3)
                    ax.imshow(((xt + 1) / 2).cpu().detach().numpy().reshape(
                        3, 256, 256).transpose(1, 2, 0))
                    ax.set_title('Analytic posterior t = %.1f' % sampled_time)
                    ax.set_axis_off()
                    ax = fig.add_subplot(1, 5, 4)
                    ax.imshow(((pseudo_label + 1) /
                               2).cpu().detach().numpy().reshape(
                                   3, 256, 256).transpose(1, 2, 0))
                    ax.set_title('Label t = %.1f' % sampled_time)
                    ax.set_axis_off()
                    ax = fig.add_subplot(1, 5, 5)
                    ax.imshow(((pred + 1) / 2).cpu().detach().numpy().reshape(
                        3, 256, 256).transpose(1, 2, 0))
                    ax.set_title('Pred t = %.1f' % sampled_time)
                    ax.set_axis_off()
                    fig.tight_layout()
                    os.makedirs('./debug_plot/', exist_ok=True)
                    fig.savefig('./debug_plot/check_%s' % str(it).zfill(5))

                loss = F.mse_loss(pred, pseudo_label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1 + it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            # if it % 5000 == 0:
            if it % 200 == 0:
                if opt.global_rank == 0:
                    torch.save(
                        {
                            "net":
                            self.net.state_dict(),
                            "ema":
                            ema.state_dict(),
                            "optimizer":
                            optimizer.state_dict(),
                            "sched":
                            sched.state_dict() if sched is not None else sched,
                        }, opt.ckpt_path / "latest.pt")
                    log.info(
                        f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # if it == 500 or it % 3000 == 0:  # 0, 0.5k, 3k, 6k 9k
            if it % 200 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self,
                      opt,
                      x1,
                      mask=None,
                      cond=None,
                      clip_denoise=False,
                      nfe=None,
                      log_count=10,
                      verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe + 1)

        # create log steps
        log_count = min(len(steps) - 1, log_count)
        log_steps = [
            steps[i] for i in util.space_indices(len(steps) - 1, log_count)
        ]
        assert log_steps[0] == 0
        self.log.info(
            f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0], ),
                                  step,
                                  device=opt.device,
                                  dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step,
                                            xt,
                                            out,
                                            clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps,
                pred_x0_fn,
                x1,
                mask=mask,
                ot_ode=opt.ot_ode,
                log_steps=log_steps,
                verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        images, times, cond = self.sample_batch(opt, val_loader)

        x_source = images[0][None, ...].to(opt.device)
        x_target = images[-1][None, ...].to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(opt,
                                          x_source,
                                          mask=None,
                                          cond=cond,
                                          clip_denoise=opt.clip_denoise,
                                          verbose=opt.global_rank == 0)

        log.info("Collecting tensors ...")
        x_source = all_cat_cpu(opt, log, x_source)
        x_target = all_cat_cpu(opt, log, x_target)
        # y           = all_cat_cpu(opt, log, y)
        xs = all_cat_cpu(opt, log, xs)
        pred_x0s = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert x_source.shape == x_target.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        # assert y.shape == (batch, )
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag,
                                  tu.make_grid((img + 1) / 2,
                                               nrow=nrow))  # [1,1] -> [0,1]

        # def log_accuracy(tag, img):
        #     pred = self.resnet(img.to(opt.device))  # input range [-1,1]
        #     accu = self.accuracy(pred, y.to(opt.device))
        #     self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        x_recon = xs[:, 0, ...]
        x_pred = pred_x0s[:, 0, ...]
        log_image("image/source", x_source)
        log_image("image/target", x_target)
        log_image("image/recon", x_recon)
        log_image("image/pred_target", x_pred)
        log_image("debug/pred_target_traj",
                  pred_x0s.reshape(-1, *xdim),
                  nrow=len_t)
        log_image("debug/recon_traj", xs.reshape(-1, *xdim), nrow=len_t)

        log.info("Not ImageNet data, not logging accuracies ...")
        # log.info("Logging accuracies ...")
        # log_accuracy("accuracy/source", x_source)
        # log_accuracy("accuracy/target", x_target)
        # log_accuracy("accuracy/recon", x_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
