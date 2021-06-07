import copy
import re
from pathlib import Path
from random import randint
from typing import List, Optional

import PIL
import click
import clip
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import dnnlib
import legacy

norm = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
noise_floor = 0.02


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def best_guess_for_seed(G, model, text_features):
    seeds = (np.random.random(10) * 50000).astype(int)
    losses = []
    for seed in seeds:
        w = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(w, None, truncation_psi=1, noise_mode='const')
        clip_image = torch.nn.functional.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
        image_features = model.encode_image(clip_image)
        loss = torch.cosine_similarity(text_features, image_features, dim=-1).mean()
        losses.append(loss)

        return seeds[losses.index(min(losses))]


def approach(
        G,
        *,
        one_cycle=False,
        num_steps=100,
        initial_learning_rate=0.02,
        initial_noise_factor=0.02,
        noise_floor=0.02,
        psi=0.8,
        noise_ramp_length=1.0,  # was 0.75
        regularize_noise_weight=10000,  # was 1e5
        seed=69097,
        autoseed=True,
        autoseed_samples=20,
        noise_opt=True,
        ws=None,
        text='a computer generated image',
        device: torch.device
):
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    lr = initial_learning_rate
    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                           (0.26862954, 0.26130258, 0.27577711))

    # Load the perceptor
    print('Loading perceptor for text:', text)
    perceptor, preprocess = clip.load('ViT-B/32', jit=True)
    perceptor = perceptor.eval()
    tx = clip.tokenize(text)
    text_features = perceptor.encode_text(tx.cuda()).detach().clone()

    # autoseed
    if autoseed:
        print(f'Guessing the best seed using {autoseed_samples} samples')
        pod = np.full((autoseed_samples), 0)
        for i in range(autoseed_samples):
            seed = randint(0, 500000)
            pod[i] = seed
        staffel = []
        for i in range(autoseed_samples):
            snap = G(torch.from_numpy(np.random.RandomState(pod[i]).randn(1, G.z_dim)).to(device), None,
                     truncation_psi=psi, noise_mode='const')
            snap = torch.nn.functional.interpolate(snap, (224, 224), mode='bilinear', align_corners=True)
            eignung = int(
                torch.cosine_similarity(text_features, perceptor.encode_image(snap), dim=-1).cpu().detach().numpy() * 1000)
            staffel.append((pod[i], eignung))
        staffel = sorted(staffel, key=lambda x: (-x[1]))
        np_staffel = np.array(staffel, int)
        staffel_avg = np.mean(np_staffel, axis=0)[1]
        staffel_std = np.std(np_staffel, axis=0)[1]
        for i in range(autoseed_samples):
            if abs(np_staffel[i][1] - staffel_avg) < 1.7 * staffel_std:  # first non-outlier
                seed = np_staffel[i][0]
                break
        print(f'Top guess {staffel[i][0]}')

    # derive W from seed
    if ws is None:
        print('Generating w for seed %i' % seed)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w_samples = G.mapping(z, None, truncation_psi=psi)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
    else:
        w_samples = torch.tensor(ws, device=device)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

    w_std = 2  # ~9.9 for portraits network. should compute if using median median

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable

    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    if noise_opt:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w + noise')
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w')

    scheduler = None
    if one_cycle:
       scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_learning_rate * 10, total_steps=num_steps)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Descend
    for step in range(num_steps):
        # noise schedule
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        # floor
        if w_noise_scale < noise_floor:
            w_noise_scale = noise_floor

        # do G.synthesis
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')
        synth_images = nom(synth_images)  # normalize copied from CLIP preprocess. doesn't seem to affect tho

        # scale to CLIP input size
        into = torch.nn.functional.interpolate(synth_images, (224, 224), mode='bilinear', align_corners=True)
        # CLIP expects [1, 3, 224, 224], so we should be fine
        image_features = perceptor.encode_image(into)
        loss = -30 * torch.cosine_similarity(text_features, image_features, dim=-1).mean()  # Dunno why 30 works lol

        # noise reg, from og projector
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        if noise_opt:
            loss = loss + reg_loss * regularize_noise_weight
        else:
            loss = loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if one_cycle:
            scheduler.step()

        print(f'step {step + 1:>4d}/{num_steps}:  loss {float(loss):<5.2f} ', 'lr',
              lr, f'noise scale: {float(w_noise_scale):<5.6f}', f'proximity: {float(loss / (-30)):<5.6f}')

        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--lr', 'lr', help='Learning rate', default=0.02)
@click.option('--seed', type=int, help='The seed')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text', help='text', type=str, required=True)
@click.option('--one-cycle', 'one_cycle', type=bool, required=False)
@click.option('--num-steps', help='num step', type=int, default=100)
@click.option('--noise-factor', 'noise_factor', help='noise factor', type=float, default=0.02)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
def main(ctx: click.Context,
         network_pkl: str,\
         seed: Optional[int],
         outdir: str,
         text: str,
         lr: float,
         one_cycle: bool,
         noise_factor: float,
         num_steps: int,
         projected_w: str):

    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    projected_ws = approach(G,
                            ws=np.load(projected_w)['w'],
                            num_steps=num_steps,
                            seed=seed,
                            text=text,
                            device=device,
                            autoseed=seed is None,
                            initial_learning_rate=lr,
                            initial_noise_factor=noise_factor,
                            one_cycle=one_cycle)

    Path(outdir).mkdir(exist_ok=True)
    video = imageio.get_writer(f'{outdir}/out.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
    for projected_w in projected_ws:
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        video.append_data(np.concatenate([synth_image], axis=1))
    video.close()

    projected_w = projected_ws[0]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/in.png')

    projected_w = projected_ws[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/out.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

if __name__ == '__main__':
    main()
