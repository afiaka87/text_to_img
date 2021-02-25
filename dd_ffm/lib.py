from base64 import b64encode
import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                lim = 1 / self.in_features
            else:
                lim = np.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-lim, lim)

    def forward(self, _input):
        return torch.sin(self.omega_0 * self.linear(_input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., width=512, height=512):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for hidden_layer in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                lim = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-lim, lim)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        self.width = width
        self.height = height

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords.cuda())
        return output.view(1, self.width, self.height, 3).permute(0, 3, 1, 2)  # .sigmoid_()


def get_mesh_grid(side_x, side_y):
    tensors = [np.linspace(-1, 1, num=side_y), np.linspace(-1, 1, num=side_x)]
    mesh_grid = np.stack(np.meshgrid(*tensors), axis=-1)
    mesh_grid = mesh_grid.reshape(-1, 2)  # dim 2
    return mesh_grid


def fourierfm(xy, _map=256, fourier_scale=4, mapping_type='gauss'):
    def input_mapping(x, B):  # feature mappings
        x_proj = (2. * np.pi * x) @ B
        y = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
        print(' mapping input:', x.shape, 'output', y.shape)
        return y

    if mapping_type == 'gauss':  # Gaussian Fourier feature mappings
        B = np.random.randn(2, _map)
        B *= fourier_scale  # scale Gauss
    else:  # basic
        B = np.eye(2).T
    xy = input_mapping(xy, B)
    return xy


def slice_images(imgs, count, transform=None, uniform=False, micro=None):
    def _map(x, a, b):
        return x * (b - a) + a

    rnd_size = torch.rand(count)
    if uniform is True:
        rnd_offx = torch.rand(count)
        rnd_offy = torch.rand(count)
    else:  # normal around center
        rnd_offx = torch.clip(torch.randn(count) * 0.2 + 0.5, 0, 1)
        rnd_offy = torch.clip(torch.randn(count) * 0.2 + 0.5, 0, 1)

    sz = [img.shape[2:] for img in imgs]
    sz_min = [np.min(s) for s in sz]
    if uniform is True:
        sz = [[2 * s[0], 2 * s[1]] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], _type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        for c in range(count):
            if micro is True:  # both scales, micro mode
                c_size = _map(rnd_size[c], 64, max(224, 0.25 * sz_min[i])).int()
            elif micro is False:  # both scales, macro mode
                c_size = _map(rnd_size[c], 0.5 * sz_min[i], 0.98 * sz_min[i]).int()
            else:  # single scale
                c_size = _map(rnd_size[c], 64, 0.98 * sz_min[i]).int()
            offset_x = _map(rnd_offx[c], 0, sz[i][1] - c_size).int()
            offset_y = _map(rnd_offy[c], 0, sz[i][0] - c_size).int()
            cut = img[:, :, offset_y:offset_y + c_size, offset_x:offset_x + c_size]
            cut = torch.nn.functional.interpolate(cut, (224, 224), mode='bicubic')
            if transform is not None:
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced


# def create_video(seq_dir, size=None):
#   out_sequence = seq_dir + '/%03d.jpg'
#   out_video = seq_dir + '.mp4'
#   # !ffmpeg -y -v warning -i $out_sequence $out_video
#   data_url = "data:video/mp4;base64," + b64encode(open(out_video,'rb').read()).decode()
#   wh = '' if size is None else 'width=%d height=%d' % (size, size)
#   return """<video %s controls><source src="%s" type="video/mp4"></video>""" % (wh, data_url)


def tile_pad(xt, padding):
    h, w = xt.shape[-2:]
    left, right, top, bottom = padding

    def tile(x, minx, maxx):
        rng = maxx - minx
        mod = np.remainder(x - minx, rng)
        out = mod + minx
        return np.array(out, dtype=x.dtype)

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)
    x_pad = tile(x_idx, -0.5, w - 0.5)
    y_pad = tile(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return xt[..., yy, xx]


def pad_up_to(x, size, _type='centr'):
    sh = x.shape[2:][::-1]
    if list(x.shape[2:]) == list(size): return x
    padding = []
    for i, s in enumerate(size[::-1]):
        if 'side' in _type.lower():
            padding = padding + [0, s - sh[i]]
        else:  # centr
            p0 = (s - sh[i]) // 2
            p1 = s - sh[i] - p0
            padding = padding + [p0, p1]
    y = tile_pad(x, padding)
    return y
