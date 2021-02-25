import os
import imageio
import numpy as np

import torch
import torchvision
import clip
from skimage import exposure

from dd_ffm.lib import Siren, get_mesh_grid, fourierfm, slice_images

clip_normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                  (0.26862954, 0.26130258, 0.27577711))


def save_image(image, fname=None):
    image = np.array(image)[:, :, :]
    image = np.transpose(image, (1, 2, 0))
    image = exposure.equalize_adapthist(np.clip(image, -1., 1.))
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    if fname is not None:
        imageio.imsave(fname, np.array(image))
        imageio.imsave('result.jpg', np.array(image))


class DeepDazeFFM:
    def __init__(self, text="a blood black nothingness began to spin", prime_image=False, fine_details="", subtract="",
                 invert=False, sideX=512, sideY=512, uniform=True, sync_cut=True, steps=600, save_freq=1,
                 learning_rate=.00008, num_samples=60, siren_layers=16, use_fourier_feat_map=True, fourier_maps=128,
                 fourier_scale=2, prime_image_path="./img.png"):
        self.uniform = uniform
        self.text = text
        self.sync_cut = sync_cut
        self.prime_image = prime_image
        self.num_samples = num_samples
        self.save_freq = save_freq
        self.perceptor, self.preprocess = clip.load('ViT-B/32')
        self.sign = 1 if invert is True else -1,
        self.workdir = '_out'
        self.tempdir = os.path.join(self.workdir, 'ttt')
        os.makedirs(self.tempdir, exist_ok=True)
        self.out_name = text.replace(' ', '_')

        mgrid = get_mesh_grid(sideY, sideX)  # [262144,2]
        if use_fourier_feat_map:
            mgrid = fourierfm(mgrid, fourier_maps, fourier_scale)
        self.mgrid = torch.from_numpy(mgrid.astype(np.float32)).cuda()
        self.siren_model = Siren(mgrid.shape[-1], 256, siren_layers, 3, width=sideX, height=sideY).cuda()

        primed_image_enc = None
        self.img_in = None
        if prime_image:
            self.img_in = torch.from_numpy(imageio.imread(prime_image_path).astype(np.float32) / 255.).unsqueeze(0).permute(
                0, 3, 1, 2).cuda()
            if sync_cut is True:
                self.num_samples = num_samples // 2
            else:
                in_sliced = slice_images([self.img_in], num_samples, transform=clip_normalize, uniform=uniform)[0]
                primed_image_enc = self.perceptor.encode_image(in_sliced).detach().clone()
                del self.img_in, in_sliced
                torch.cuda.empty_cache()

        if len(text) > 2:
            print(' macro:', text)
            tx = clip.tokenize(text)
            self.primary_txt_enc = self.perceptor.encode_text(tx.cuda()).detach().clone()

        if len(fine_details) > 0:
            print(' micro:', fine_details)
            tx2 = clip.tokenize(fine_details)
            self.fine_details_txt_enc = self.perceptor.encode_text(tx2.cuda()).detach().clone()

        if len(subtract) > 0:
            print(' without:', subtract)
            tx0 = clip.tokenize(subtract)
            self.subtract_txt_enc = self.perceptor.encode_text(tx0.cuda()).detach().clone()

        self.optimizer = torch.optim.Adam(self.siren_model.parameters(), learning_rate)

        for i in range(steps):
            self.train(i, primed_image_enc)

    def checkin(self, num):
        with torch.no_grad():
            img = self.siren_model(self.mgrid).cpu().numpy()[0]
        save_image(img, os.path.join(self.tempdir, '%03d.jpg' % num))

    def train(self, iteration, sync_primed_image_encode):
        img_out = self.siren_model(self.mgrid)
        if self.prime_image and self.sync_cut is True:
            images_sliced = slice_images([self.img_in, img_out], self.num_samples, clip_normalize, self.uniform)
            sync_primed_image_encode = self.perceptor.encode_image(images_sliced[0])
        else:
            images_sliced = slice_images([img_out], self.num_samples, clip_normalize, self.uniform)
        generated_image_encode = self.perceptor.encode_image(images_sliced[-1])
        loss = 0
        if self.prime_image:
            loss += self.sign * 100 * torch.cosine_similarity(sync_primed_image_encode, generated_image_encode, dim=-1).mean()

        if self.primary_txt_enc is not None:
            loss += -100 * torch.cosine_similarity(self.primary_txt_enc, generated_image_encode, dim=-1).mean()

        if self.fine_details_txt_enc is not None:
            images_sliced = slice_images([img_out], self.num_samples, clip_normalize, uniform=self.uniform, micro=True)
            fine_details_img_encode = self.perceptor.encode_image(images_sliced[-1])
            loss += self.sign * 100 * torch.cosine_similarity(self.fine_details_txt_enc, fine_details_img_encode,
                                                              dim=-1).mean()
            del fine_details_img_encode
            torch.cuda.empty_cache()

        if self.subtract_txt_enc is not None:
            loss += -self.sign * 100 * torch.cosine_similarity(self.subtract_txt_enc, generated_image_encode, dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if iteration % self.save_freq == 0:
            self.checkin(iteration // self.save_freq)
