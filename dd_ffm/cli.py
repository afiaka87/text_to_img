import os
import sys
import fire
import clip

from dd_ffm.deep_daze_ffm import DeepDazeFFM

perceptor, preprocess = clip.load('ViT-B/32')

workdir = '_out'
tempdir = os.path.join(workdir, 'ttt')
os.makedirs(tempdir, exist_ok=True)


def run(
        text,
        fine_details="", subtract="", invert=False, sideX=512, sideY=512, uniform=True, sync_cut=True, steps=600,
        save_freq=1, learning_rate=.00008, num_samples=60, siren_layers=16, use_fourier_feat_map=True, fourier_maps=128,
        fourier_scale=2, prime_image_path="./img.png"
):
    if any("--help" in arg for arg in sys.argv):
        print("Type `img2txt --help` for usage info.")
        sys.exit()

    DeepDazeFFM(
        text,
        fine_details=fine_details, subtract=subtract, invert=invert, sideX=sideX, sideY=sideY, uniform=uniform, sync_cut=sync_cut, steps=steps,
        save_freq=save_freq, learning_rate=learning_rate, num_samples=num_samples, siren_layers=siren_layers, use_fourier_feat_map=use_fourier_feat_map, fourier_maps=fourier_maps,
        fourier_scale=fourier_scale, prime_image_path=prime_image_path
    )


def main():
    fire.Fire(run)
