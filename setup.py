import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['dd_ffm']
from version import __version__

setup(
    name='dd_ffm',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'txt_to_img = dd_ffm.cli:main',
        ],
    },
    version=__version__,
    license='MIT',
    description='Deep Daze - Fourier Feature Map',
    author='Afiaka87',
    author_email='samsepi0l@fastmail.com',
    url='https://github.com/afiaka87/dd_ffm',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'implicit neural representations',
        'text to image'
    ],
    install_requires=[
        'einops>=0.3',
        'fire',
        'ftfy',
        'torch>=1.7.1',
        'torch_optimizer',
        'torchvision>=0.8.2',
        'tqdm',
        'regex'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
