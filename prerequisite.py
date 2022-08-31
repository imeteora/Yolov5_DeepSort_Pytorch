import pip

_all_ = [
    "matplotlib>=3.2.2",
    "numpy>=1.18.5",
    "opencv-python>=4.1.2",
    "Pillow>=7.1.2",
    "PyYAML>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "tqdm>=4.41.0",

    # plotting ------------------------------------

    "pandas>=1.1.4",
    "seaborn>=0.11.0",

    # deep_sort -----------------------------------

    "easydict~=1.9",

    # torchreid

    "Cython~=0.29.30",
    "h5py",
    "six",
    "tb-nightly",
    "future",
    "yacs~=0.1.8",
    "gdown~=4.5.1",
    "flake8",
    "yapf",
    "isort==4.3.21",
    "imageio",
    "argparse~=1.4.0",
    "flask~=1.1.2",
    "thop~=0.1.1-2207130030",
    "setuptools~=61.2.0",
    "munkres~=1.1.4",
]

windows = [
    "PyQt5==5.15.2",
]

linux = [
    "PyQt5==5.14.1",
]

darwin = []


def install(packages):
    for package in packages:
        pip.main(['install', package])


if __name__ == '__main__':

    from sys import platform

    install(_all_)
    if platform == 'windows':
        install(windows)
    if platform.startswith('linux'):
        install(linux)
    if platform == 'darwin':  # MacOS
        install(darwin)
