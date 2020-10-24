from setuptools import find_packages, setup

setup(
    name="corr_net",
    version="0.0.1",
    url="unknown",
    description="Implementation of paper: Video Modeling With Correlation Networks",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "detectron2",
        "opencv-python",
        "pandas",
        "torchvision>=0.4.2",
        "sklearn",
        "tensorboard"
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
