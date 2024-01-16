from setuptools import setup

setup(
    name="compx",
    version="0.0.1",
    description="Compositional RL",
    packages=["compx"],
    install_requires=["stable_baselines3==2.0.0", "seaborn", "robosuite==1.4.0"],
)
