from setuptools import setup, find_packages

setup(
    name='fevd_vqvae',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="(CS229 Project) FEVD-VQVAE: Frame-aware Encoding Video-aware Decoding VQ-VAE",
    author='Nick Soulounias',
    author_email='s0ul@stanford.edu',
    install_requires=[line for line in open('requirements.txt').readlines()],
    keywords=['Two-Stage-Video-Prediction', 'Discrete-Variational-Autoencoder' 'Robotics', 'Manipulation']
)