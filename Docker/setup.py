from setuptools import setup, find_packages

setup(
    name="tigeralgorithmexample",
    version="0.0.1",
    author="Arian Arab",
    author_email="arian.arab@fda.hhs.gov",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "numpy",
        "tqdm==4.62.3",
        "tensorflow==2.5.0",
        "scipy",
        "segmentation_models",
        "protobuf~=3.19.0"                 
    ],
)
