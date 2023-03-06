from setuptools import setup, find_packages, Extension
import os


lib_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_dir + "/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

lib = Extension(name="data_analysis.segmentation.extensions.image_operations",
                sources=[
                    "data_analysis/segmentation/extensions/image_operations.cpp",
                    "data_analysis/segmentation/extensions/HeadLampObject.cpp"
                ])

setup(
    name="paper",
    version="0.0.0",
    packages=find_packages(),
    url="",
    license="Creative Commons Legal Code ",
    author="Lukas Ewecker",
    author_email="lukas.ewecker@porsche.de",
    description="Code for the upcoming paper.",
    install_requires=install_requires,
    ext_modules=[lib]
)
