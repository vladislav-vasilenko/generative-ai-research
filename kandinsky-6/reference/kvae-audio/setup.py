from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="kvae-audio",
    version="1.0.0",
    description="Minimal inference package for KVAE-Audio.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "argbind>=0.3.7",
        "descript-audiotools>=0.7.2",
        "einops",
        "numpy",
        "soundfile",
        "torch",
        "torchaudio",
        "tqdm",
        "pesq",
    ],
    python_requires=">=3.8",
)
