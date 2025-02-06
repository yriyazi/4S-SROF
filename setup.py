from setuptools import setup, find_packages

setup(
    name="SFOF4S",
    version="1.0.13",
    author="Sajjad Shumaly",
    maintainer="Yassin Riyazi",
    maintainer_email="iyasiniyasin98@gmail.com",
    description="This toolkit aids in analyzing drop sliding on tilted plates, allowing researchers to study various variables and their correlations in detail.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # Fix for PyPI description issue
    url="https://github.com/yriyazi/SFOF4S",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6",
        "torchaudio>=2.4",
        "torchvision>=0.21",
        "numpy>=1.26",
        "opencv-python>=4.10",
        "scipy>=1.15",
        "tqdm>=4.67",
        "pandas>=2.2",
        "natsort>=8.4",
        "requests>=2.32",
        "matplotlib>=3.9"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    # Project keywords for discoverability
    keywords="machine learning, computer vision, PyTorch, SciPy, OpenCV, droplet, contact angle",

    # Project URLs for additional resources
    project_urls={
        "Source Code": "https://github.com/AK-Berger/4S-SROF",
        "documentation": "https://github.com/yriyazi/SFOF4S-Documentation",
    },
)
