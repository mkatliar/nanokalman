# setup.py
from setuptools import setup, find_packages

setup(
    name="nanokalman",
    version="0.1.1",
    author="Mikhail Katliar",
    author_email="mkatliar@protonmail.com",
    description="Very minimal implementation of Kalman filter in Python",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mkatliar/nanokalman",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.3",
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.15.1"
    ],
)
