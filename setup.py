from setuptools import setup, find_packages

# minimal setup script for the mmpb package
setup(
    name="mmpb",
    packages=find_packages(exclude=["test"]),
    version='1.0.0',
    author="Constantin Pape",
    url="https://github.com/platybrowser/platybrowser-backend"
)
