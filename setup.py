from setuptools import setup

name = "rydberg"

with open("README.md", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", encoding="utf-8") as file:
    install_requirements = file.read().splitlines()

setup(
    name=name,
    packages=[name],
    long_description=long_description,
    install_requires=install_requirements,
)