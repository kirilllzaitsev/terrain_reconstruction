
from setuptools import setup

requirements = open("requirements.txt").read().splitlines()
requirements = [r for r in requirements if not r.startswith("#")]

setup(
    name="terrain_representation",
    version="0.0.1",
    author="Kirill Zaitsev",
    packages=["terrain_representation"],
    install_requires=requirements,
)
