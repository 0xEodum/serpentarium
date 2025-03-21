from setuptools import setup, find_packages

setup(
    name="serpentarium",
    version="0.2.0",
    description="Фреймворк для обучения агентов игре 'Змейка'",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pyyaml",
        "neat-python"
    ]
)