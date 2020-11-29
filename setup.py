import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snnident",
    version="0.0.1",
    author="michaelrst",
    author_email="michael.rauchensteiner@ma.tum.de",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scikit-learn", "seaborn", "pandas"],
    python_requires='>=3.8',
)
