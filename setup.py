from setuptools import setup, find_packages

# Setting as official description of the project
with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name='PyLens',
    version='1.0',
    description="",
    packages=find_packages(),
    long_description = long_description,
    author="AUTHOR",
    author_email="EMAIL_ADDRESS",
    long_description_content_type="text/markdown",
    url="https://github.com/blairarmstrong/PyLens",
    classifiers=[
        #https://pypi.org/classifiers/
        "Programming Language :: Python :: 3.14"
    ],
    install_requires=[
        "numpy~=2.4.6",
        "webcolors~=1.13",
        "ipykernel~=6.25.2",
        "pyscreenshot==3.1",
        "pandas",
        "keyboard",
        "matplotlib~=3.10.5",
        "humanfriendly~=10.0",
        "scipy",
        "torch~=2.13",
        "ray~=2.57.0", 
        "regex"
    ],
    # dev requirements
    extras_require = {
        "dev":[
            "pytest >=3.7",
            "sphinx",
        ],
    }
)
