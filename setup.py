from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ViH-GAT",
    version="0.1.0",
    author="dapao111",
    description="A Python package for ViH-GAT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dapao111/ViH-GAT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your project dependencies here
        # Example: "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            # Development dependencies
            # Example: "pytest>=6.0",
            # Example: "black>=21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Define command-line scripts here if needed
            # Example: "my-script=my_package.module:main",
        ],
    },
)
