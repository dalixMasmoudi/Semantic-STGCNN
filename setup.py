"""
Setup script for Semantic-STGCNN package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove version constraints for setup.py
            package = line.split('>=')[0].split('==')[0].split('<=')[0]
            requirements.append(package)

# Development requirements
dev_requirements = [
    'pytest>=6.2.0',
    'pytest-cov>=3.0.0',
    'black>=21.9.0',
    'flake8>=4.0.0',
    'isort>=5.9.0',
    'mypy>=0.910',
    'pre-commit>=2.15.0',
    'sphinx>=4.2.0',
    'sphinx-rtd-theme>=1.0.0',
]

setup(
    name="semantic-stgcnn",
    version="1.0.0",
    author="Dali Masmoudi",
    author_email="your.email@domain.com",
    description="Semantic Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dalixMasmoudi/Semantic-STGCNN",
    project_urls={
        "Bug Tracker": "https://github.com/dalixMasmoudi/Semantic-STGCNN/issues",
        "Documentation": "https://github.com/dalixMasmoudi/Semantic-STGCNN/blob/main/README.md",
        "Source Code": "https://github.com/dalixMasmoudi/Semantic-STGCNN",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": ["torch-geometric>=2.0.0", "torch-scatter>=2.0.0"],
        "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "semantic-stgcnn-train=semantic_stgcnn.scripts.train:main",
            "semantic-stgcnn-eval=semantic_stgcnn.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "semantic_stgcnn": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    keywords=[
        "trajectory prediction",
        "graph neural networks",
        "spatio-temporal modeling",
        "semantic features",
        "pedestrian prediction",
        "computer vision",
        "deep learning",
        "pytorch",
    ],
    zip_safe=False,
)
