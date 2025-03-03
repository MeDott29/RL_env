from setuptools import setup, find_packages

setup(
    name="sakana",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pygame>=2.0.0",
        "gymnasium>=0.28.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
    ],
    author="AI Assistant",
    author_email="example@example.com",
    description="A simple physics-based reinforcement learning environment",
    keywords="reinforcement-learning, ai, physics, simulation",
    url="https://github.com/yourusername/sakana",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 