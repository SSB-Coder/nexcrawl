from setuptools import setup, find_packages

setup(
    name="nexcrawl-ml",
    version="1.0.0",
    description="Perception & ML subsystem for the NexCrawl-M mine rescue rover (SIH 2026, PS 26039)",
    packages=find_packages(exclude=["tests", "scripts", "serve"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "lightgbm>=4.0",
    ],
    extras_require={
        "mesh": ["open3d>=0.18.0"],
        "audio": ["torch>=2.0", "librosa>=0.10", "onnxruntime>=1.16"],
        "serve": ["fastapi>=0.109", "uvicorn>=0.27"],
        "hardware": ["pyserial>=3.5", "rplidar-roboticia>=1.0.0"],
    },
)
