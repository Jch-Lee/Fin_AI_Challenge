"""
Fin AI Challenge - 프로젝트 설정 스크립트
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fin-ai-challenge",
    version="0.1.0",
    author="FSKU Team",
    description="Financial Security Knowledge Understanding AI Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fin-ai-challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch==2.1.0",
        "transformers==4.41.2",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "trl==0.9.6",
    ],
    extras_require={
        "dev": [
            "pytest==8.2.0",
            "black==23.7.0",
            "ruff==0.0.285",
            "pre-commit==3.3.3",
        ],
        "rag": [
            "sentence-transformers==2.7.0",
            "faiss-cpu==1.8.0",
            "bm25s==0.2.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "fin-train=src.train:main",
            "fin-inference=src.inference:main",
            "fin-eval=src.evaluate:main",
        ],
    },
)