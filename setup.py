from setuptools import setup, find_packages

setup(
    name="agentic-rag",
    version="1.0.0",
    packages=find_packages(),
    py_modules=["cli", "config", "main"],
    install_requires=[
        "typer[all]>=0.12",
        "rich>=13.7",
        "httpx>=0.27",
        "python-dotenv>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "rag=cli:app",
        ],
    },
    python_requires=">=3.11",
)
