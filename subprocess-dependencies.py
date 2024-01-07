import subprocess

# List of packages with specified versions
packages = [
    "charset-normalizer==2.0",
    "botocore==1.31.81",
    "urllib3<2.1,>=1.25.4",
    "fsspec<=2023.10.0,>=2023.1.0",
    # Add other packages with specific versions here
]

# Loop through packages and install them
for package in packages:
    subprocess.run(["pip", "install", package])

# Optionally, you can also install additional packages without specific versions
additional_packages = [
    "certifi",
    "filelock",
    "huggingface-hub",
    "idna",
    "numpy",
    "packaging",
    "pyyaml",
    "regex",
    "requests",
    "safetensors",
    "tokenizers",
    "tqdm",
    "transformers",
    "typing-extensions",
]

# Install additional packages
for package in additional_packages:
    subprocess.run(["pip", "install", package])
