## Project Description
This repository contains the implementation of Context Embedding Injection (CEI) a training-free hallucination mitigation method for large vision-language models (LVLM).
## Implementation Details
Our implementation leverages the Hugging Face versions of LLaVA-1.5 and InstructBLIP, built on Transformer version 4.47.

## Dataset Requirements
To run the experiments, you need to download the following datasets:

- **AMBER Dataset**: Please download the AMBER dataset from its original repository [AMBER](https://github.com/junyangwang0410/AMBER). Ensure it is extracted and accessible on your system.
- **MS COCO 2014 Validation Set**: Required for the POPE and CHAIR benchmarks. Download the validation set from the official [MS COCO](https://cocodataset.org/#home) website and prepare it for use in the experiments.

## Environment Setup
To set up the required environment, follow these steps:

1. **Install Conda**: Ensure you have Miniconda or Anaconda installed on your system. If not, download and install it from the official website.
2. **Create Conda Environment**: Create a new Conda environment named `CAAC` with Python 3.10 by running:
   ```bash
   conda create -n CEI python=3.10
   ```
3. **Activate Environment**: Activate the environment with:
   ```bash
   conda activate CEI
   ```
4. **Install Dependencies**: Install the required dependencies listed in the `requirements.txt` file by running:
   ```bash
   pip install -r requirements.txt
   ```
