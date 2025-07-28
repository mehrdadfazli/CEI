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

# Sina

Here is some introduction to the codes in this repo. 
1. Read our paper "Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration" to familiarize yourself with the workflow, benchmarks, LVLMs, etc. (if you have not already). It is because this project is a follow-up on our previous work. Also, for the time being, we are calling the project context embedding injection (CEI) to mitigate hallucination; hence, CEI is in the name of the Python scripts.
2. The injection idea is mainly based on the paper "The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?", which suggests that the first generated token (its logit) has a lot of useful information. Hence, we came up with the idea of injecting its embedding into the subsequent generations to foster visual grounding over long generations. I'll talk with you about the details.
3. As a preliminary experiment, we tested the idea of injecting the context embedding with a fixed weight at a certain layer of the decoder into the last input embedding at every generation round. Context embedding is what we call the embedding that we believe has baked all the information from the image and query into itself and is very informative (e.g., "first token" in the paper mentioned above). The choice of the last input embedding (to apply the injection to) is because it will be used to get the logit distribution. The hyperparameters of this simplistic intervention are `context_embedding_idx`, `context_embedding_layer`, which will be used to select the "context embedding" given an image and a query, `injection_layer`, and `alpha`, which decide the layer to inject to and the strength of the injection. The "injection" is a simple weighted average. When `alpha=0`, there is no injection, and when `alpha=1`, context embedding completely replaces the last embedding at the `injection_layer`. The code may also help you understand our method better. However, we will discuss further in our meeting.
4. Your first step should be setting up the environment, downloading the datasets, and running some basic experiments successfully. To do that, I suggest you focus on the AMBER benchmark. `CEI_utils.py` and `model_utils.py` are helper modules. `run_AMBER.py` is the main code to run an LVLM + CEI on the AMBER benchmark. `run_AMBER.sh` provides a convenient way of reading the experiment configs from `config` and passing them as command-line arguments to `run_AMBER.py`. For the config files, you can use any of them, but not the ones under `configs/dynamic`. Basically, ignore everything that has `dyn` or `dynamic` for now. You need to change the paths in the codes/configs according to your system. You also need to download the AMBER dataset.
5. Once you were able to run one experiment on the AMBER benchmark successfully, we can proceed with the next steps in the project.
