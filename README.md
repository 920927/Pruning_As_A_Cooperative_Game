# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models

![This image is not available for now.](assets/framework.png)

* We rethink LLM pruning from a game-theoretic perspective, treating layers as interdependent players and revealing inter-layer dependencies that static heuristics fail to capture.
* We propose a scalable approximation framework that leverages stratified Monte Carlo mask sampling and a lightweight surrogate network, enabling efficient Shapley-based estimation of layer contributions in large LLMs.
* We validate our method on language modeling tasks and zero-shot benchmarks, showing consistent improvements over strong pruning baselines across diverse architectures.


## Installation
  ```bash
  conda create -n Shapley python=3.12
  conda activate Shapley
  pip install -r requirement.txt
  ```


## Models we used in article:
  | Source<br>Model | Pruning<br>Ratio | 🤗Hugging Face<br>Link 
  |:---:|:---:|:---:|
  | Llama-2-7B | 20%, 35% | [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) 
  | Llama-2-13B | 20%, 35% | [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) 
  | Llama-3-8B | 20%, 35% | [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 
  | Vicuna-v1.3-7B | 20%, 35% | [Vicuna-v1.3-7B ](https://huggingface.co/lmsys/vicuna-7b-v1.3) 
  | Vicuna-v1.3-13B | 20%, 35% |[Vicuna-v1.3-7B ](https://huggingface.co/lmsys/vicuna-13b-v1.3) 



