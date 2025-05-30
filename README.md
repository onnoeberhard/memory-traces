# Memory Traces for Reinforcement Learning
This is the official repository to the paper "Partially Observable Reinforcement Learning with Memory Traces" by Onno Eberhard, Michael Muehlebach, and Claire Vernade (published at ICML 2025).
All algorithms discussed in the paper are included here.

## Installation
```
pip install git+https://github.com/onnoeberhard/memory-traces
```

## Examples
Once installed, you can run the examples in the `examples` folder, e.g.
```
python examples/ppo_tmaze.py
```
This script trains a Proximal Policy Optimization (PPO) agent in a T-Maze environment with corridor length 64. The training takes about 5 minutes on a standard laptop. Weights and Biases logging can be enabled in the configuration dictionary in the example script.

With `memory='trace'`, the agent achieves a success rate of about 80% after 20 million steps, while `memory='window'` fails to learn anything. (Subject to stochasticity, results may vary!)

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{eberhard-2025-partially,
  title = {Partially Observable Reinforcement Learning with Memory Traces},
  author = {Eberhard, Onno and Muehlebach, Michael and Vernade, Claire},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year = {2025},
  series = {Proceedings of Machine Learning Research},
  volume = {267},
  url = {https://arxiv.org/abs/2503.15200}
}
```

If there are any problems, or if you have a question, don't hesitate to open an issue here on GitHub.

