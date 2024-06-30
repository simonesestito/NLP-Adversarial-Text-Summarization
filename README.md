# NLP-Adversarial-Text-Summarization
NLP project to perform Adversarial Attack on Text summarization models, using Seq2Sick

[Download PDF paper](https://github.com/simonesestito/NLP-Adversarial-Text-Summarization/blob/main/paper.pdf?raw=true)

---

The original implementation of [Seq2Sick](https://github.com/cmhcbb/Seq2Sick) is broken with modern Python environment because of ancient dependencies.

The implementation from [FSE22_NMTSloth](https://github.com/SeekingDream/FSE22_NMTSloth/blob/64c50a6edc38262a69a89edb596b25b6d85f500e/src/baseline_attack.py#L94) is very good, but we needed to make a small copy for a different experiment, and not something related to their paper.

## Usage instructions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fSmhqp1gSV8ZRe56ARnYhoSfVzwPCZYH?usp=sharing)

1. Install the required dependencies (use virtual environments!)
2. Run the attack on Google Pegasus model (the default one) with:
```sh
python generate_adv.py --beam 1
```
3. Pretty print results from previous executions:
```sh
python pretty_print_results.py ./adversarial_result__2024-06-19-15-00-00.json  # Replace with YOUR results file
```

## Credits

Code adapted from original repository from [Simin Chen](https://chensimin.site/), as stated in the first section of this README. A big thank-you!

Authors of this repository:
- [Simone Sestito](https://github.com/simonesestito)
- [Lorenzo Antonelli](https://github.com/Lorenzoantonelli)
