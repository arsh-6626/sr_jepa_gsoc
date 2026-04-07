# LM-JEPA for Symbolic Regression

## Intro
This project implements a Joint Embedding Predictive Architecture (JEPA) with a decoder-only transformer for symbolic regression. Instead of explicit tree search, it learns latent representations of data clouds and maps them to symbolic expressions in prefix form.

## Repository Structure
```
src/
  datasets/        # Data loading & preprocessing
  embeddings/      # T-Net (data cloud encoder)
  models/          # Transformer + JEPA model
  train.py         # Training (dense attention)
  train_sparse.py  # Training (sparse attention)

scripts/
  prefix_parser_bfgs.py
  prefix_vocab.py
  vocab_gen.py

utils/
  FeynmanPrefixEquations.csv
  FeynmanAugmentPrefixEquations.csv
  vocab_bfgs_prefix.json
```
## Training
For training the dense attention network 
` python src/train.py`

For training the sparse attention network
`python src/train_sparse.py`

## Features
- JEPA-based representation learning
- Prefix symbolic expressions
- Data cloud (set-based) encoding
- Sparse attention for small datasets
- BFGS-based constant optimization

| Model                  | Dataset           | Token Accuracy | Levenshtein Accuracy |
|------------------------|------------------|----------------|----------------------|
| Sparse (from scratch)  | Feynman          | 0.32           | 0.54                 |
| Sparse (finetuned)     | Augmented Feynman| 0.51           | 0.76                 |

## Author
Arsh Abbas Naqvi
GitHub: https://github.com/arsh-6626