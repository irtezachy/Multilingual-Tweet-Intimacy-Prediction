# Final Project Report: Multilingual Tweet Intimacy Prediction

## Introduction
This report summarizes the objectives, methodology, experiments, and results of the Multilingual Tweet Intimacy Prediction project. The goal was to develop a model capable of predicting the intimacy level of tweets across multiple languages using state-of-the-art NLP techniques.

## Objectives
- Predict intimacy scores for tweets in various languages.
- Leverage pre-trained transformer models and fine-tune them on a multilingual dataset.
- Provide reproducible code and checkpoints for further research and application.

## Methodology
- **Data:** Used provided train and test datasets containing tweets and their intimacy scores.
- **Model:** Fine-tuned transformer-based models (e.g., XLM-RoBERTa) using Hugging Face Transformers and PyTorch.
- **Training:** Performed training and validation, saving checkpoints and final models.
- **Evaluation:** Evaluated model performance on test data and generated predictions.

## Experiments
- Compared baseline and improved models.
- Experimented with different hyperparameters and training strategies.
- Used multilingual embeddings to handle tweets in various languages.

## Results
- The improved model checkpoint outperformed the baseline in terms of accuracy and generalization.
- Final predictions are provided in the `outputs/test_predictions.csv` file.

## Usage
- See the README for installation and usage instructions.
- Run the provided Jupyter notebooks for inference and evaluation.

## Conclusion
The project successfully demonstrates multilingual tweet intimacy prediction using modern NLP techniques. The codebase, models, and data are available for further research and development.
