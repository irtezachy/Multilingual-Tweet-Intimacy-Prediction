# Multilingual Tweet Intimacy Prediction

This project provides a solution for predicting the intimacy level of tweets in multiple languages using state-of-the-art NLP models. It leverages Hugging Face Transformers and PyTorch for model training and inference. The repository includes pre-trained models, data, and Jupyter notebooks for experimentation and evaluation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Running the Notebooks](#running-the-notebooks)
- [Model Checkpoints](#model-checkpoints)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to predict the intimacy score of tweets across different languages. The model is fine-tuned on a multilingual dataset and can be used for both research and practical applications in social media analysis, content moderation, and more.

## Features
- Multilingual support for tweet intimacy prediction
- Pre-trained and improved model checkpoints
- Easy-to-use Jupyter notebooks for inference and evaluation
- Example data and output predictions

## Project Structure
```
final_model.tar.gz                # Compressed final model
improved_model_checkpoint.tar.gz  # Compressed improved model checkpoint
LICENSE
outputs/
    test_predictions.csv          # Example output predictions
final_model/
    ...                          # Final model files
improved_model/
    checkpoint-5935/             # Improved model checkpoint files
    ...
data/
    semeval_test.csv             # Test data
    train.csv                    # Training data
tweetIntimacy.ipynb              # Main notebook for model usage
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/irtezachy/Multilingual-Tweet-Intimacy-Prediction.git
   cd Multilingual-Tweet-Intimacy-Prediction
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided, install the main dependencies manually:
   ```bash
   pip install torch transformers pandas scikit-learn jupyter
   ```
4. **Hugging Face API Token:**
   - Sign up at [Hugging Face](https://huggingface.co/) and get your API token from your account settings.
   - Add your token to your environment:
     ```bash
     export HUGGINGFACE_TOKEN=your_token_here
     ```
   - Alternatively, login using the CLI:
     ```bash
     huggingface-cli login
     ```

## Usage
- **Run the main notebook:**
  Open `tweetIntimacy.ipynb` or `tweetIntimacy_forgit.ipynb` in Jupyter Notebook or JupyterLab.
  ```bash
  jupyter notebook
  ```
- **Follow the instructions in the notebook** to load the model, run predictions, and evaluate results.

## Running the Notebooks
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `tweetIntimacy.ipynb` or `tweetIntimacy_forgit.ipynb`.
3. Run the cells step by step. Make sure your Hugging Face API token is set as described above.

## Model Checkpoints
- `final_model/`: Contains the final trained model and tokenizer files.
- `improved_model/checkpoint-5935/`: Contains an improved checkpoint with additional training.
- You can use these checkpoints for inference or further fine-tuning.

## Data
- `data/train.csv`: Training data.
- `data/semeval_test.csv`: Test data for evaluation.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the terms of the LICENSE file in this repository.
