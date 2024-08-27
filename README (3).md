
# GUVI GPT Model using Hugging Face

This repository contains the code and instructions to deploy the Generative Pre-trained Transformer(GPT) using data from GUVI. This model is designed to generate text that is both coherent and aligns contextually with the provided input.


## Table of Contents 

* Project Overview
* Dataset
* Model Architecture
* Training
* Evaluation
* Usage
* Disclaimer
* Acknowledgements
## Project Overview

The aim of this project is to fine-tune a GPT model using data from GUVI to create a specialized text generation tool. The model is designed to assist with tasks such as content creation, automated responses, and educational support. By leveraging GUVI-specific data, the model generates contextually relevant and accurate text for various use cases.
## Dataset

The dataset for training the model is sourced from multiple websites, encompassing diverse topics and contexts. This variety ensures a comprehensive and robust training set. The wide-ranging text data enhances the model's ability to generate accurate and contextually appropriate responses.
## Model Architecture

The model architecture is based on GPT-2, a transformer-based model designed for generating human-like text. It has been fine-tuned with the collected dataset to enhance performance on specific tasks. This fine-tuning allows the model to produce more accurate and contextually relevant outputs.
## Training


The training process includes these key steps:

Data Collection: Gather text data from various websites to build a comprehensive dataset.
Data Preprocessing: Clean and format the text data for training.
Tokenization: Convert the text into tokens using the GPT-2 tokenizer.
Fine-tuning: Adjust the GPT-2 model using the processed and tokenized data.
Evaluation: Assess the model’s performance with metrics to ensure it meets quality standards.
## Requirements

The project requires Python 3.8+, PyTorch, and the Transformers library from Hugging Face. It also uses Streamlit for the user interface and accelerate for optimized training. Additional dependencies are listed in requirements.txt.
## Fine-tuning Script

Fine-tune the model in Google Colab notebook and export it:

Open the Colab notebook and run the training script.
Download the fine-tuned model.
## Upload to Hugging Face

Upload the fine-tuned model folder to Hugging Face.
## Evaluation

This model is designed to generate text that is both coherent and aligns contextually with the provided input.
## Usage

1. Run the Streamlit App: streamlit run app.py

2. Interact with the Model: Enter seed text and generate text using the Streamlit interface.

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#Load the fine-tuned model and tokenizer
model_name_or_path = "path_to_your_finetuned_model_on_hugging_face"

model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

#Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):

#Tokenize the input text with padding
inputs = tokenizer(seed_text, return_tensors='pt', padding=True, truncation=True)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
    
# Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0] skip_special_tokens=True)
## Disclaimer

The information provided in this application is for educational purposes only.The creators of this app make no representations or warranties of any kind, express or implied,about the completeness, accuracy, reliability, suitability, or availability with respect to the app or the information contained within it.
## Acknowledgements

https://openai.com/ for developing GPT-2. https://huggingface.co/ for providing the Transformers library and hosting the model. Various websites for providing the data used in this project mainly:https://www.guvi.in/
## Deployment

To deploy this project run just click the below link

https://huggingface.co/spaces/ammuharshiya/Adventure
## Demo

link to demo