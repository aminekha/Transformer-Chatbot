# Transformer Chatbot
## Introduction
Transformer chatbot is a web app built using Transformers, Tensorflow 2.0, and Django using the Cornell Movie-Dialogs Corpus Dataset (<a href="https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html">Link Here</a>).

## Features
<ul>
    <li>Train your own model or use the pretrained model (included in this repo)</li>
    <li>Beautiful UI to interact with the Chatbot</li>
    <li>Uses Tensorflow 2.0</li>
</ul>

## Setup
Setup everything using the requirements.txt file. 
Just run ``` python3 -r requirements.txt ```

## Usage
If you have everything installed (including Tensorflow and Django) you just need to run:
``` python3 manage.py runserver ``` 
Then wait for it to process the dataset and load the model. 
Then all you need is to open 127.0.0.1:8000 and interact with the chatbot.