# Airline_sentiment_analysis
Fine tuned a BERT based model for a binary classification task and deployed using FastAPI

Dataset : https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment (cleaned to remove neutral values)

Link to the fine tuned model: https://drive.google.com/file/d/15whKUNQY5CTVtZHaZ-4hPZf-n99hSAoV/view?usp=sharing

Model : BERT based

Requirements:
1. Torch
2. Transformers
3. Scikit
4. FastAPI
5. Uvicorn

To run the API via uvicorn type: uvicorn main:app --reload on the command line
