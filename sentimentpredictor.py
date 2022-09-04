import config
import train
import metricsfile
import engine
import torch
import numpy as np
#final predictor

def predict(new_sentence):

    #device = config.DEVICE
    device='cpu'
    tokenizer = config.TOKENIZER #config.TOKENIZER
    max_len = config.MAX_LEN  #config.MAX_LEN 
    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = engine.preprocessing(new_sentence, tokenizer)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    MODEL = config.MODEL
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.to(device)
    MODEL.eval()
    #model = torch.load(r'C:\Users\goran\Projects\nlp_exp\FastApi_Bert_Sentiment_airlines\airline_model.pt', map_location=torch.device('cpu'))
    #model.eval()

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = MODEL(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))
    #output = MODEL(ids=test_ids, mask=test_attention_mask)

    prediction = 'positive' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'negative'
    print(np.argmax(output.logits.cpu().numpy()).flatten().item())

    return prediction