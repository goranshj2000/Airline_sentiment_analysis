import transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification

#config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 64
BATCH_SIZE = 16
VAL_RATIO = 0.2
EPOCHS = 2
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "airline_model.bin"
TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
        )
