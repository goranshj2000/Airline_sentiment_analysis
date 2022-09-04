import config
import torch
import engine
import metricsfile
from tabulate import tabulate
from tqdm import trange
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np


def run():
    df = pd.read_csv("/content/sample_data/airline_cleaned.csv")
    df = df.drop(df.columns[0], axis=1)
    text = df.text.values
    labels = df.labels.values
    labels.astype('int32')

    token_id = []
    attention_masks = []
    for sample in text:
      encoding_dict = engine.preprocessing(sample, config.TOKENIZER)  #config.TOKENIZER
      token_id.append(encoding_dict['input_ids']) 
      attention_masks.append(encoding_dict['attention_mask'])

    token_id = torch.cat(token_id, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)
    val_ratio = 0.2
    # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
    batch_size = 16

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size = val_ratio,
        shuffle = True,
        stratify = labels)

    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], 
                              attention_masks[train_idx], 
                              labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])

    # Prepare DataLoader
    train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = batch_size
            )

    device = torch.device(config.DEVICE)  #torch.device(config.DEVICE)
    model = model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
        )
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )
    model.cuda()
    #num_train_steps = int(len(df_train) / config.BATCH_SIZE * EPOCHS)  #int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    #scheduler = get_linear_schedule_with_warmup(
        #optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    #)
    best_accuracy = 0
   
    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    epochs = 2

    for _ in trange(epochs, desc = 'Epoch'):
        
        # ========== Training ==========
        
        # Set model to training mode
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
              # Forward pass
              eval_output = model(b_input_ids, 
                                  token_type_ids = None, 
                                  attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = metricsfile.b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
      
        torch.save(model.state_dict(), config.MODEL_PATH)
