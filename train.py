
import sys
import math
import json
import os
from collections import Counter
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import plotly.graph_objects as go
from microbert.model import MicroBERTForClassification
from microbert.tokenizer import WordTokenizer
from microbert.utils import IMDBDataloader, get_attention_scores, plot_parallel, plot_results, save_model, load_model
from hiq.vis import print_model

NUM_EPOCHS = 50
BATCH_SIZE = 32
MAX_SEQ_LEN = 128
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if trained model already exists
home_dir = os.path.expanduser('~')
model_save_dir = os.path.join(home_dir, '.microbert_model')

if os.path.exists(model_save_dir) and os.path.exists(os.path.join(model_save_dir, 'microbert_classification.pth')):
    print("✅ Found pre-trained model!")
    print(f"Model location: {model_save_dir}")
    
    # Load the existing model to show its configuration
    try:
        model, tokenizer, config = load_model(model_save_dir, device)
        print(f"Model config: vocab_size={config['vocab_size']}, n_layers={config['n_layers']}, n_heads={config['n_heads']}")
        print("Training completed, exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will start training from scratch...")

print("🚀 Starting new model training...")

data = None
with open('imdb_train.json') as f:
    data = [json.loads(l) for l in f.readlines()]

vocab = set()
for d in data:
    vocab |= set([w.lower() for w in d['text']])

test_data = None
with open('imdb_test.json') as f:
   test_data = [json.loads(l) for l in f.readlines()]

def encode_label(label):
    if label == 'pos':
        return 1
    elif label == 'neg':
        return 0
    raise Exception(f'Unknown Label: {label}!')


tokenizer = WordTokenizer(vocab=vocab, max_seq_len=MAX_SEQ_LEN)
dataloader = IMDBDataloader(data, test_data, tokenizer, encode_label, batch_size=BATCH_SIZE)

bert = MicroBERTForClassification(
    vocab_size=len(tokenizer.vocab),
    n_layers=1,
    n_heads=1,
    max_seq_len=MAX_SEQ_LEN,
    n_classes=2
).to(device)

print_model(bert)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(bert)

history = {
    'train_losses': [],
    'val_losses': [],
    'train_acc': [],
    'val_acc': [],
    'train_f1': [],
    'val_f1': []
}

optimizer = torch.optim.Adam(bert.parameters(), lr=LEARNING_RATE)

for i in range(NUM_EPOCHS):
    print(f'Epoch: {i + 1}')
    train_loss = 0.0
    train_preds = []
    train_labels = []

    bert.train()
    total = dataloader.steps('train')
    the_split = dataloader.get_split('train')
    for step, batch in enumerate(tqdm(the_split, total=total)):
        logits = bert(batch['input_ids'].to(device)) # (B, Seq_Len, 2)
        probs = F.softmax(logits[:, 0, :], dim=-1).cpu()
        pred = torch.argmax(probs, dim=-1) # (B)
        train_preds += pred.detach().tolist()
        train_labels += [l.item() for l in batch['label_ids']]
        loss = F.cross_entropy(logits[:, 0, :].cpu(), batch['label_ids'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0.0
    val_preds = []
    val_labels = []

    bert.eval()
    for step, batch in enumerate(tqdm(dataloader.get_split('val'), total=dataloader.steps('val'))):
        logits = bert(batch['input_ids'].to(device))
        probs = F.softmax(logits[:, 0, :], dim=-1).cpu()
        pred = torch.argmax(probs, dim=-1) # (B)
        val_preds += pred.detach().tolist()
        val_labels += [l.item() for l in batch['label_ids']]
        loss = F.cross_entropy(logits[:, 0, :].cpu(), batch['label_ids'])
        val_loss += loss.item()

    history['train_losses'].append(train_loss)
    history['val_losses'].append(val_loss)
    history['train_acc'].append(accuracy_score(train_labels, train_preds))
    history['val_acc'].append(accuracy_score(val_labels, val_preds))
    history['train_f1'].append(f1_score(train_labels, train_preds))
    history['val_f1'].append(f1_score(val_labels, val_preds))

    print()
    print(f'Train loss: {train_loss / dataloader.steps("train")} | Val loss: {val_loss / dataloader.steps("val")}')
    print(f'Train acc: {accuracy_score(train_labels, train_preds)} | Val acc: {accuracy_score(val_labels, val_preds)}')
    print(f'Train f1: {f1_score(train_labels, train_preds)} | Val f1: {f1_score(val_labels, val_preds)}')


plot_results(history)

test_loss = 0.0
test_preds = []
test_labels = []

bert.eval()
for step, batch in enumerate(tqdm(dataloader.get_split('test'), total=dataloader.steps('test'))):
    logits = bert(batch['input_ids'].to(device))
    probs = F.softmax(logits[:, 0, :], dim=-1).cpu()
    pred = torch.argmax(probs, dim=-1) # (B)
    test_preds += pred.detach().tolist()
    test_labels += [l.item() for l in batch['label_ids']]
    loss = F.cross_entropy(logits[:, 0, :].cpu(), batch['label_ids'])
    test_loss += loss.item()

print()
print(f'Test loss: {test_loss / dataloader.steps("test")}')
print(f'Test acc: {accuracy_score(test_labels, test_preds)}')
print(f'Test f1: {f1_score(test_labels, test_preds)}')

# Save trained model to home directory
home_dir = os.path.expanduser('~')
model_save_dir = os.path.join(home_dir, '.microbert_model')

# Prepare configuration
config = {
    'vocab_size': len(tokenizer.vocab),
    'n_layers': 1,
    'n_heads': 1,
    'max_seq_len': MAX_SEQ_LEN,
    'n_classes': 2,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'num_epochs': NUM_EPOCHS
}

# Use utility function to save model
save_model(bert, tokenizer, history, config, model_save_dir)

# Interpreting and visualizing the results
test_dataloader = IMDBDataloader(data, test_data, tokenizer, encode_label, batch_size=1)
# examples with less than 16 words are easier to visualize, so focus on them
examples_ids = []
for i, v in enumerate(test_dataloader.splits['test']):
    if len(v) <= 16:
        examples_ids.append(i)
print(examples_ids)

for sample_index in examples_ids:
    # extract example, decode to tokens and get the sequence length (ingoring padding)
    test_tokenized_batch = test_dataloader.peek_index_tokenized(index=sample_index, split='test')
    tokens = tokenizer.decode([t.item() for t in test_tokenized_batch['input_ids'][0] if t != 0], ignore_special=False).split(' ')[:MAX_SEQ_LEN]
    seq_len = len(tokens)
    # calculate attention scores
    att_matrix = get_attention_scores(bert, test_tokenized_batch['input_ids'].to(device))[0, :seq_len, :seq_len]
    plot_parallel(att_matrix, tokens=tokens)

scatters = []
for sample_index in examples_ids:
    # extract example, decode to tokens and get the sequence length (ingoring padding)
    test_tokenized_batch = test_dataloader.peek_index_tokenized(index=sample_index, split='test')
    tokens = tokenizer.decode([t.item() for t in test_tokenized_batch['input_ids'][0] if t != 0], ignore_special=False).split(' ')[:MAX_SEQ_LEN]
    seq_len = len(tokens)
    embed = bert.micro_bert.embedding(test_tokenized_batch['input_ids'].to(device))
    x, y, z = embed[0, :seq_len, 0].detach().cpu().numpy(), embed[0, :seq_len, 1].detach().cpu().numpy(), embed[0, :seq_len, 2].detach().cpu().numpy()
    scatters.append(go.Scatter3d(
        x=x, y=y, z=z, mode='markers+text', name=f'Example: {sample_index}',
        text=tokens,
    ))

fig = go.Figure(
    data=scatters,
    layout=go.Layout(
        title=go.layout.Title(text='Embeddings')
    ))
fig.show()

scatters = []
for sample_index in examples_ids:
    # extract example, decode to tokens and get the sequence length (ingoring padding)
    test_tokenized_batch = test_dataloader.peek_index_tokenized(index=sample_index, split='test')
    tokens = tokenizer.decode([t.item() for t in test_tokenized_batch['input_ids'][0] if t != 0], ignore_special=False).split(' ')[:MAX_SEQ_LEN]
    seq_len = len(tokens)
    embed = bert.micro_bert(test_tokenized_batch['input_ids'].to(device))
    x, y, z = embed[0, :seq_len, 0].detach().cpu().numpy(), embed[0, :seq_len, 1].detach().cpu().numpy(), embed[0, :seq_len, 2].detach().cpu().numpy()
    scatters.append(go.Scatter3d(
        x=x, y=y, z=z, mode='markers+text', name=f'Example: {sample_index}',
        text=tokens,
    ))
fig = go.Figure(
    data=scatters,
    layout=go.Layout(
        title=go.layout.Title(text='Raw Embeddings')
    ))
fig.show()