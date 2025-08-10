from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import math



class IMDBDataloader:
    def __init__(self, data, test_data, tokenizer, label_encoder, batch_size, val_frac=0.2):
        train_data, val_data = train_test_split(data, shuffle=True, random_state=42, test_size=val_frac)

        self.splits = {
            'train': [d['text'] for d in train_data],
            'test': [d['text'] for d in test_data],
            'val': [d['text'] for d in val_data]
        }

        self.labels = {
            'train': [d['label'] for d in train_data],
            'test': [d['label'] for d in test_data],
            'val': [d['label'] for d in val_data]
        }

        self.tokenized = {
            'train': [tokenizer(record).unsqueeze(0) for record in
                      tqdm(self.splits['train'], desc='Train Tokenization')],
            'test': [tokenizer(record).unsqueeze(0) for record in tqdm(self.splits['test'], desc='Test Tokenization')],
            'val': [tokenizer(record).unsqueeze(0) for record in tqdm(self.splits['val'], desc='Val Tokenization')],
        }

        self.encoded_labels = {
            'train': [label_encoder(label) for label in tqdm(self.labels['train'], desc='Train Label Encoding')],
            'test': [label_encoder(label) for label in tqdm(self.labels['test'], desc='Test Label Encoding')],
            'val': [label_encoder(label) for label in tqdm(self.labels['val'], desc='Val Label Encoding')],
        }

        self.curr_batch = 0
        self.batch_size = batch_size
        self.iterate_split = None

    def peek(self, split):
        return {
            'input_ids': self.splits[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
            'label_ids': self.labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
        }

    def take(self, split):
        batch = self.splits[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        labels = self.labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        self.curr_batch += 1
        return {
            'input_ids': batch,
            'label_ids': labels,
        }

    def peek_tokenized(self, split):
        return {
            'input_ids': torch.cat(
                self.tokenized[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
                dim=0),
            'label_ids': torch.tensor(
                self.encoded_labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
                dtype=torch.long),
        }

    def peek_index_tokenized(self, index, split):
        return {
            'input_ids': torch.cat(
                [self.tokenized[split][index]],
                dim=0),
            'label_ids': torch.tensor(
                [self.encoded_labels[split][index]],
                dtype=torch.long),
        }

    def peek_index(self, index, split):
        return {
            'input_ids': [self.splits[split][index]],
            'label_ids': [self.labels[split][index]],
        }

    def take_tokenized(self, split):
        batch = self.tokenized[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        labels = self.encoded_labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        self.curr_batch += 1
        return {
            'input_ids': torch.cat(batch, dim=0),
            'label_ids': torch.tensor(labels, dtype=torch.long),
        }

    def get_split(self, split):
        self.iterate_split = split
        return self

    def steps(self, split):
        return len(self.tokenized[split]) // self.batch_size

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.batch_size * self.curr_batch < len(self.splits[self.iterate_split]):
            return self.take_tokenized(self.iterate_split)
        else:
            raise StopIteration

    def reset(self):
        self.curr_batch = 0



def plot_results(history, do_val=True, save_path=None):
    """
    Plot training results with loss, accuracy, and F1 score
    
    Args:
        history: Dictionary containing training metrics
        do_val: Whether to plot validation metrics
        save_path: Optional path to save the plot image
    """
    # Set style for better looking plots
    plt.style.use('default')
    
    # Create subplots with better layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Progress Metrics', fontsize=16, fontweight='bold', y=0.95)
    
    x = list(range(0, len(history['train_losses'])))
    
    # Plot 1: Loss
    axes[0].plot(x, history['train_losses'], label='Train Loss', linewidth=2, color='#2E86AB', marker='o', markersize=4)
    if do_val:
        axes[0].plot(x, history['val_losses'], label='Validation Loss', linewidth=2, color='#A23B72', marker='s', markersize=4)
    
    axes[0].set_title('Loss Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    # Plot 2: Accuracy
    axes[1].plot(x, history['train_acc'], label='Train Accuracy', linewidth=2, color='#18A558', marker='o', markersize=4)
    if do_val:
        axes[1].plot(x, history['val_acc'], label='Validation Accuracy', linewidth=2, color='#F7931E', marker='s', markersize=4)
    
    axes[1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    
    # Plot 3: F1 Score
    axes[2].plot(x, history['train_f1'], label='Train F1', linewidth=2, color='#6A4C93', marker='o', markersize=4)
    if do_val:
        axes[2].plot(x, history['val_f1'], label='Validation F1', linewidth=2, color='#FF6B6B', marker='s', markersize=4)
    
    axes[2].set_title('F1 Score Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_facecolor('#f8f9fa')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Training results plot saved to: {save_path}')
    
    # Display the plot (works in both Jupyter and regular Python)
    try:
        # Check if we're in a Jupyter environment
        get_ipython()
        # In Jupyter, use display() for better integration
        from IPython.display import display
        display(fig)
    except NameError:
        # Not in Jupyter, use regular plt.show()
        plt.show()
    
    return fig


def plot_mlm_results(history, save_path=None):
    """
    Plot MLM training results using the same styling as plot_results
    
    Args:
        history: Dictionary containing 'train_losses' and 'val_losses' lists
        save_path: Optional path to save the plot image
    """
    # Set style for better looking plots
    plt.style.use('default')
    
    # Create subplots with better layout
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('MLM Training Progress', fontsize=16, fontweight='bold', y=0.95)
    
    x = list(range(1, len(history['train_losses']) + 1))
    
    # Plot 1: Loss
    axes[0].plot(x, history['train_losses'], label='Train Loss', linewidth=2, color='#2E86AB', marker='o', markersize=6)
    axes[0].plot(x, history['val_losses'], label='Validation Loss', linewidth=2, color='#A23B72', marker='s', markersize=6)
    
    axes[0].set_title('Loss Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    # Plot 2: Loss Difference (Overfitting Analysis)
    loss_diff = [abs(train - val) for train, val in zip(history['train_losses'], history['val_losses'])]
    axes[1].plot(x, loss_diff, label='|Train - Val| Loss', linewidth=2, color='#18A558', marker='^', markersize=6)
    
    axes[1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold', pad=20)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss Difference', fontsize=12)
    axes[1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Plot saved to: {save_path}')
    
    # Display the plot (works in both Jupyter and regular Python)
    try:
        # Check if we're in a Jupyter environment
        get_ipython()
        # In Jupyter, use display() for better integration
        from IPython.display import display
        display(fig)
    except NameError:
        # Not in Jupyter, use regular plt.show()
        plt.show()
    
    return fig


def get_attention_scores(model, input_ids):
    """
    This is just a wrapper to easily access attention heads of the last layer
    """
    mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1)
    embed = model.micro_bert.embedding(input_ids)
    # can be any layer, and we can also control what to do with output for each layer (aggregate, sum etc.)
    layer = model.micro_bert.encoder.layers[-1]
    x = layer.layer_norm1(embed)
    B, seq_len, n_embed = x.shape
    # if have more than 1 head, or interested in more than 1 head output just add aggregation here
    head = layer.self_attention.heads[0]
    # this is just a part of the single head that does all the computations (same code is present in AttentionHead)
    q = head.query(x)
    k = head.key(x)
    v = head.values(x)
    weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])  # (B, Seq_len, d_k)
    # Use a value compatible with float16 (Half) type
    neg_inf = torch.finfo(weights.dtype).min
    weights = weights.masked_fill(mask == 0, neg_inf)  # mask out not attended tokens
    scores = F.softmax(weights, dim=-1)
    return scores


def plot_parallel(matrix, tokens, show_all_connections=False):
    """
    Visualize attention scores between tokens in a parallel coordinate plot.

    This function creates a visualization showing how much each token attends to other tokens
    in the sequence. The attention scores are represented by the thickness of lines connecting
    tokens on two parallel vertical lines (A and B).

    Args:
        matrix: Attention score matrix of shape (seq_len, seq_len)
        tokens: List of token strings corresponding to the sequence
        show_all_connections: If True, show all token connections. If False, only show first token connections.
    """
    # Set style for better looking plots
    plt.style.use('default')
    
    # Convert tensor to numpy if needed
    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().detach().numpy()
    elif hasattr(matrix, 'numpy'):
        matrix = matrix.numpy()
    
    # Set figure size and style
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    input_len = len(tokens)
    
    # Create a color gradient for attention scores
    max_score = matrix.max()
    min_score = matrix.min()
    
    # Draw two vertical parallel lines (A and B) with better styling
    ax.axvline(x=1, color='#2E86AB', linestyle='-', linewidth=3, alpha=0.8)
    ax.axvline(x=5, color='#A23B72', linestyle='-', linewidth=3, alpha=0.8)
    
    # Add labels for the parallel lines
    ax.text(1, input_len + 0.5, 'A', fontsize=16, color='#2E86AB', fontweight='bold', 
            ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(5, input_len + 0.5, 'B', fontsize=16, color='#A23B72', fontweight='bold', 
            ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot attention connections between tokens
    for i in range(input_len):
        for j in range(input_len):
            # Skip if not showing all connections and this is not the first token
            if not show_all_connections and i > 0:
                continue
                
            attention_score = matrix[i][j]
            
            # Normalize attention score for color and linewidth
            normalized_score = (attention_score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
            
            # Create color gradient from blue to red based on attention score
            color = plt.cm.RdYlBu(normalized_score)
            
            # Draw lines connecting tokens with thickness proportional to attention score
            linewidth = 2 + 8 * normalized_score
            alpha = 0.3 + 0.7 * normalized_score
            
            ax.plot([1, 5], [i, j], marker='o', color=color, linewidth=linewidth, alpha=alpha,
                   markersize=6, markeredgecolor='white', markeredgewidth=1)
            
            # Add token labels on line A (left side) - only for first token or when showing all
            if i == 0 or show_all_connections:
                ax.text(
                    1 - 0.25,  # x-axis position
                    i,  # y-axis position
                    tokens[i],  # Token text
                    fontsize=10,  # Text size
                    color='#2E86AB',  # Text color
                    fontweight='bold',
                    ha='right',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='#2E86AB')
                )
            
            # Add token labels on line B (right side)
            ax.text(
                5 + 0.25,  # x-axis position
                j,  # y-axis position
                tokens[j],  # Token text
                fontsize=10,  # Text size
                color='#A23B72',  # Text color
                fontweight='bold',
                ha='left',
                va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='#A23B72')
            )
    
    # Add title and styling
    ax.set_title('Attention Mechanism Visualization', fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
    ax.set_xlabel('Token Positions', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Token Index', fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle='--', color='#95a5a6')
    
    # Set axis limits and ticks
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.5, input_len - 0.5)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(['Position A', 'Position B'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(input_len))
    ax.set_yticklabels([f'Token {i+1}' for i in range(input_len)], fontsize=10)
    
    # Add colorbar to show attention score scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(min_score, max_score))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Attention Score', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot (works in both Jupyter and regular Python)
    try:
        # Check if we're in a Jupyter environment
        get_ipython()
        # In Jupyter, use display() for better integration
        from IPython.display import display
        display(fig)
    except NameError:
        # Not in Jupyter, use regular plt.show()
        plt.show()
    
    return fig


def save_model(model, tokenizer, history, config, save_dir):
    """
    Save trained model and related files to specified directory

    Args:
        model: trained model
        tokenizer: tokenizer
        history: training history
        config: model configuration
        save_dir: save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, 'microbert_classification.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')

    # Save tokenizer vocabulary
    tokenizer_path = os.path.join(save_dir, 'tokenizer_vocab.json')
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(list(tokenizer.vocab), f, ensure_ascii=False, indent=2)
    print(f'Tokenizer vocabulary saved to: {tokenizer_path}')

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'Training history saved to: {history_path}')

    # Save model configuration
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'Model configuration saved to: {config_path}')

    print(f'\nAll files saved to directory: {save_dir}')


def load_model(model_dir, device=None):
    """
    Load trained model from specified directory

    Args:
        model_dir: directory containing model files
        device: device (cpu/cuda)

    Returns:
        model: loaded model
        tokenizer: loaded tokenizer
        config: model configuration
    """
    # Import here to avoid circular imports
    from microbert.model import MicroBERTForClassification
    from microbert.tokenizer import WordTokenizer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Load vocabulary
    vocab_path = os.path.join(model_dir, 'tokenizer_vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = set(json.load(f))

    # Create tokenizer
    tokenizer = WordTokenizer(vocab=vocab, max_seq_len=config['max_seq_len'])

    # Create model
    model = MicroBERTForClassification(
        vocab_size=config['vocab_size'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        n_classes=config['n_classes']
    ).to(device)

    # Load model weights
    model_path = os.path.join(model_dir, 'microbert_classification.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f'Model loaded from {model_dir}')
    return model, tokenizer, config


def predict_sentiment(model, tokenizer, text, device=None):
    """
    Predict sentiment using trained model
    
    Args:
        model: trained model
        tokenizer: tokenizer
        text: input text
        device: device
    
    Returns:
        prediction: prediction result (0: negative, 1: positive)
        confidence: confidence score
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        # Tokenize
        input_ids = tokenizer(text).unsqueeze(0).to(device)
        
        # Predict
        logits = model(input_ids)
        probs = torch.softmax(logits[:, 0, :], dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, prediction].item()
        
        return prediction, confidence
    
    
