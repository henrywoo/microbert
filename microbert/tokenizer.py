import re
import json
from typing import List, Dict, Optional
from collections import Counter
import torch


class WordTokenizer:
    """
    The WordTokenizer class provides a flexible and customizable tokenization tool for processing textual data.
    It encapsulates the logic to convert input sentences or lists of words into sequences of token IDs, suitable for nano BERT.
    This tokenizer adds special tokens, such as '[PAD]', '[CLS]', '[SEP]', '[UNK]', and '[MASK]', to the vocabulary,
    It offers methods to encode input text, decode token IDs back to human-readable sentences, and can be used as a callable object for quick tokenization.
    The class is designed to handle both single sentences and batches of sentences.
    
    Enhanced features:
    - Smart text cleaning and normalization
    - Dynamic vocabulary building from data
    - Frequency-based token filtering
    - Intelligent punctuation handling
    - Configurable vocabulary size limits
    """

    def __init__(self, 
                 vocab: Optional[List[str]] = None,
                 max_vocab_size: Optional[int] = None,
                 min_frequency: Optional[int] = None,
                 sep=' ', 
                 max_seq_len=16, 
                 special_tokens: Optional[List[str]] = None):
        # Call the constructor of the parent class (object)
        super().__init__()

        # Initialize tokenizer properties
        self.max_seq_len = max_seq_len  # Maximum sequence length for tokenization
        self.sep = sep  # Separator used to split input sentences into words
        self.max_vocab_size = max_vocab_size  # Maximum vocabulary size (None for unlimited)
        self.min_frequency = min_frequency  # Minimum token frequency (None for no filtering)
        
        # Store token frequencies for dynamic vocabulary management
        self.token_frequencies = Counter()
        
        # Define special tokens in proper order
        if special_tokens is None:
            self.special_tokens = [
                '[PAD]',
                '[CLS]',
                '[SEP]',
                '[UNK]',
                '[MASK]',  # Added for MLM pretraining
            ]
        else:
            self.special_tokens = special_tokens

        # Initialize vocabulary
        if vocab is not None:
            self._build_vocab_from_list(vocab)
        else:
            # Initialize with just special tokens
            self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
            self.de_vocab = {i: token for token, i in self.vocab.items()}

    def _build_vocab_from_list(self, vocab_list: List[str]) -> None:
        """Build vocabulary from a list of tokens with proper special token ordering."""
        # Start with special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add regular vocabulary tokens
        for i, token in enumerate(vocab_list):
            if token not in self.special_tokens:
                self.vocab[token] = i + len(self.special_tokens)
        
        # Create reverse vocabulary
        self.de_vocab = {i: token for token, i in self.vocab.items()}

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better tokenization."""
        # Normalize quotes and apostrophes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Handle contractions
        text = re.sub(r"n't", " n't", text)
        text = re.sub(r"'ll", " 'll", text)
        text = re.sub(r"'re", " 're", text)
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"'d", " 'd", text)
        text = re.sub(r"'s", " 's", text)
        
        # Clean up punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'([()])', r' \1 ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def _split_text(self, text: str) -> List[str]:
        """Split text into tokens with better handling of punctuation and contractions."""
        # Clean the text first
        text = self._clean_text(text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Further split tokens that contain punctuation
        final_tokens = []
        for token in tokens:
            # Skip if it's already a special token
            if token in self.special_tokens:
                final_tokens.append(token)
                continue
            
            # Split on punctuation boundaries
            sub_tokens = re.split(r'([.,!?;:()])', token)
            for sub_token in sub_tokens:
                if sub_token.strip():
                    final_tokens.append(sub_token.strip())
        
        return final_tokens

    def build_vocab_from_texts(self, texts: List[str], 
                              max_vocab_size: Optional[int] = None,
                              min_frequency: Optional[int] = None) -> None:
        """Build vocabulary from a list of text samples with dynamic sizing."""
        if max_vocab_size is None:
            max_vocab_size = self.max_vocab_size
        
        if min_frequency is None:
            min_frequency = self.min_frequency
        
        # Count token frequencies
        token_counts = Counter()
        for text in texts:
            tokens = self._split_text(text)
            token_counts.update(tokens)
        
        # Store frequencies for future use
        self.token_frequencies = token_counts
        
        # Apply frequency filtering if specified
        if min_frequency is not None:
            filtered_tokens = [token for token, count in token_counts.items() 
                              if count >= min_frequency]
        else:
            # Auto-determine frequency threshold based on data distribution
            total_tokens = sum(token_counts.values())
            avg_frequency = total_tokens / len(token_counts)
            min_freq = max(1, int(avg_frequency * 0.1))
            filtered_tokens = [token for token, count in token_counts.items() 
                              if count >= min_freq]
        
        # Sort by frequency (most frequent first)
        filtered_tokens.sort(key=lambda x: token_counts[x], reverse=True)
        
        # Apply vocabulary size limit if specified
        if max_vocab_size is not None:
            available_slots = max_vocab_size - len(self.special_tokens)
            selected_tokens = filtered_tokens[:available_slots]
        else:
            selected_tokens = filtered_tokens
        
        # Build vocabulary
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        for i, token in enumerate(selected_tokens):
            if token not in self.special_tokens:
                self.vocab[token] = i + len(self.special_tokens)
        
        # Update reverse vocabulary
        self.de_vocab = {i: token for token, i in self.vocab.items()}
        
        print(f"Built vocabulary: {len(self.vocab)} tokens")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Regular tokens: {len(self.vocab) - len(self.special_tokens)}")

    def _handle_rare_tokens(self, token: str) -> str:
        """Intelligently handle rare tokens with fallback strategies."""
        if token in self.vocab:
            return token
        
        # Try to find similar tokens
        similar_token = self._find_similar_token(token)
        if similar_token:
            return similar_token
        
        # Use [UNK] as last resort
        return '[UNK]'

    def _find_similar_token(self, token: str) -> Optional[str]:
        """Find the most similar known token using simple heuristics."""
        # Simple prefix matching
        for known_token in self.vocab:
            if known_token.startswith(token[:3]) or token.startswith(known_token[:3]):
                return known_token
        
        # Try lowercase matching
        lower_token = token.lower()
        if lower_token in self.vocab:
            return lower_token
        
        return None

    # Method to encode input sentence(s) into token IDs
    def encode(self, sentence: str | List[str]):
        """
        :param sentence: a string (will be split by 'sep') or a list of tokens (already split so each word will be encoded)
        :return: torch.Tensor((max_seq_len,), dtype=torch.long) - ids of encoded tokens
        """
        # If input is a string, use improved text splitting
        if isinstance(sentence, str):
            tokens = self._split_text(sentence)
        else:
            tokens = sentence

        # Truncate the input sentence to fit within the maximum sequence length
        tokens = tokens[:self.max_seq_len - 2]
        
        # Convert words to lowercase for better matching (keep special tokens as-is)
        processed_tokens = []
        for token in tokens:
            if token in self.special_tokens:
                processed_tokens.append(token)
            else:
                processed_tokens.append(token.lower())
        
        # Add special tokens ([CLS], [SEP]) and padding tokens ([PAD]) to the input sentence
        tokens = ['[CLS]'] + processed_tokens + ['[SEP]'] + ['[PAD]'] * (self.max_seq_len - len(processed_tokens) - 2)
        
        # Map words to their corresponding IDs in the vocabulary or use intelligent fallback
        token_ids = []
        for w in tokens:
            if w in self.vocab:
                token_ids.append(self.vocab[w])
            else:
                # Use intelligent token handling
                fallback_token = self._handle_rare_tokens(w)
                token_ids.append(self.vocab[fallback_token])
        
        return torch.tensor(token_ids, dtype=torch.long)

    # Method to decode token IDs back to original sentence
    def decode(self, ids: List, ignore_special=True):
        # Join token IDs to form a sentence, ignoring special tokens if specified
        return self.sep.join(
            [self.de_vocab[id] for id in ids if self.de_vocab[id] not in self.special_tokens or not ignore_special])

    # Callable method to allow using the object as a function for tokenization
    def __call__(self, sentence: str | List[str]):
        # If input is a string, tokenize it; otherwise, tokenize the list of words
        if isinstance(sentence, str):
            return self.encode(sentence.split(self.sep))
        return self.encode(sentence)

    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from a JSON file."""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.de_vocab = {i: token for token, i in self.vocab.items()}

    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'total_tokens': len(self.vocab),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.vocab) - len(self.special_tokens),
            'max_seq_len': self.max_seq_len,
            'max_vocab_size': self.max_vocab_size,
            'min_frequency': self.min_frequency
        }

    # Method to provide a string representation of the tokenizer object
    def __repr__(self):
        stats = self.get_vocab_stats()
        return f'WordTokenizer[vocab={stats["total_tokens"]}, special_tokens={len(self.special_tokens)}, max_len={self.max_seq_len}, sep={self.sep}]'
