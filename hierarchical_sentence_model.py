import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional
import re

class SpecialToken(Enum):
    PAD = auto()
    EOS = auto()
    BOS = auto()

@dataclass
class Config:
    batch_size: int = 2
    max_sequence_length: int = 512
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 6
    dim_feedforward: int = 3072
    dropout: float = 0.1

class BooleanMask:
    def __init__(self, size: Tuple[int, int, int]):
        self.size = size
        self.data = torch.ones(size, dtype=torch.bool)

    def __getitem__(self, key: Tuple[int, int, int]) -> bool:
        return self.data[key]

    def __setitem__(self, key: Tuple[int, int, int], value: bool) -> None:
        self.data[key] = value

    def __str__(self) -> str:
        return "\n".join(
            "\n".join("".join("□" if self[b, i, j] else "■" for j in range(self.size[2])) 
                      for i in range(self.size[1]))
            for b in range(self.size[0])
        )

    def visualize(self, title: str, tokens_batch: List[List[str]], sentences: Optional[List[str]] = None) -> None:
        print(f"\n{title}")
        
        if sentences:
            for i, sentence in enumerate(sentences):
                print(f"Sentence {i + 1}: {sentence}")
        
        for batch_idx, tokens in enumerate(tokens_batch):
            print(f"\nBatch {batch_idx + 1}:")
            
            # Calculate the width of each cell based on the longest token
            cell_width = max(len(token) for token in tokens) + 2
            
            # Print the tokens
            print(" " * 4 + "".join(f"{token:^{cell_width}}" for token in tokens))
            
            # Print the mask with corresponding letters/tokens
            for i in range(self.size[1]):
                row = "".join("□" if self[batch_idx, i, j] else "■" for j in range(self.size[2]))
                print(f"{tokens[i]:>3} {row}")

    def get_pytorch_mask(self) -> torch.Tensor:
        return self.data

class IsolatedCausalMask:
    @staticmethod
    def create_mask(tokenized_batch: List[List[int]], split_token: int, pad_token: int) -> BooleanMask:
        batch_size = len(tokenized_batch)
        max_length = max(len(seq) for seq in tokenized_batch)
        mask = BooleanMask((batch_size, max_length, max_length))
        
        for batch_idx, sequence in enumerate(tokenized_batch):
            current_position = 0
            for i, token in enumerate(sequence):
                for j in range(current_position, i + 1):
                    mask[batch_idx, i, j] = False
                if token == split_token:
                    current_position = i + 1
                elif token == pad_token:
                    break
        
        return mask


import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

class SpecialToken(Enum):
    PAD = auto()
    EOS = auto()
    BOS = auto()


class TokenHandler:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._initialize_vocab()

    def _initialize_vocab(self) -> None:
        special_tokens = [token.name for token in SpecialToken]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        for i in range(32, 127):  # ASCII printable characters
            char = chr(i)
            self.vocab[char] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = char
        
        # Add underscore as space representation
        self.vocab['_'] = len(self.vocab)
        self.inverse_vocab[len(self.inverse_vocab)] = '_'

    @torch.jit.export
    def tokenize(self, text: str, split_on_space: bool = False) -> List[int]:
        tokens = [self.vocab['BOS']]
        for char in text:
            if char == ' ':
                tokens.append(self.vocab['_'])
            elif char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['PAD'])  # Use PAD for unknown characters
            
            if char in '.!?' and not split_on_space:
                tokens.append(self.vocab['EOS'])
        
        if split_on_space:
            tokens.append(self.vocab['EOS'])
        return tokens

    def tokenize_batch(self, sentences: List[str], split_on_space: bool = False) -> List[List[int]]:
        return [self.tokenize(sentence, split_on_space) for sentence in sentences]

    def pad_sequences(self, sequences: List[List[int]], pad_token: int) -> List[List[int]]:
        max_length = max(len(seq) for seq in sequences)
        return [seq + [pad_token] * (max_length - len(seq)) for seq in sequences]

    @torch.jit.export
    def detokenize(self, tokens: List[int]) -> str:
        # Convert list to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create a mask for special tokens
        special_mask = (tokens_tensor != self.vocab['BOS']) & (tokens_tensor != self.vocab['EOS']) & (tokens_tensor != self.vocab['PAD'])
        
        # Filter out special tokens
        filtered_tokens = tokens_tensor[special_mask]
        
        # Convert underscore to space
        filtered_tokens = torch.where(filtered_tokens == self.vocab['_'], torch.tensor(32), filtered_tokens)
        
        # Convert to list of integers with explicit type annotation
        result: List[int] = filtered_tokens.tolist()
        
        # Convert to string
        return ''.join([chr(t) if t < 128 else self.inverse_vocab[t] for t in result])

    
# Use TorchScript to optimize TokenHandler
token_handler_script = torch.jit.script(TokenHandler())


class HierarchicalSentenceTransformer(nn.Module):
    def __init__(self, config: Config, token_handler: TokenHandler):
        super().__init__()
        self.config = config
        self.token_handler = token_handler
        
        self.embedding = nn.Embedding(len(token_handler.vocab), config.d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=config.num_layers)
        self.fc_out = nn.Linear(config.d_model, len(token_handler.vocab))

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        print(f"Input shape: {src.shape}")
        print(f"Input tensor:\n{src}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask tensor:\n{mask}")
        
        embedded = self.embedding(src)
        print(f"Embedded shape: {embedded.shape}")
        
        # Use the mask directly without reshaping
        transformer_out = self.transformer(embedded, src_key_padding_mask=mask)
        print(f"Transformer output shape: {transformer_out.shape}")
        
        output = self.fc_out(transformer_out)
        print(f"Final output shape: {output.shape}")
        
        return output

def test_batched_sentence_isolated_causal_mask(model: HierarchicalSentenceTransformer, token_handler: TokenHandler) -> bool:
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs."
    ]
    
    print("Input sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Test with space-based splitting
    tokenized_space = token_handler.tokenize_batch(sentences, split_on_space=True)
    padded_space = token_handler.pad_sequences(tokenized_space, token_handler.vocab['PAD'])
    
    print("Tokenized and padded input:")
    for tokens in padded_space:
        print([token_handler.inverse_vocab.get(token, f"<{token}>") for token in tokens])
    
    # Prepare input tensors
    input_tensor_space = torch.tensor(padded_space, dtype=torch.long)
    
    # Create a simple mask (False for padding, True for non-padding)
    mask = (input_tensor_space != token_handler.vocab['PAD'])
    
    print(f"Input tensor shape: {input_tensor_space.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Run forward pass
    try:
        with torch.no_grad():
            output_space = model(input_tensor_space, mask)
        
        print(f"Model output shape: {output_space.shape}")
        print("Forward pass successful!")
        return True
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main() -> None:
    config = Config()
    token_handler = TokenHandler()
    model = HierarchicalSentenceTransformer(config, token_handler)

    # Test batched sentence-isolated causal mask
    test_result = test_batched_sentence_isolated_causal_mask(model, token_handler)
    
    if test_result:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above for details.")

if __name__ == '__main__':
    main()