import torch
import torch.nn as nn
import unittest

class Config:
    batch_size = 128
    seq_length = 256
    d_model = 256
    feed_forward_expand_dim = d_model * 4
    num_layers = 8
    num_heads = 4
    num_epochs = 10000
    checkpoint_path = ""
    vocab_size = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAD_IDX = 1
    EOS_IDX = 2
    BOS_IDX = 3

def unsqueeze_with_names(tensor: torch.Tensor, dim: int, names: tuple) -> torch.Tensor:
    tensor_unnamed = tensor.rename(None)
    tensor_unsqueezed = tensor_unnamed.unsqueeze(dim)
    return tensor_unsqueezed.refine_names(*names)

def repeat_with_names(tensor: torch.Tensor, repeats: tuple, names: tuple) -> torch.Tensor:
    tensor_unnamed = tensor.rename(None)
    tensor_repeated = tensor_unnamed.repeat(*repeats)
    return tensor_repeated.refine_names(*names)

def flatten_with_names(tensor: torch.Tensor, start_dim: int, end_dim: int, names: tuple) -> torch.Tensor:
    tensor_unnamed = tensor.rename(None)
    tensor_flattened = tensor_unnamed.flatten(start_dim, end_dim)
    return tensor_flattened.refine_names(*names)

class SentenceAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    def forward(self, x, attn_mask=None):
        assert x.names == ('batch', 'sequence', 'embedding')
        if (attn_mask is not None) and attn_mask.names != ('batch', 'sequence', 'sequence_mask'):
            raise AssertionError(f"Expected attn_mask names to be ('batch', 'sequence', 'sequence_mask'), got {attn_mask.names}")
        
        x_unnamed = x.rename(None)
        attn_mask_unnamed = attn_mask.rename(None) if attn_mask is not None else None
        
        output, _ = self.multihead_attn(x_unnamed, x_unnamed, x_unnamed, attn_mask=attn_mask_unnamed, need_weights=False)
        
        output = output.refine_names('batch', 'sequence', 'embedding')
        assert output.names == ('batch', 'sequence', 'embedding')
        return output

class SentenceProcessingLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        assert x.names == ('batch', 'sequence', 'embedding')
        x_unnamed = x.rename(None)
        output = self.norm(x_unnamed + self.linear2(self.activation(self.linear1(x_unnamed))))
        output = output.refine_names(*x.names)
        assert output.names == ('batch', 'sequence', 'embedding')
        return output

class HierarchicalSentenceTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.seq_length, config.d_model).refine_names('batch', 'sequence', 'embedding'))
        self.layers = nn.ModuleList([SentenceAwareAttention(config.d_model, config.num_heads) for _ in range(config.num_layers)])
        self.sentence_processing = SentenceProcessingLayer(config.d_model, config.feed_forward_expand_dim)
        self.norm = nn.LayerNorm(config.d_model)
        self.fc_layer = nn.Linear(config.d_model, config.vocab_size)

    def create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(self.config.device).refine_names('sequence', 'sequence_mask')
        assert mask.names == ('sequence', 'sequence_mask')
        return mask

    def create_sentence_mask(self, x):
        assert x.names == ('batch', 'sequence')
        eos_mask = (x == self.config.EOS_IDX).float()
        sentence_ids = torch.cumsum(eos_mask, dim=1)
        sentence_ids = sentence_ids.rename(None)
        mask = sentence_ids.unsqueeze(2) != sentence_ids.unsqueeze(1)
        mask = mask.refine_names('batch', 'sequence', 'sequence_mask')
        assert mask.names == ('batch', 'sequence', 'sequence_mask')
        return mask

    def forward(self, x):
        assert x.dim() == 2, f"Expected input to have 2 dimensions, got {x.dim()}"
        x = x.refine_names('batch', 'sequence')
        
        x_emb = self.embedding(x.rename(None)).refine_names('batch', 'sequence', 'embedding')
        assert x_emb.names == ('batch', 'sequence', 'embedding')
        
        x_emb = x_emb + self.pos_encoding
        
        causal_mask = self.create_causal_mask(x.size('sequence'))
        sentence_mask = self.create_sentence_mask(x)
        
        # Remove names for operations that do not support named tensors
        causal_mask_unnamed = causal_mask.rename(None)
        sentence_mask_unnamed = sentence_mask.rename(None)
        
        combined_mask = causal_mask_unnamed | sentence_mask_unnamed
        combined_mask = combined_mask.float().masked_fill(combined_mask == 1, float('-inf'))
        
        # Reassign names after operations
        combined_mask = combined_mask.refine_names('batch', 'sequence', 'sequence_mask')
        assert combined_mask.names == ('batch', 'sequence', 'sequence_mask')
        
        # Adjust the shape of the combined_mask for multi-head attention
        combined_mask = unsqueeze_with_names(combined_mask, 1, ('batch', 'num_heads', 'sequence', 'sequence_mask'))
        combined_mask = repeat_with_names(combined_mask, (1, self.config.num_heads, 1, 1), ('batch', 'num_heads', 'sequence', 'sequence_mask'))
        combined_mask = flatten_with_names(combined_mask, 0, 1, ('batch_head', 'sequence', 'sequence_mask'))
        
        for i, layer in enumerate(self.layers):
            x_emb = layer(x_emb, attn_mask=combined_mask)
            assert x_emb.names == ('batch', 'sequence', 'embedding')
            
            if i == len(self.layers) // 2:
                eos_mask = (x == self.config.EOS_IDX).any(dim='embedding')
                eos_tokens = x_emb[eos_mask]
                if eos_tokens.size('batch') > 0:
                    processed_eos = self.sentence_processing(eos_tokens)
                    x_emb[eos_mask] = processed_eos

        x_emb = self.norm(x_emb.rename(None)).refine_names('batch', 'sequence', 'embedding')
        output = self.fc_layer(x_emb.rename(None)).refine_names('batch', 'sequence', 'vocab')
        
        assert output.names == ('batch', 'sequence', 'vocab')
        return output

class TestHierarchicalSentenceTransformer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.model = HierarchicalSentenceTransformer(self.config).to(self.config.device)

    def test_forward_pass(self):
        input_tensor = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length)).to(self.config.device)
        input_tensor = input_tensor.refine_names('batch', 'sequence')
        input_tensor = input_tensor.rename(None)
        output = self.model(input_tensor)
        self.assertEqual(output.names, ('batch', 'sequence', 'vocab'))
        self.assertEqual(output.shape, (2, self.config.seq_length, self.config.vocab_size))

    def test_mask_creation(self):
        input_tensor = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length)).to(self.config.device)
        input_tensor = input_tensor.refine_names('batch', 'sequence')
        
        causal_mask = self.model.create_causal_mask(self.config.seq_length)
        self.assertEqual(causal_mask.names, ('sequence', 'sequence_mask'))
        self.assertEqual(causal_mask.shape, (self.config.seq_length, self.config.seq_length))
        
        sentence_mask = self.model.create_sentence_mask(input_tensor)
        self.assertEqual(sentence_mask.names, ('batch', 'sequence', 'sequence_mask'))
        self.assertEqual(sentence_mask.shape, (2, self.config.seq_length, self.config.seq_length))

    def test_combined_mask(self):
        input_tensor = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length)).to(self.config.device)
        input_tensor = input_tensor.refine_names('batch', 'sequence')
        
        causal_mask = self.model.create_causal_mask(self.config.seq_length)
        sentence_mask = self.model.create_sentence_mask(input_tensor)
        
        # Remove names for operations that do not support named tensors
        causal_mask_unnamed = causal_mask.rename(None)
        sentence_mask_unnamed = sentence_mask.rename(None)
        
        combined_mask = causal_mask_unnamed | sentence_mask_unnamed
        combined_mask = combined_mask.float().masked_fill(combined_mask == 1, float('-inf'))
        
        # Reassign names after operations
        combined_mask = combined_mask.refine_names('batch', 'sequence', 'sequence_mask')
        
        # Adjust the shape of the combined_mask for multi-head attention
        combined_mask = unsqueeze_with_names(combined_mask, 1, ('batch', 'num_heads', 'sequence', 'sequence_mask'))
        combined_mask = repeat_with_names(combined_mask, (1, self.config.num_heads, 1, 1), ('batch', 'num_heads', 'sequence', 'sequence_mask'))
        combined_mask = flatten_with_names(combined_mask, 0, 1, ('batch_head', 'sequence', 'sequence_mask'))
        
        self.assertEqual(combined_mask.names, ('batch_head', 'sequence', 'sequence_mask'))
        self.assertEqual(combined_mask.shape, (2 * self.config.num_heads, self.config.seq_length, self.config.seq_length))
        self.assertTrue(torch.isinf(combined_mask).any())

if __name__ == '__main__':
    unittest.main()