import torch
import torch.nn as nn

class SlidingAttentionModule(nn.Module):
    def __init__(self, sentence_breaking_token_ids, sequence_length, embedding_dimension, num_heads):
        super(SlidingAttentionModule, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dimension, num_heads=num_heads, batch_first=True)
        self.sentence_breaking_token_ids = sentence_breaking_token_ids
        self.embedding_dimension = embedding_dimension
        self.window_size = 16

    def forward(self, input_embeddings, token_ids):
        batch_size, seq_len, d_model = input_embeddings.shape
        
        # Generate a sliding window attention mask for each sequence in the batch
        attention_mask = self.generate_sliding_window_attention_masks(seq_len, self.window_size, input_embeddings.device)
        
        # Apply multihead attention with the sliding window attention mask
        attended_output, _ = self.multihead_attention(input_embeddings, input_embeddings, input_embeddings,
                                                      attn_mask=attention_mask)
        return attended_output

    def generate_sliding_window_attention_masks(self, seq_len, window_size, device):
        # Initialize the attention mask with False values
        attention_mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        # Populate the mask for sliding window attention
        for i in range(seq_len):
            left = max(i - window_size // 2, 0)
            right = min(i + window_size // 2 + 1, seq_len)
            attention_mask[i, left:right] = 0  # Set to 0 (or True) to allow attention within the window

        # The nn.MultiheadAttention expects a mask where 0 means "attend" and "-inf" means "do not attend"
        # So, we do not need to invert the mask values as we're setting "do not attend" to "-inf" directly
        return attention_mask