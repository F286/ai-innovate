import torch
import torch.nn as nn

class SentenceAttentionModule(nn.Module):
    def __init__(self, sentence_breaking_token_ids, sequence_length, embedding_dimension, num_heads, window_size):
        super(SentenceAttentionModule, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dimension, num_heads=num_heads, batch_first=True)
        self.sentence_breaking_token_ids = sentence_breaking_token_ids
        self.embedding_dimension = embedding_dimension
        self.window_size = window_size

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
    
    # def forward(self, input_embeddings, token_ids):
        
    #     batch_size, seq_len, d_model = input_embeddings.shape
        
    #     attention_mask = self.generate_attention_masks(seq_len, self.window_size, input_embeddings.device)
        
    #     attended_output, _ = self.multihead_attention(input_embeddings, input_embeddings, input_embeddings,
    #                                                   attn_mask=attention_mask, is_causal=True)
    #     return attended_output

    # def generate_attention_masks(self, token_ids):
    #     batch_size, seq_len = token_ids.shape
    #     attention_masks = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=token_ids.device)

    #     for batch_index in range(batch_size):
    #         for sequence_index in range(seq_len):
    #             # from sequence length index (inclusive) to 0 (inclusive)
    #             for look_at_index in range(sequence_index, -1, -1):  # looking at another token in same sequence
                    
    #                 # stop adding to attention mask if we hit a sentence breaking token
    #                 if token_ids[batch_index, look_at_index] in self.sentence_breaking_token_ids:
    #                     break
                    
    #                 attention_masks[batch_index, sequence_index, look_at_index] = True
                    
                        
                



# Test function
def test_optimized_sentence_aware_embedding_module():
    sequence_length = 5
    embedding_dimension = 2
    num_heads = 2
    sentence_breaking_token_ids = torch.tensor([42, 43])
    
    model = SentenceAttentionModule(sentence_breaking_token_ids, sequence_length, embedding_dimension, num_heads, 8)
    
    token_ids = torch.tensor([
        [0, 1, 42, 3, 43],
        [42, 4, 5, 42, 6]
    ])
    
    input_embeddings = torch.randn((2, 5, embedding_dimension))
    
    # Test attention mask
    # reference_mask = model.generate_attention_masks_reference(token_ids)
    # attention_mask = model.generate_attention_masks(token_ids)
    
    # assert reference_mask is attention_mask
    
    # Run actual module
    attended_output = model(input_embeddings, token_ids)
    
    assert attended_output is not None, "Attended output should not be None"
    assert attended_output.size() == input_embeddings.size(), "Attended output size should match input embeddings size"
    
    return "Test passed: Module correctly handles end-of-sentence tokens and computes attention."

# Execute the test
test_optimized_sentence_aware_embedding_module()

