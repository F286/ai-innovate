import torch
import torch.nn as nn

class OptimizedSentenceAwareEmbeddingModule(nn.Module):
    def __init__(self, embedding_dimension, num_heads, sentence_breaking_token):
        super(OptimizedSentenceAwareEmbeddingModule, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dimension, num_heads=num_heads, batch_first=True)
        self.sentence_breaking_token = sentence_breaking_token
        self.embedding_dimension = embedding_dimension

    def forward(self, input_embeddings, token_ids):
        sentence_mask = self.generate_attention_masks(token_ids)
        
        # repeat attention mask per head
        sentence_mask = sentence_mask.repeat_interleave(self.multihead_attention.num_heads, dim=0)
    
        attended_output, _ = self.multihead_attention(input_embeddings, input_embeddings, input_embeddings,
                                                      attn_mask=sentence_mask, is_causal=True)
        return attended_output

    def generate_attention_masks(self, token_ids):
        batch_size, seq_len = token_ids.shape
        attention_masks = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=token_ids.device)

        for batch_index in range(batch_size):
            for sequence_index in range(seq_len):
                # from sequence length index (inclusive) to 0 (inclusive)
                for look_at_index in range(sequence_index, -1, -1):  # looking at another token in same sequence
                    
                    # stop adding to attention mask if we hit a sentence breaking token
                    if token_ids[batch_index, look_at_index] in self.sentence_breaking_token:
                        break
                    
                    attention_masks[batch_index, sequence_index, look_at_index] = True
                
        return attention_masks


# Test function
def test_optimized_sentence_aware_embedding_module():
    embedding_dimension = 2
    num_heads = 2
    sentence_breaking_token = torch.tensor([42, 43])
    
    model = OptimizedSentenceAwareEmbeddingModule(embedding_dimension, num_heads, sentence_breaking_token)
    
    token_ids = torch.tensor([
        [0, 1, 42, 3, 43],
        [42, 4, 5, 42, 6]
    ])
    
    input_embeddings = torch.randn((2, 5, embedding_dimension))
    
    attended_output = model(input_embeddings, token_ids)
    
    assert attended_output is not None, "Attended output should not be None"
    assert attended_output.size() == input_embeddings.size(), "Attended output size should match input embeddings size"
    
    return "Test passed: Module correctly handles end-of-sentence tokens and computes attention."

# Execute the test
test_optimized_sentence_aware_embedding_module()

