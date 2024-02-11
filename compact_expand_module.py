import torch
import torch.nn as nn

# Implementing the updated CompactExpandModule to handle batches and running the updated test function

class CompactExpandModule(nn.Module):
    def __init__(self, keep_token_ids, sequence_length, embedding_dimension, compacted_max_sequence_length):
        super(CompactExpandModule, self).__init__()
        self.keep_token_ids = keep_token_ids
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension
        self.compacted_max_sequence_length = compacted_max_sequence_length
        self.kept_positions_batch = None

    def forward(self, input_embeddings, token_ids=None, is_compacting=True):
        if is_compacting:
            return self._compact(input_embeddings, token_ids)
        else:
            return self._expand(input_embeddings)

    def _compact(self, input_embeddings, token_ids):
        assert self.kept_positions_batch is None, "Compact can only be called once before an expand"
        
        batch_size = input_embeddings.size(0)
        compacted_embeddings_batch = []
        kept_positions_batch = []

        # Determine the maximum number of tokens to keep across all sequences
        max_kept_tokens = 0

        for i in range(batch_size):
            keep_masks = torch.isin(token_ids[i], self.keep_token_ids)
            kept_positions = keep_masks.nonzero(as_tuple=True)[0]
            
            # Clamp the kept positions to the compacted_max_sequence_length
            if len(kept_positions) > self.compacted_max_sequence_length:
                kept_positions = kept_positions[:self.compacted_max_sequence_length]
            
            compacted_embeddings = input_embeddings[i][keep_masks][:self.compacted_max_sequence_length]
            
            compacted_embeddings_batch.append(compacted_embeddings)
            kept_positions_batch.append(kept_positions)
            
            # Update the maximum number of kept tokens if necessary
            max_kept_tokens = max(max_kept_tokens, len(compacted_embeddings))

        self.kept_positions_batch = kept_positions_batch

        # Padding the nested tensor to have uniform sequence lengths
        compacted_nested_tensor = torch.nested.nested_tensor(compacted_embeddings_batch)
        # Ensure the output size matches or exceeds the largest sequence length in the NestedTensor
        padded_output_size = (batch_size, max_kept_tokens, self.embedding_dimension)
        return torch.nested.to_padded_tensor(compacted_nested_tensor, 0.0, padded_output_size)


    def _expand(self, compacted_embeddings_batch) -> torch.nested.nested_tensor:
        assert self.kept_positions_batch is not None, "Expand can only be called after compact"
        
        batch_expanded_embeddings = []

        for compacted_embeddings, original_positions in zip(compacted_embeddings_batch, self.kept_positions_batch):
            original_length = max(original_positions) + 1 if len(original_positions) > 0 else 0
            expanded_embeddings = torch.zeros((original_length, self.embedding_dimension), dtype=compacted_embeddings.dtype, device=compacted_embeddings_batch.device)
            if len(original_positions) > 0:
                expanded_embeddings.scatter_(0, original_positions.unsqueeze(-1).expand(-1, self.embedding_dimension), compacted_embeddings)

            batch_expanded_embeddings.append(expanded_embeddings)

        self.kept_positions_batch = None
        expanded_nested_tensor = torch.nested.nested_tensor(batch_expanded_embeddings)
        # Output shape is [batch_size, sequence_length, embedding_dimension]
        return torch.nested.to_padded_tensor(expanded_nested_tensor, 0.0, (len(batch_expanded_embeddings), self.sequence_length, self.embedding_dimension))

# # Adjusted test function for batch processing
# def test_compact_expand_module_with_batches():
#     embedding_dimension = 4
#     sequence_length = 5
#     compacted_max_sequence_length = 8
#     keep_token_ids = [2, 3]  # Tokens to keep
#     module = CompactExpandModule(keep_token_ids, sequence_length, embedding_dimension, compacted_max_sequence_length)

#     # Test data for two batches
#     token_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 1, 3, 4, 2]])
#     input_embeddings = torch.arange(0., 10., step=0.25).reshape(2, 5, 4)

#     # Perform compacting
#     compacted_embeddings_batch = module(input_embeddings, token_ids=token_ids, is_compacting=True)

#     # Perform expanding
#     expanded_embeddings_batch = module(compacted_embeddings_batch, is_compacting=False)

#     print("Batch processing tests executed. Please verify results manually.")

# if __name__ == "__main__":
#     # Execute the batch processing test only if the script is run directly
#     test_compact_expand_module_with_batches()