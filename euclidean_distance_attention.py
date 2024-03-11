import numpy as np
import torch
from torch import nn

# Parameters
num_sequences = 100
max_length = 20
vocab_size = 50  # Number of unique tokens
embedding_dim = 32  # Standard embedding size
pos_embedding_dim = 2  # Position embedding size
num_layers=2


# Model
class CustomAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pos_embedding_dim, max_length, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.base_pos_embeddings = nn.Parameter(self.init_base_position_embeddings(max_length, pos_embedding_dim),
                                                requires_grad=False)
        self.pos_adjustment_layers = nn.ModuleList(
            [nn.Linear(pos_embedding_dim, pos_embedding_dim) for _ in range(num_layers)])
        self.linear = nn.Linear(embedding_dim, vocab_size)  # Output layer to predict the next token
        self.num_layers = num_layers

    def init_base_position_embeddings(self, max_length, pos_embedding_dim):
        positions = torch.arange(0., max_length).unsqueeze(1)
        sin_positions = torch.sin(0.5 * np.pi * positions / max_length)
        cos_positions = torch.cos(0.5 * np.pi * positions / max_length)
        pos_embeddings = torch.cat((sin_positions, cos_positions), dim=1)
        return pos_embeddings

    def forward(self, x):
        token_embeds = self.token_embedding(x)
        pos_embeds = self.base_pos_embeddings[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)

        for layer in self.pos_adjustment_layers:
            # Apply learned adjustments to position embeddings
            pos_adjustments = layer(pos_embeds)
            pos_embeds = pos_embeds + pos_adjustments

            # Compute attention using adjusted position embeddings
            dists = torch.cdist(pos_embeds, pos_embeds, p=2).pow(2)
            attention_scores = 1 / (dists + 1) # minimum distance starts at 1, so will never get an attention mask value larger than 1
            token_embeds = torch.bmm(attention_scores, token_embeds)  # Update token embeddings based on attention

        output = self.linear(token_embeds)
        return output


# Example use
model = CustomAttentionModel(vocab_size, embedding_dim, pos_embedding_dim, max_length, num_layers)

# Generate synthetic dataset for training
data = torch.randint(1, vocab_size, (num_sequences, max_length))

# Simple training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # Simple example, adjust epochs as needed
    for sequence in data:
        sequence = sequence.unsqueeze(0)  # Add batch dimension
        input_seq = sequence[:, :-1]  # All tokens except the last
        target_seq = sequence[:, 1:]  # All tokens except the first

        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.transpose(1, 2), target_seq)  # CrossEntropy expects input of (N, C, L)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')