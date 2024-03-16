import os
from multiprocessing import freeze_support
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Adafactor
from torch.utils.tensorboard import SummaryWriter
import math

from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, trainers, normalizers, pre_tokenizers
import pickle
from tokenizers.models import BPE

from torch.cuda.amp import autocast, GradScaler

class Config:
    def __init__(self):
        self.batch_size = 128
        self.seq_length = 256  # The length to pad or truncate to
        self.n_embed = 64
        self.feed_forward_expand_dim = self.n_embed * 2
        self.num_layers = 2
        self.n_head = 2
        self.num_epochs = 10000
        self.checkpoint_directory = "softmax_auto_encoder/"
        self.checkpoint_filename = "none"
        self.log_directory = self.checkpoint_directory
        self.checkpoint_save_every_epochs = 1
        self.vocab_size = 4096
        self.attn_pdrop = 0.1  # Attention dropout
        self.resid_pdrop = 0.1  # Residual dropout
        self.block_size = self.seq_length  # For causal mask in self-attention


config: Config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 1

class ForwardConfig:
    def __init__(self):
        self.additional_loss = 0

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embed


        self.custom_pathway = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(
                in_channels=config.num_heads,  # Input channels
                out_channels=config.n_embed,  # Arbitrary choice, can be adjusted
                kernel_size=32,
                padding=16,  # Padding can be adjusted based on your requirements
                groups=config.num_heads  # Using groups=num_heads if you wish to apply a separate convolution for each head
            ),
            nn.ReLU(),  # Non-linear activation function
            # Second convolutional layer
            nn.Conv1d(
                in_channels=config.n_embed,  # Must match the out_channels of the previous layer
                out_channels=config.n_embed,  # This can be the same as above or different, depending on your design
                kernel_size=32,
                padding=16,  # Adjust padding as necessary
                groups=config.num_heads  # Maintaining separate convolutions per head
            ),
            nn.ReLU()  # Another non-linearity
        )
# Then, inside forward


    def forward(self, x, forward_config:ForwardConfig):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side


        # Custom softmax
        softmax_att = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Inside forward, replace the custom_att line with:
        
        # Assuming att has shape: [batch_size, num_heads, seq_length, seq_length]
        batch_size, num_heads, seq_length, _ = att.size()

        # Reshape for Conv1d to treat each head as a separate channel and each sequence position as a separate sample.
        # Merge batch and sequence length dimensions, treat num_heads as channels.
        att_reshaped = att.transpose(2, 3).reshape(batch_size * seq_length, num_heads, seq_length)

        # Define your convolutional layers to expect `num_heads` as in_channels.
        # Assuming self.custom_pathway is an nn.Sequential containing Conv1d layers configured with in_channels=num_heads.
        custom_att = self.custom_pathway(att_reshaped)

        # Assuming you want to reshape custom_att back to its original form or another desired shape,
        # you should adjust the reshape operation accordingly.



        custom_att = F.softmax(custom_att, dim=-1)  # Then apply softmax
        custom_att = self.attn_dropout(custom_att)  # Apply dropout

        # Compute additional loss (no changes here, just for context)
        forward_config.additional_loss += F.mse_loss(custom_att, softmax_att.detach())

        # Modify the output to use the custom pathway
        y = custom_att @ v  # Changed from softmax_att to custom_att



        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config:Config):
        super(Block, self).__init__()

        self.layer = CausalSelfAttention(config)

    def forward(self, x, forward_config:ForwardConfig):
        # Apply Transformer block
        output = self.layer(x, forward_config)
        # In the original Transformer, output and input are combined inside the Transformer block.
        return output


class TransformerModel(nn.Module):
    def __init__(self, config: Config):
        super(TransformerModel, self).__init__()

        self.d_model = config.n_embed  # Adjusted to use n_embed

        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, self.d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)])

        # Define a sequential layer for expansion, ReLU, and expansion to vocab_size
        self.fc_layer = nn.Sequential(
            nn.Linear(self.d_model, config.feed_forward_expand_dim),  # Adjusted to use feed_forward_expand_dim
            nn.ReLU(),
            nn.Linear(config.feed_forward_expand_dim, config.vocab_size)  # Adjusted to use feed_forward_expand_dim
        )

    def forward(self, x, forward_config:ForwardConfig):
        x = self.embedding(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, forward_config)

        x = self.fc_layer(x)

        return x




class TensorBoardCheckpointWriter:
    def __init__(self, log_dir=None, checkpoint_dir="checkpoints"):
        if log_dir is None:
            log_dir = f"runs/train_{int(time.time())}"
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 0  # Initialize internal epoch counter
        self.global_step = 0  # Initialize internal global step counter
        print(f"TensorBoard logs will be saved to {log_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        return False

    def add_scalar(self, tag, scalar_value):
        self.writer.add_scalar(tag, scalar_value, self.global_step, time.time())
        
    def add_text(self, tag, text_string):
        self.writer.add_text(tag, text_string, self.global_step, time.time())

    def save_checkpoint(self, model, optimizer, additional_info=None):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pt")
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': self.global_step
        }
        if additional_info is not None:
            checkpoint.update(additional_info)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, filename):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint.get('epoch', -1)
            self.global_step = checkpoint.get('global_step', 0) 
            print(f"Checkpoint loaded from {checkpoint_path}, epoch {self.epoch}, global step {self.global_step}")
            return self.epoch
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return -1

    def increment_epoch(self):
        self.epoch += 1

    def increment_global_step(self):
        self.global_step += 1

def evaluate(tokenizer, model, writer_checkpoint_manager:TensorBoardCheckpointWriter):        
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
    
        # Generate text after each epoch
        seed_text = "Once upon a time a blue"
        seed_tokens = tokenizer.encode(seed_text).ids

        # Find <bos> token ID
        bos_token_id = tokenizer.token_to_id("<bos>")
        # Prepend <bos> token ID to the seed_tokens
        seed_tokens = [bos_token_id] + seed_tokens

        input_text = torch.tensor([seed_tokens]).long().to(device)

        for _ in range(220):
            forward_config = ForwardConfig()

            predictions = model(input_text, forward_config)
            next_token_idx = predictions[:, -1, :].argmax(dim=-1)
            input_text = torch.cat([input_text, next_token_idx.unsqueeze(0)], dim=1)

        # Decode the generated tokens using the tokenizer
        generated_text = tokenizer.decode(input_text[0].cpu().tolist())

        writer_checkpoint_manager.add_text("Evaluate/text", generated_text)
        print(f"\n{generated_text}")

        print('')

def validate(model, val_loader, criterion, device, writer_checkpoint_manager:TensorBoardCheckpointWriter):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_tokens = 0  # Keep track of the total number of tokens processed

    with torch.no_grad():  # No gradients needed for validation
        for text, target in val_loader:
            text = text.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            forward_config = ForwardConfig()

            predictions = model(text, forward_config)
            predictions = predictions.view(-1, predictions.size(2))
            target = target.view(-1)

            loss = criterion(predictions, target)
            total_loss += loss.item() * target.size(0)  # Multiply loss by the number of tokens in target
            total_tokens += target.size(0)  # Accumulate total tokens
            
            if total_tokens > 1000000:
                break

    # Normalize the total loss by the total number of tokens processed
    avg_loss = total_loss / total_tokens
    print(f"Average Validation Loss: {avg_loss:.4f}")
    writer_checkpoint_manager.add_scalar("Loss/validate", avg_loss)

def train(train_loader, model, criterion, optimizer, scaler, writer_checkpoint_manager:TensorBoardCheckpointWriter):
    model.train()

    start_time_total = time.time()

    total_loss = 0.0
    for i, (text, target) in enumerate(train_loader):
        with autocast():
            start_time = time.time()
            tokens_processed = 0

            text = text.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            assert text.shape[0] == target.shape[0], f"Batch size mismatch: text {text.shape}, target {target.shape}"

            # Increment the token counter
            tokens_processed += config.batch_size * config.seq_length

            forward_config = ForwardConfig()

            # Run the model
            prediction = model(text, forward_config)
            
            # Prediction
            loss = forward_config.additional_loss
            
            assert prediction.shape[0] == text.shape[
                    0], f"Predictions batch size mismatch: predictions {prediction.shape}, text {text.shape}"

            prediction = prediction.view(-1, prediction.size(2))
            target = target.view(-1)
            assert prediction.size(0) == target.size(
                    0), f"Predictions and target size mismatch after reshaping: predictions {prediction.size(0)}, target {target.size(0)}"

            # Compute loss
            loss += criterion(prediction, target)
            total_loss += loss.item()

            # Scale the loss and compute the gradients
            scaler.scale(loss).backward()

            # Update the weights using the scaled gradients
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            # Calculate tokens per second for this batch
            elapsed_time_total = time.time() - start_time_total
            elapsed_time = time.time() - start_time
            tokens_per_second = tokens_processed / elapsed_time
            avg_loss = total_loss / (i + 1)
            epoch_progress = (i + 1) / len(train_loader) * 100

            if writer_checkpoint_manager.global_step % 10 == 0:
                writer_checkpoint_manager.add_scalar("Loss/train", avg_loss)
                writer_checkpoint_manager.add_scalar("Tokens per second (k)", tokens_per_second / 1000)
            
            writer_checkpoint_manager.increment_global_step()


def create_tokenizer_function(tokenizer):
    def tokenize_data(example):
        text = example['text']
        # Encode the text using the tokenizer
        encoding = tokenizer.encode(text)
        tokens = encoding.ids
        # Add <bos> and <eos> tokens
        bos_token_id = tokenizer.token_to_id("<bos>")
        eos_token_id = tokenizer.token_to_id("<eos>")
        tokens = [bos_token_id] + tokens + [eos_token_id]
        return {'input_ids': tokens}
    return tokenize_data


def collate_batch(batch):
    new_seq_length = config.seq_length

    # Filter out short sequences
    valid_items = [item['input_ids'] for item in batch if len(item['input_ids']) >= 16]

    # Truncate or pad sequences
    processed_texts = [torch.tensor(item[:new_seq_length], dtype=torch.long) for item in valid_items]
    processed_targets = [torch.tensor(item[1:new_seq_length + 1], dtype=torch.long) for item in valid_items]

    # Stack lists into tensors, padding where necessary
    text_tensor = torch.nn.utils.rnn.pad_sequence(processed_texts, batch_first=True, padding_value=PAD_IDX)
    target_tensor = torch.nn.utils.rnn.pad_sequence(processed_targets, batch_first=True, padding_value=PAD_IDX)

    return text_tensor, target_tensor


def print_parameter_count(model):
    param_counts = {}
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            param_counts[name] = num_params

    print(f"Total Trainable Parameters: {total_params}")


def load_or_create_tokenizer_and_vocab(train_dataset):
    tokenizer_filename = "tokenizer.pkl"

    # Try to load tokenizer from files
    try:
        with open(tokenizer_filename, "rb") as file:
            tokenizer = pickle.load(file)
        print("Loaded tokenizer from files.")
    except FileNotFoundError:
        # Create BPE tokenizer
        tokenizer = Tokenizer(BPE())

        # Normalizer
        tokenizer.normalizer = normalizers.Lowercase()

        # Pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Extract text data and write to file
        train_text_data = [item['text'] for item in train_dataset]
        with open("train_text.txt", "w", encoding="utf-8") as f:
            for text in train_text_data:
                f.write(text + "\n")

        # Train the tokenizer
        trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
        tokenizer.train(files=["train_text.txt"], trainer=trainer)

        # Save tokenizer to files
        with open(tokenizer_filename, "wb") as file:
            pickle.dump(tokenizer, file)
        print("Created and saved tokenizer to files.")

    return tokenizer


def main():
    print("Using device:", device)
    
    # Load the TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # Tokenizer and vocab functions
    tokenizer = load_or_create_tokenizer_and_vocab(train_dataset)

    # Create the tokenize function with the specific tokenizer and vocab
    tokenize_func = create_tokenizer_function(tokenizer)

    # Use this function to tokenize the dataset
    train_dataset = train_dataset.map(tokenize_func, cache_file_name="tokenized_train")
    validation_dataset = validation_dataset.map(tokenize_func, cache_file_name="tokenized_train")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=16, pin_memory=True, prefetch_factor=8, drop_last=True)
    
    val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=16, pin_memory=True, prefetch_factor=8, drop_last=True)

    # Model definition
    assert(config.vocab_size == tokenizer.get_vocab_size())
    
    model = TransformerModel(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    print_parameter_count(model)

    with TensorBoardCheckpointWriter(log_dir=config.log_directory, checkpoint_dir=config.checkpoint_directory) as writer_checkpoint_manager:
        # Your training/validation loop and any checkpoint saving/loading logic
        # Use writer_checkpoint_manager for logging and checkpointing

        writer_checkpoint_manager.load_checkpoint(model, optimizer, config.checkpoint_filename)

        scaler = GradScaler()

        while writer_checkpoint_manager.epoch < config.num_epochs:
            
            print(f"\n-- Epoch {writer_checkpoint_manager.epoch} --\n")
            
            validate(model, val_loader, criterion, device, writer_checkpoint_manager)

            evaluate(tokenizer, model, writer_checkpoint_manager)

            train(train_loader, model, criterion, optimizer, scaler, writer_checkpoint_manager)
            
            writer_checkpoint_manager.save_checkpoint(model, optimizer)
            
            writer_checkpoint_manager.increment_epoch()



if __name__ == '__main__':
    freeze_support() # Optional, only if you plan to create an executable
    
    # Do not change this line
    main()