import os
from multiprocessing import freeze_support
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import Adafactor

from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, trainers, normalizers, pre_tokenizers
import pickle
from tokenizers.models import BPE

from torch.cuda.amp import autocast, GradScaler

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
from compact_expand_module import CompactExpandModule
from sentence_attention_module import SentenceAttentionModule
    
batch_size = 128
seq_length = 256
d_model = 256
feed_forward_expand_dim = d_model * 4
num_layers = 32
num_heads = 2
num_epochs = 10000
checkpoint_directory = "mamba_standard/"
checkpoint_filename = "none"
log_directory = checkpoint_directory
checkpoint_save_every_epochs = 1
vocab_size = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 1



class NetworkLayer(nn.Module):
    def __init__(self, d_model):
        super(NetworkLayer, self).__init__()

        mamba_instance = Mamba(d_model, device=device)
        self.mamba_block = Block(dim=d_model, mixer_cls=lambda dim: mamba_instance)

    def forward(self, x):
        
        x, residual = self.mamba_block(x)

        return x, residual


class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super(TransformerLayer, self).__init__()

        self.layer = NetworkLayer(d_model)

    def forward(self, x):
        # Apply Mamba block
        output, residual = self.layer(x)
        
        return x + output
    

class SentenceLocalLayer(nn.Module):
    def __init__(self, sentence_breaking_token_ids, sequence_length, embedding_dimension, num_heads):
        super(SentenceLocalLayer, self).__init__()

        self.layer = SentenceAttentionModule(sentence_breaking_token_ids, sequence_length, embedding_dimension, num_heads)

    def forward(self, x, token_ids):
        output = self.layer(x, token_ids)
        
        return x + output

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, keep_token_ids):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Attention layers
        # self.layers0 = nn.ModuleList([
        #     TransformerLayer(d_model)
        #     for _ in range(2)])
        
        self.layers1 = nn.ModuleList([
            TransformerLayer(d_model)
            for _ in range(num_layers)])
        
        # self.layers2 = nn.ModuleList([
        #     TransformerLayer(d_model)
        #     for _ in range(2)])
        
        # Compact expand
        compacted_max_sequence_length = 32
        # self.compact_expand = CompactExpandModule(keep_token_ids, sequence_length=seq_length, embedding_dimension=d_model, compacted_max_sequence_length=compacted_max_sequence_length)
        
        # Define a sequential layer for expansion, ReLU, and expansion to vocab_size
        self.fc_layer = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, vocab_size)
            )

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        
        additional_loss = 0 
        
        # for idx, layer in enumerate(self.layers0):
        #     x = layer(x)
            
        # self.compact_expand(x, token_ids, is_compacting=True)

        for idx, layer in enumerate(self.layers1):
            x = layer(x)
            
        # self.compact_expand(x, is_compacting=False)
            
        # for idx, layer in enumerate(self.layers2):
        #     x = layer(x)

        x = self.fc_layer(x)

        return x, additional_loss



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
            predictions, additional_loss = model(input_text)
            next_token_idx = predictions[:, -1, :].argmax(dim=-1)
            input_text = torch.cat([input_text, next_token_idx.unsqueeze(0)], dim=1)

        # Decode the generated tokens using the tokenizer
        generated_text = tokenizer.decode(input_text[0].cpu().tolist())

        writer_checkpoint_manager.add_text("Evaluate/text", generated_text)
        # print(f"\n{generated_text}")

        # print('')

def validate(model, val_loader, criterion, device, writer_checkpoint_manager:TensorBoardCheckpointWriter):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_tokens = 0  # Keep track of the total number of tokens processed

    with torch.no_grad():  # No gradients needed for validation
        for text, target in val_loader:
            text = text.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            predictions, _ = model(text)
            predictions = predictions.view(-1, predictions.size(2))
            target = target.view(-1)

            loss = criterion(predictions, target)
            total_loss += loss.item() * target.size(0)  # Multiply loss by the number of tokens in target
            total_tokens += target.size(0)  # Accumulate total tokens
            
            if total_tokens > 1000000:
                break

    # Normalize the total loss by the total number of tokens processed
    avg_loss = total_loss / total_tokens
    # print(f"Average Validation Loss: {avg_loss:.4f}")
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
            tokens_processed += batch_size * seq_length

            # Run the model
            prediction, additional_loss = model(text)
            
            # Prediction
            loss = additional_loss
            
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

            # print(f"\rEpoch {writer_checkpoint_manager.epoch + 1}: {elapsed_time_total:.0f} seconds ({tokens_per_second:.0f} tok/s), Loss: {avg_loss:.4f}, Progress: {epoch_progress:.2f}%",
                    # end='')
            
            writer_checkpoint_manager.add_scalar("Loss/train", avg_loss)
            writer_checkpoint_manager.add_scalar("Tokens per second", tokens_per_second)
            
            writer_checkpoint_manager.increment_global_step()

    # Print a newline at the end to move to the next line in the console
    # print('')


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
    new_seq_length = seq_length 

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
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=16, pin_memory=True, prefetch_factor=8, drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=16, pin_memory=True, prefetch_factor=8, drop_last=True)

    # Model definition
    assert(vocab_size == tokenizer.get_vocab_size())
    
    # Common end-of-sentence specifiers
    eos_tokens = ['.', '!', '?']

    # Retrieve their token IDs from the tokenizer
    keep_token_ids = torch.tensor([tokenizer.token_to_id(tok) for tok in eos_tokens], device=device)

    model = TransformerModel(num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, d_model=d_model, dim_feedforward=feed_forward_expand_dim, keep_token_ids=keep_token_ids).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    print_parameter_count(model)

    with TensorBoardCheckpointWriter(log_dir=log_directory, checkpoint_dir=checkpoint_directory) as writer_checkpoint_manager:
        # Your training/validation loop and any checkpoint saving/loading logic
        # Use writer_checkpoint_manager for logging and checkpointing

        writer_checkpoint_manager.load_checkpoint(model, optimizer, checkpoint_filename)

        scaler = GradScaler()

        while writer_checkpoint_manager.epoch < num_epochs:
            
            print(f"\n-- Epoch {writer_checkpoint_manager.epoch} --\n")
            
            validate(model, val_loader, criterion, device, writer_checkpoint_manager)

            evaluate(tokenizer, model, writer_checkpoint_manager)

            train(train_loader, model, criterion, optimizer, scaler, writer_checkpoint_manager)
            
            writer_checkpoint_manager.save_checkpoint(model, optimizer)
            
            writer_checkpoint_manager.increment_epoch()



if __name__ == '__main__':
    freeze_support() # Optional, only if you plan to create an executable
    main()
