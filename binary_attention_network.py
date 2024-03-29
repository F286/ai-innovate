import os
from multiprocessing import freeze_support
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
batch_size = 128
seq_length = 256  # The length to pad or truncate to
d_model = 128
feed_forward_expand_dim = d_model * 4
num_layers = 4
num_heads = 8
num_epochs = 10000
checkpoint_path = ""
vocab_size = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 1

    
class BinarySoftmax(nn.Module):
    def __init__(self, tau=1.0):
        super(BinarySoftmax, self).__init__()
        self.tau = tau  # Temperature parameter for Gumbel-Softmax

    @staticmethod
    def sample_gumbel(shape, device, eps=1e-20):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits):
        gumbel_noise = self.sample_gumbel(logits.size(), logits.device)
        y_soft = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
        return y_soft


class InternalAttention(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(InternalAttention, self).__init__()
        
        self.binary_softmax = BinarySoftmax()
        
        # self.normalize = nn.LayerNorm(d_model) 
        
        # self.fc_layer = nn.Sequential(
        #         nn.Linear(d_model, dim_feedforward),
        #         nn.ReLU(),
        #         nn.Linear(dim_feedforward, d_model)
        #     )

    def forward(self, x):
        
        x = self.binary_softmax(x)
        # # Normalize inputs so reconstruction loss is between both sides normalized
        # x = self.normalize(x)
        
        # # Calculate raw probabilities for mask
        # mask = self.fc_layer(x)
        
        # # Apply softmax to get probabilities
        # mask = F.softmax(mask, dim=-1)

        # # Apply probability mask
        # x = x * mask
        
        return x

    

class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super(MambaLayer, self).__init__()

        mamba_instance = Mamba(d_model, device=device)
        self.mamba_block = Block(dim=d_model, mixer_cls=lambda dim: mamba_instance)

    def forward(self, x):
        
        x, residual = self.mamba_block(x)

        return x, residual



class TransformerLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(TransformerLayer, self).__init__()

        self.internal_attention1 = InternalAttention(d_model, dim_feedforward)
        self.layer = MambaLayer(d_model)
        # self.internal_attention2 = InternalAttention(d_model, dim_feedforward)

    def forward(self, x):
        
        x = self.internal_attention1(x)
        
        output, residual = self.layer(x)
        
        # x = self.internal_attention2(x)
        
        return x + output

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Internal attention on embedding
        # self.internal_attention = InternalAttention(d_model, dim_feedforward)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, dim_feedforward)
            for _ in range(num_layers)])
        
        # Define a sequential layer for expansion, ReLU, and expansion to vocab_size
        self.fc_layer = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, vocab_size)
            )

    def forward(self, x):
        x = self.embedding(x)
        
        additional_loss = 0 

        # x = self.internal_atten7tion(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x)

        x = self.fc_layer(x)

        return x, additional_loss



def evaluate(tokenizer, model, epoch):        
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

        print(f"\n{generated_text}")

        print('')

def validate(model, val_loader, criterion, device):
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
    print(f"Average Validation Loss: {avg_loss:.4f}")

def train(train_loader, model, criterion, optimizer, scaler, epoch):
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

            print(f"\rEpoch {epoch + 1}: {elapsed_time_total:.0f} seconds ({tokens_per_second:.0f} tok/s), Loss: {avg_loss:.4f}, Progress: {epoch_progress:.2f}%",
                    end='')

    # Print a newline at the end to move to the next line in the console
    print('')


def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        # Load the state dictionary from the checkpoint file
        checkpoint_load = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_load['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint_load:
            optimizer.load_state_dict(checkpoint_load['optimizer_state_dict'])

        if 'epoch' in checkpoint_load:
            start_epoch = checkpoint_load['epoch']

        model.to(device)
    else:
        print(f"Checkpoint file {checkpoint_path} does not exist. Continuing without loading.")

def save_checkpoint(epoch, model, optimizer):
    # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        os.makedirs("transformer-grid", exist_ok=True)
        checkpoint_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # other stuff you want to save
        }
        torch.save(checkpoint_save, f"transformer-grid/tinystories_{epoch + 1}.pt")


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
    new_seq_length = seq_length  # 1 for <bos>, 1 for <eos>

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
    model = TransformerModel(num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, d_model=d_model, dim_feedforward=feed_forward_expand_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    print_parameter_count(model)

    load_checkpoint(model, optimizer)

    scaler = GradScaler()

    for epoch in range(num_epochs):
        
        print(f"\n-- Epoch {epoch} --\n")
        
        validate(model, val_loader, criterion, device)

        evaluate(tokenizer, model, epoch)

        train(train_loader, model, criterion, optimizer, scaler, epoch)
        
        save_checkpoint(epoch, model, optimizer)


    print("Training completed.")


if __name__ == '__main__':
    freeze_support() # Optional, only if you plan to create an executable
    main()
