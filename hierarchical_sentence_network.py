# hierarchical_sentence_network.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import Adafactor
from datasets import load_dataset
from tokenizers import Tokenizer, trainers, normalizers, pre_tokenizers
from tokenizers.models import BPE
import pickle
import time
from hierarchical_sentence_model import Config, HierarchicalSentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data processing
class DataProcessor:
    @staticmethod
    def load_or_create_tokenizer(train_dataset, config):
        tokenizer_filename = "hierarchical_tokenizer.pkl"
        try:
            with open(tokenizer_filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            tokenizer = Tokenizer(BPE())
            tokenizer.normalizer = normalizers.Lowercase()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            train_text_data = [item['text'] for item in train_dataset]
            with open("train_text.txt", "w", encoding="utf-8") as f:
                for text in train_text_data:
                    f.write(text + "\n")
            trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=["<unk>", "<pad>", "<eos>", "<bos>"])
            tokenizer.train(files=["train_text.txt"], trainer=trainer)
            with open(tokenizer_filename, "wb") as file:
                pickle.dump(tokenizer, file)
            return tokenizer

    @staticmethod
    def create_tokenizer_function(tokenizer, config):
        def tokenize_data(example):
            text = example['text']
            sentences = text.split('.')
            tokenized_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    tokens = tokenizer.encode(sentence.strip()).ids
                    tokenized_sentences.extend(tokens + [config.EOS_IDX])
            return {'input_ids': [config.BOS_IDX] + tokenized_sentences}
        return tokenize_data

    @staticmethod
    def collate_batch(batch, config):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_IDX)
        
        if input_ids.size(1) > config.seq_length:
            input_ids = input_ids[:, :config.seq_length]
        
        targets = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]
        
        if input_ids.size(1) < config.seq_length - 1:
            padding = torch.full((input_ids.size(0), config.seq_length - 1 - input_ids.size(1)), config.PAD_IDX, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
            targets = torch.cat([targets, padding], dim=1)
        
        return input_ids, targets

# Training and evaluation
class Trainer:
    def __init__(self, model, criterion, optimizer, scaler, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, (input_ids, targets) in enumerate(train_loader):
            with autocast():
                input_ids = input_ids.to(self.config.device)
                targets = targets.to(self.config.device)
                output = self.model(input_ids)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
                start_time = time.time()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(self.config.device)
                targets = targets.to(self.config.device)
                output = self.model(input_ids)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"hierarchical_model_checkpoint_epoch_{epoch}.pt")

    def load_checkpoint(self):
        if os.path.exists(self.config.checkpoint_path):
            checkpoint = torch.load(self.config.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {start_epoch}")
            return start_epoch
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

# Main execution
def main():
    config = Config()

    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # Tokenizer and data processing
    tokenizer = DataProcessor.load_or_create_tokenizer(train_dataset, config)
    tokenize_func = DataProcessor.create_tokenizer_function(tokenizer, config)
    train_dataset = train_dataset.map(tokenize_func, cache_file_name="tokenized_train")
    validation_dataset = validation_dataset.map(tokenize_func, cache_file_name="tokenized_validation")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              collate_fn=lambda b: DataProcessor.collate_batch(b, config), 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, 
                            collate_fn=lambda b: DataProcessor.collate_batch(b, config), 
                            num_workers=4, pin_memory=True)

    # Initialize model, criterion, optimizer, and scaler
    model = HierarchicalSentenceTransformer(config).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    scaler = GradScaler()

    # Initialize trainer
    trainer = Trainer(model, criterion, optimizer, scaler, config)

    # Load checkpoint if exists
    start_epoch = trainer.load_checkpoint()

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)

    print("Training completed.")

if __name__ == '__main__':
    main()