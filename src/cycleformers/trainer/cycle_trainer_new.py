"""Completely fresh attempt, based on the huggingface trainer but not using it for now"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from datasets import load_from_disk, Dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader


BATCH_SIZE = 4


def check_gradients(model):
    """Helper function to check if gradients exist and their magnitudes"""
    has_grad = False
    grad_mag = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_mag += param.grad.norm().item()
    return has_grad, grad_mag


def check_and_print_grads(model_a, model_b):
    has_grad_a, grad_mag_a = check_gradients(model_a)
    has_grad_b, grad_mag_b = check_gradients(model_b)
    print(f"Model Train has gradients: {has_grad_a}, magnitude: {grad_mag_a:.6f}")
    print(f"Model Gen has gradients: {has_grad_b}, magnitude: {grad_mag_b:.6f}")



class CycleTrainer(Trainer):
    def __init__(self, model_a, model_b, args, tokenizer, train_dataset_a, train_dataset_b, eval_dataset_a, eval_dataset_b, data_collator):
        # Gradient accumulation of 2 doesn't work - Only one model is trained
        self.accelerator = Accelerator(gradient_accumulation_steps=1)
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.eval_dataset_a = eval_dataset_a
        self.eval_dataset_b = eval_dataset_b
        
        optimizer_a = AdamW(model_a.parameters(), lr=1e-3)
        optimizer_b = AdamW(model_b.parameters(), lr=1e-3)

        train_dataloader_a = self.get_train_dataloader(train_dataset_a)
        train_dataloader_b = self.get_train_dataloader(train_dataset_b)

        # Add everything to accelerator
        self.model_a, self.model_b, self.optimizer_a, self.optimizer_b, self.train_dataloader_a, self.train_dataloader_b = \
            self.accelerator.prepare(model_a, model_b, optimizer_a, optimizer_b, train_dataloader_a, train_dataloader_b)
        
        self.model_a = self.model_a.to(self.accelerator.device)
        self.model_b = self.model_b.to(self.accelerator.device)


    def get_train_dataloader(self, dataset):
        dataloader_params = {
            "batch_size": BATCH_SIZE,
            "collate_fn": self.data_collator
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
    

    def get_eval_dataloader(self, dataset):
        dataloader_params = {
            "batch_size": BATCH_SIZE,
            "collate_fn": self.data_collator
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))


    def train(self):
        for i, (batch_a, batch_b) in enumerate(zip(self.train_dataloader_a, self.train_dataloader_b)):
            print("Cycle B")
            self._cycle_step(self.model_a, self.model_b, self.optimizer_a, batch_a)
            print("Cycle A")
            self._cycle_step(self.model_b, self.model_a, self.optimizer_b, batch_b)
            

    def _cycle_step(self, model_train, model_gen, optimizer, batch):
        model_gen.eval()
        model_train.train()

        with torch.no_grad():   
            synth_batch = model_gen.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)

        # Clean up output for model b - Can be made much more efficient
        synth_batch = self.tokenizer.batch_decode(synth_batch, skip_special_tokens=True)
        synth_batch = {k: v.to(self.accelerator.device) for k, v in self.tokenizer(synth_batch, return_tensors="pt", padding=True).items()}        

        with self.accelerator.accumulate(model_train, model_gen):
            loss = model_train(**synth_batch, labels=batch["input_ids"]).loss # Need to add labels column to batch_b
            print(f"Loss: {loss}")
            self.accelerator.backward(loss)            
            optimizer.step()
            check_and_print_grads(model_train, model_gen)
            optimizer.zero_grad()
            check_and_print_grads(model_train, model_gen)
    

    def evaluate(self, ignore_keys=None):
        # First evaluate model a
        eval_dataloader_a = self.get_eval_dataloader(self.eval_dataset_a)
        eval_dataloader_b = self.get_eval_dataloader(self.eval_dataset_b)

        self.model_a.eval()
        
        for batch_a in eval_dataloader_a:
            losses, logits, labels = self.prediction_step(self.model_a, batch_a, ignore_keys=ignore_keys)





            

# ==============================
# This will all be removed after testing

if __name__ == "__main__":
    dataset_en, dataset_de = load_from_disk("data/en"), load_from_disk("data/de")

    model_a = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model_b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", padding=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    args = {}

    trainer = Trainer(
        model_a,
        model_b,
        args,
        tokenizer,
        dataset_en['train'],
        dataset_de['train'],
        dataset_en['test'],
        dataset_de['test'],
        data_collator
    )
    trainer.train()
