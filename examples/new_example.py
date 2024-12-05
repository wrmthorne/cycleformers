import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig

from cycleformers.trainer.cycle_trainer import CycleTrainer
from cycleformers.trainer.cycle_training_arguments import CycleTrainingArguments


def generate_model_samples(model, tokenizer, dataset, data_collator, num_samples=10):
    model.eval()
    samples = []
    for batch in tqdm(dataset, total=num_samples, desc="Generating samples"):
        # Dataset already contains tokenized inputs
        batch = data_collator([batch])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)
            samples.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    for sample in samples:
        print(sample)
        print()


# ==============================
# This will all be removed after testing

if __name__ == "__main__":
    dataset_en, dataset_de = load_from_disk("data/en"), load_from_disk("data/de")

    MODEL_NAME = "google/flan-t5-small"
    model_A = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model_B = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # MODEL_NAME = "meta-llama/Llama-3.2-1B"
    # model_A = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    # model_B = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

    rank = 32
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM" if model_A.config.is_encoder_decoder else "CAUSAL_LM",
        r=rank,
        lora_alpha=rank*2,
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        use_rslora=True,
        bias="none",
    )
    model_A = get_peft_model(model_A, peft_config, adapter_name="A")
    model_A.add_adapter("B", peft_config)

    print('='*60)
    print(model_A.config.is_encoder_decoder)
    print(model_B.config.is_encoder_decoder)
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True)

    args = CycleTrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_steps=1,
        logging_strategy="steps",
        learning_rate=3e-4,
        save_steps=1,
        evaluation_strategy="no",
    )

    trainer = CycleTrainer(
        args,
        models=model_A,
        tokenizers=tokenizer,
        train_dataset_A=dataset_en['train'].select(range(2)),
        train_dataset_B=dataset_de['train'].select(range(2)),
        # eval_dataset_A=dataset_en['test'],
        # eval_dataset_B=dataset_de['test']
    )
    trainer.train()

    print("English to German")
    generate_model_samples(trainer.model_B, trainer.tokenizer_B, trainer.eval_dataset_A, trainer.data_collator_B)

    print("German to English")
    generate_model_samples(trainer.model_A, trainer.tokenizer_A, trainer.eval_dataset_B, trainer.data_collator_A)