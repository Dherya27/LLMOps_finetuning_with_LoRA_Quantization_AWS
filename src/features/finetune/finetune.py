import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# import schemas
from src.api.schemas import TrainingRequest

class FineTune:
    def __init__(self, finetune_req:TrainingRequest) -> None:
        self.model_name = finetune_req.model_name
        self.dataset_name = finetune_req.dataset_name
        self.new_model = finetune_req.new_model
        self.config = finetune_req.config

    def load_bits_and_bytes_config(self):
        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
        return compute_dtype, bnb_config

    def load_lora_config(self):
        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            r=self.config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return peft_config

    def set_training_argutments(self):
        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim=self.config.optim,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=self.config.group_by_length,
            lr_scheduler_type=self.config.lr_scheduler_type,
            report_to="tensorboard"
        )
        return training_arguments
    
    def push_model_to_hugging_face_hub(self,model,tokenizer,username:str,new_model_name:str,auth_token:str):
        try:
            import locale
            locale.getpreferredencoding = lambda: "UTF-8"
            directory = username + "/" + new_model_name
            model.push_to_hub(directory,token=auth_token)
            tokenizer.push_to_hub(directory,token=auth_token)
        except Exception as e:
            print(f"An error occurred while pushing to the hugging face: {str(e)}")


    def train(self):
        # Load dataset
        dataset = load_dataset(self.dataset_name, split="train")

        compute_dtype, bnb_config = self.load_bits_and_bytes_config()

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.config.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        peft_config = self.load_lora_config()

        training_arguments = self.set_training_argutments()

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=self.config.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.config.packing,
        )

        # Train model
        trainer.train()

        # Save trained model
        trainer.model.save_pretrained(self.new_model)

        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.config.device_map,
        )
        model = PeftModel.from_pretrained(base_model, self.new_model)
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Push model to the hugging face hub
        self.push_model_to_hugging_face_hub(model,tokenizer,username="dherya",new_model_name=self.new_model,auth_token="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")


# # checking examples
# finetune_instance = FineTune(finetune_req=TrainingRequest)
# finetune_instance.train()

