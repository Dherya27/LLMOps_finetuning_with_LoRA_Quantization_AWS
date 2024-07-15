from pydantic import BaseModel
from typing import Optional, List, Any
from enum import Enum

class TrainingConfig(BaseModel):

    ################################
    # QLoRA parameters
    ################################
   
    # LoRA attention dimension
    lora_r: Optional[int] = 64
    # Alpha parameter for LoRA scaling
    lora_alpha: Optional[int] = 16
    # Dropout probability for LoRA layers
    lora_dropout: Optional[float] = 0.1

    ################################
    # bitsandbytes parameters
    ################################

    # Activate 4-bit precision base model loading
    use_4bit: Optional[bool] = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype: Optional[str] = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type: Optional[str] = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant: Optional[bool] = False

    ################################
    # TrainingArguments parameters
    ################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir: Optional[str] = "./results"

    # Number of training epochs
    num_train_epochs: Optional[int] = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16: Optional[bool] = False
    bf16: Optional[bool] = False

    # Batch size per GPU for training
    per_device_train_batch_size: Optional[int] = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size: Optional[int] = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps: Optional[int] = 1

    # Enable gradient checkpointing
    gradient_checkpointing: Optional[bool] = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm: Optional[float] = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate: Optional[float] = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay: Optional[float] = 0.001

    # Optimizer to use
    optim: Optional[str] = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type: Optional[str] = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps: Optional[int] = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio: Optional[float] = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length: Optional[bool] = True

    # Save checkpoint every X updates steps
    save_steps: Optional[int] = 0

    # Log every X updates steps
    logging_steps: Optional[int] = 25

    ################################
    # SFT parameters
    ################################

    # Maximum sequence length to use
    max_seq_length: Optional[int] = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing: Optional[bool] = False

    # Load the entire model on the GPU 0
    device_map: Optional[dict[str, int]] = {"": 0}

class TrainingRequest(BaseModel):
    model_name: str
    dataset_name: str
    new_model: str
    config: Optional[TrainingConfig] = None


# tr_request = TrainingRequest(
#     model_name="my_model",
#     dataset_name="my_dataset",
#     new_model="my_new_model",
#     config=TrainingConfig()
# )

# lora_value = tr_request.config.lora_r
# print(lora_value)


