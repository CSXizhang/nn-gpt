import argparse
import sys
from peft import LoraConfig
from transformers import TrainingArguments
from ab.gpt.util.Tune import tune
from ab.gpt.util.Const import nngpt_dir

def main():
    parser = argparse.ArgumentParser(description='Run Backbone Tuning.')
    parser.add_argument('--llm_conf', type=str, default='sft_config.json', help='LLM config file name')
    parser.add_argument('--test_nn', type=int, default=2, help='Number of NNs to generate')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of LLM fine-tuning epochs')
    parser.add_argument('--nn_train_epochs', type=int, default=1, help='Number of training epochs for generated NNs')
    
    args = parser.parse_args()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=str(nngpt_dir / 'outputs'),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=args.num_train_epochs,
        logging_steps=5,
        bf16=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Call tune
    # We use None for prompt configs as SFTGenPrompt handles its own data fetching
    tune(
        test_nn=args.test_nn,
        nn_train_epochs=args.nn_train_epochs,
        nn_name_prefix='rl-bb-init',
        skip_epoch=0,   
        llm_path=None,
        llm_tune_conf='dummy.json',
        nn_gen_conf='dummy.json',
        conf_keys=['dummy_key'],
        llm_conf=args.llm_conf,
        training_args=training_args,
        peft_config=peft_config,
        # max_prompts=1000,
        use_backbone=True
    )

if __name__ == '__main__':
    main()
