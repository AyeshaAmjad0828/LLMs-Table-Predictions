import os
from typing import Any, Dict, Tuple
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel
)

import json
import textwrap

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write an astera expression in response to complete the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

class InstructionTextGenerationPipeline:
    def __init__(
        self,
        model_name,
        lora_model_name=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_new_tokens=1024, temperature=0.9, top_p=0.95, top_k=40
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
        if lora_model_name:
            print("Loading peft model")
            self.model = PeftModel.from_pretrained(
            self.model,
            lora_model_name,
            is_trainable=False,
            )
        
        print(type(self.model))
            
        tokenizer = AutoTokenizer.from_pretrained(
            "tiiuae/falcon-7b",
            trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)
        
        self.generation_config = GenerationConfig(
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )

    def format_instruction(self, instruction):
        return PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
    
    def get_model(self):
        return self.model

    def __call__(
        self, instruction: str, **generate_kwargs: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        s = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        input_ids = self.tokenizer(s, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(inputs=input_ids, generation_config=self.generation_config)
        # Slice the output_ids tensor to get only new tokens
        new_tokens = output_ids.sequences[0, len(input_ids[0]) :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text

def get_prompt(instruction):
    prompt_template = f"Below is an instruction that describes a task. Write an astera expression in response to complete the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt_template

print(get_prompt('What is the meaning of life?'))

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        return wrapped_text +'\n\n'

def get_model(model_path, lora_path=None):
    # Initialize the model and tokenizer
    return InstructionTextGenerationPipeline(
        model_path,
        lora_model_name=lora_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

def predict(prompt, generate):    
    stop_token_ids = generate.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
    
    generated_text = generate(prompt)
    return parse_text(generated_text)

if __name__ == "__main__":
    generate = get_model("tiiuae/falcon-7b", "./models/falcon-7b-ae")
    # generate = get_model("./models/falcon-7b-ae/merged")
    print(predict(get_prompt("Give me an example of astera expression syntax"), generate))