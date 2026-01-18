import os
from unsloth import FastLanguageModel
from langchain_core.language_models.llms import LLM
from typing import Optional, List
import torch

class FoodHubMistral(LLM):
    model: any = None
    tokenizer: any = None

    def __init__(self, model_path="finetuned_mistral_llm"):
        super().__init__()
        # Load the model with 4-bit quantization for T4 GPU efficiency
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            load_in_4bit = True,
            device_map = "auto"
        )
        FastLanguageModel.for_inference(self.model)

    @property
    def _llm_type(self) -> str: return "mistral_finetuned"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Utilizing the Alpaca instruction template from the fine-tuning logic
        instruction_prompt = f"### Instruction:\nSupport the customer with their order query.\n\n### Input:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(instruction_prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=150, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decoding generated tokens only
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response.strip()
