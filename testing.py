from transformers import pipeline
import torch

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

instruction = "Generate a high-level website portfolio template, using HTML and CSS."

prompt = MAGICODER_PROMPT.format(instruction=instruction)
generator = pipeline(
    model="ise-uiuc/Magicoder-S-DS-6.7B",
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)
print(result[0]["generated_text"])