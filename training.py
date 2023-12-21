# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ise-uiuc/Magicoder-S-DS-6.7B")
model = AutoModelForCausalLM.from_pretrained("ise-uiuc/Magicoder-S-DS-6.7B")