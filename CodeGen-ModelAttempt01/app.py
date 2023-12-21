from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("alecsharpie/codegen_350m_html")

text = '<!DOCTYPE html>\n<html lang="en">\n\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Professional CyberSecurity Portfolio</title>\n<link rel="stylesheet" href="css/style.css">\n</head>\n\n<body>\n<div class="container">\n<div class="header">\n<h1>Professional CyberSecurity Portfolio</h1>\n</div>\n<div class="nav">\n'

input_ids = tokenizer(text, return_tensors="pt")

generated_ids = model.generate(**input_ids, max_length=2048)

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
