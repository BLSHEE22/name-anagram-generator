from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./anagram_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

model.eval()

def generate_anagram(name, max_new_tokens=32):
    input_text = name
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive loop
while True:
    name = input("\nEnter a name (or 'q' to quit): ")
    if name.lower() == "q":
        break
    print("→", generate_anagram(name))
