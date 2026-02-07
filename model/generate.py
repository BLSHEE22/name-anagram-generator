import torch

def generate_anagram(model, tokenizer, name):
    input_text = f"anagram name: {name}"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=32,
            num_beams=5,
            temperature=0.9
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

