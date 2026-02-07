from transformers import T5Tokenizer

def load_tokenizer(model_name="t5-small"):
    return T5Tokenizer.from_pretrained(model_name)
