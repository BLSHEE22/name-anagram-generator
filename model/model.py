from transformers import T5ForConditionalGeneration

def load_model(model_name="t5-small"):
    return T5ForConditionalGeneration.from_pretrained(model_name)
