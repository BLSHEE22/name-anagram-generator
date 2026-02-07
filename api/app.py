from fastapi import FastAPI
from model.model import load_model
from model.tokenizer import load_tokenizer
from model.generate import generate_anagram

app = FastAPI()

model = load_model("./anagram_model")
tokenizer = load_tokenizer("./anagram_model")

@app.get("/anagram")
def anagram(name: str):
    result = generate_anagram(model, tokenizer, name)
    return {"name": name, "anagram": result}
