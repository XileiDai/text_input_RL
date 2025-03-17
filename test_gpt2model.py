from transformers import GPT2Tokenizer, GPT2Model
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
a = 1