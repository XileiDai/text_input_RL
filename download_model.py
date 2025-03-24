from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model

model_name = "gpt2"
local_dir = "local_models/gpt2"

# # # Download and save
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2Model.from_pretrained(model_name)

# tokenizer.save_pretrained(local_dir)
# model.save_pretrained(local_dir)

tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
model = GPT2Model.from_pretrained(local_dir)