from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchviz import make_dot
import hiddenlayer as hl
# pip install flash-attn

model_dir = "/root/autodl-tmp/SoulChat2.0-Qwen2-7B"

model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

inputs = tokenizer("我最近失恋了，需要情感上的安慰。", return_tensors="pt")
outputs = model(**inputs)

# 生成可视化图
dot = make_dot(outputs.logits, params=dict(model.named_parameters()))
dot.format = 'png'
dot.directory = './'
dot.render('model_structure')

traced = torch.jit.trace(model, (inputs['input_ids'],))
graph = hl.build_graph(traced, inputs={'input_ids': inputs['input_ids']})
graph.save('./hl_model_structure')
