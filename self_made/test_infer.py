import argparse
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import os
import cv2
import numpy as np

import torch
import torch.distributed as dist

model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'
model_dir = '../../Qwen-VL-Chat'

# Tokenizer and VL-Model
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()

prompt_template = ['<img>{}</img>In this picture, there are two blue masks each covering an object. \
    Can you determine whether the objects they have covered belong to the same object? \
    Please response with “yes” if they belong to the same one, or “no” if they do not.',
    '<img>{}</img>In this picture, there are two blue masks each covering an object. \
    Can you determine whether the objects they have covered are of the same class? \
    Please response with "yes" if they belong to the same category, or "no" if they do not.',
    '<img>{}</img>Tell me what is the main object in this picture?'
    ]
query=prompt_template[1].format('./1.jpg')

response, history = model.chat(tokenizer=tokenizer, query=query, history=None)

print (response)