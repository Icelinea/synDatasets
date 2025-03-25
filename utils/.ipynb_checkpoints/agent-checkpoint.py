from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.parameters import *


class Agent():
    def __init__(self, model_path, model_name, role="", background="", persona=""):
        self.model_path = model_path
        self.model_name = model_name

        self.model = None
        self.tokenizer = None
        
        self.role = role    # 索引名
        self.background = background    # 背景
        self.persona = persona  # 语言风格
        self.setupPrompt = None

        self.debugMode = False

    def model_init(self, setupPrompt=None):
        assert setupPrompt != None
        print('\n--- setupPrompt for {} ---'.format(self.role))
        print(setupPrompt)
        print('-----\n')

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype = "auto",
            device_map = "auto",
            # low_cpu_mem_usage=True
        )

        if self.debugMode:
            print('--- model init ---')
            print(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.setupPrompt = {"role": "system", "content": setupPrompt}

    def generate(self, exampleNum, backgroundExample):
        assert type(exampleNum) == type(0)
        
        if self.debugMode:
            print(1, cuda_memory())

        genPrompt = "请生成 {} 条符合要求 json 格式的患者背景消息，可以参考下面提供的病患的信息丰富你生成的内容，生成的病患信息尽量不要相似或者重复，在丰富生成数据的同时保持数据分布符合现实世界常理：{}".format(exampleNum, backgroundExample)   # 案例
        messages = []
        messages.append(self.setupPrompt)
        messages.append(
            {"role": "user", "content": genPrompt}
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        with torch.no_grad():  # 禁用梯度计算
            modelInputs = self.tokenizer([text], return_tensors="pt").to(device)
            generatedIds = self.model.generate(
                modelInputs.input_ids,
                max_new_tokens=512
            )
            generatedIds = [
                outputIds[len(input_ids):] for input_ids, outputIds in zip(modelInputs.input_ids, generatedIds)
            ]
            response = self.tokenizer.batch_decode(generatedIds, skip_special_tokens=True)[0]

        # 释放 cuda 内存
        del modelInputs, generatedIds
        torch.cuda.empty_cache()
        
        if self.debugMode:
            print(2, cuda_memory())
        
        return response
    
    def chat(self, input, history):
        1


if __name__ == '__main__':
    1