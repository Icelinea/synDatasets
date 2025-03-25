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

        self.messages = []
        self.debugMode = False

    def model_init(self, setupPrompt=None):
        assert setupPrompt != None
        print('\n--- setupPrompt for {} ---'.format(self.role))
        print(setupPrompt)
        print('-----\n')

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype = torch.float16,
            device_map = "balanced",
            low_cpu_mem_usage=True
        )

        if self.debugMode:
            print('--- model init ---')
            print(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.messages.append(
            {"role": "system", "content": setupPrompt}
        )

    def generate(self, exampleNum, backgroundExample):
        assert type(exampleNum) == type(0)
        
        if self.debugMode:
            print(1, cuda_memory())

        genPrompt = "请生成 {} 条符合要求 json 格式的患者背景消息，可以参考下面提供的病患的信息丰富你生成的内容，但是生成的内容中不能与病患的信息相同：{}".format(exampleNum, backgroundExample)   # 案例
        self.messages.append(
            {"role": "user", "content": genPrompt}
        )

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if self.debugMode:
            print(2, cuda_memory())

        with torch.no_grad():  # 禁用梯度计算
            modelInputs = self.tokenizer([text], return_tensors="pt").to(device)

            if self.debugMode:
                print(3, cuda_memory())
    
            generatedIds = self.model.generate(
                modelInputs.input_ids,
                max_new_tokens=512
            )

            if self.debugMode:
                print(4, cuda_memory())
            
            generatedIds = [
                outputIds[len(input_ids):] for input_ids, outputIds in zip(modelInputs.input_ids, generatedIds)
            ]

            if self.debugMode:
                print(5, cuda_memory())
    
            response = self.tokenizer.batch_decode(generatedIds, skip_special_tokens=True)[0]

        # 释放 cuda 内存
        del modelInputs, generatedIds
        torch.cuda.empty_cache()
        
        if self.debugMode:
            print(6, cuda_memory())
        
        return response
    
    def chat(self, input, history):
        1


if __name__ == '__main__':
    1