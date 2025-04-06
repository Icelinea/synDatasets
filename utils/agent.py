from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.parameters import *


class Agent():
    def __init__(self, model_path, model_name, role="", persona=""):
        self.model_path = model_path
        self.model_name = model_name

        self.model = None
        self.tokenizer = None
        
        self.role = role    # 索引名
        self.persona = persona  # 语言风格
        self.setupPrompt = None

        self.debugMode = False

        

    def model_init(self, setupPrompt=None, printPrompt=True):
        assert setupPrompt != None
        if printPrompt:
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

    def prompt_init(self, setupPrompt=None, printPrompt=True):
        assert setupPrompt != None
        if printPrompt:
            print('\n--- setupPrompt for {} ---'.format(self.role))
            print(setupPrompt)
            print('-----\n')
            
        self.setupPrompt = {"role": "system", "content": setupPrompt}
    
    def generate(self, genPrompt):
        if self.debugMode:
            print(1, cuda_memory())

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
    
    def chat(self, background, history, temperature=0.5, top_k=50, top_p=0.95, num_return_sequences=1, seed=42):
        # general
        messages = []
        messages.append(self.setupPrompt)

        # role-specific
        genPrompt = ""
        if self.role == "patient":
            genPrompt = "请你按照你自己的性格回答医生的问题或者提出自己的问题。如实的说明你当前的想法，说明自身的心理健康问题是否得到改善或者恶化。如果你认为当前主题已经探讨完全或者自己的心理问题得到了改善，只需要回答\"EOF\"终止对话。注意字数需要控制在 10 到 100 个中文字数之间。注意使用一定要使用口语化的表达，不需要说“你好”等问候语。你与对方的历史对话记录为：{}。其中每个对话前面都有表示其身份的名称即医生或者患者，其中“患者”表示你说过的话，“医生”表示对方回复你的话，接下来你作为患者继续对话，请记住你的患者身份，不要说谎或者重复自己或者医生说过的话，并且请一定在每次回复的时候加上“患者：”。".format(history)
        elif self.role == "doctor":
            genPrompt = "按照患者所提供的背景，当前的患者核心病症应该是{}。你作为心理健康领域医生需要坚持向患者探讨这个主题，尽量不要被带偏，直到你认为患者的心理问题得到改善。你可以向患者提出问题来进一步了解患者的内心想法，也可以回答患者提出的问题。如果你认为当前主题已经探讨完全或者患者的心理问题得到了改善，只需要回答\"EOF\"终止对话。注意字数需要控制在 10 到 100 个中文字数之间。注意使用一定要使用口语化的表达，不需要说“你好”等问候语。你与对方的历史对话记录为：{}。其中每个对话前面都有表示其身份的名称即医生或者患者，其中“医生”表示你说过的话，“患者”表示对方回复你的话，接下来你作为医生继续对话，请记住你的医生身份，不要说谎或者重复自己或者患者说过的话，并且请一定在每次回复的时候加上“医生：”。".format(background, history)
        assert genPrompt != ""

        # 随机
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
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
                max_new_tokens=512,
                temperature=temperature,  # 控制生成的多样性，值越高生成越随机
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
            )
            generatedIds = [
                outputIds[len(input_ids):] for input_ids, outputIds in zip(modelInputs.input_ids, generatedIds)
            ]
            response = self.tokenizer.batch_decode(generatedIds, skip_special_tokens=True)[0]

        # 释放 cuda 内存
        del modelInputs, generatedIds
        torch.cuda.empty_cache()
        
        return response


if __name__ == '__main__':
    1