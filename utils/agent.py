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
        self.background = background    # 患者背景
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
    
    def chat(self, history):
        # general
        messages = []
        messages.append(self.setupPrompt)

        # role-specific
        genPrompt = "你与对方的历史对话记录为：{}。其中每个对话前面都有表示其身份的名称医生或者患者。".format(history)
        if self.role == "patient":
            genPrompt += "如果没有历史对话记录，请你按照自己的患者求助背景和指定的性格向医生提出合理的问题。如果有历史对话记录，请你按照你自己的性格回答医生的问题或者继续自己的行为。请不要说谎，如实的说明你当前的想法，自身的心理健康问题是否得到改善或者恶化。注意字数需要控制在 10 到 100 个中文字数之间。注意使用一定要使用口语化的表达。"
        elif self.role == "doctor":
            genPrompt += "按照患者所提供的背景，当前的患者核心病症主题应该是{}。你作为心理健康领域医生需要坚持向患者探讨这个主题，尽量不要被带偏，直到患者的心理问题得到改善。你可以向患者提出问题来进一步了解患者的内心想法，也可以回答患者提出的问题。如果你认为当前主题已经探讨完全或者患者的心理问题得到了很大的改善，只需要回答\"EOF\"，不要有其他单词。注意字数需要控制在 10 到 100 个中文字数之间。注意使用一定要使用口语化的表达。".format(self.background["主题"])
        assert genPrompt != ""

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
        
        return response


if __name__ == '__main__':
    1