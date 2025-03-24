from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from parameters import *


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

    def model_init(self, setupPrompt=None):
        assert setupPrompt != None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype = "auto",
            device_map = "auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.messages.append(
            {"role": "system", "content": setupPrompt}
        )

    def generate(self, exampleNum, backgroundExample):
        assert type(exampleNum) == type(0)

        genPrompt = "请生成 {} 条符合要求 json 格式的患者背景消息，可以参考下面提供的病患的信息丰富你生成的内容，但是生成的内容中不能与病患的信息相同：{}".format(exampleNum, backgroundExample)   # 案例
        self.messages.append(
            {"role": "user", "content": genPrompt}
        )

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def chat(self):
        1


if __name__ == '__main__':
    1