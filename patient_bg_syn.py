import random
import json
from utils.parameters import *
from utils.agent import Agent

# 设置环境变量来控制 PyTorch 显存管理
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.9'

# Parameters
dialogueTopics = [
    "[情绪表现]指个体在某段时间内的整体情感状态，如愉快、焦虑等，影响思维和行为，一般持续时间较短",
    "[兴趣爱好]指个体对活动、事物的关注和吸引力。兴趣缺失可能与抑郁等心理问题相关",
    "[心理状态]指个体的思维、情绪、行为和认知功能的总体表现，相对于情绪来说持续时间更久并且影响更大",
    "[睡眠情况]指休息性状态，影响身心健康。失眠或过度睡眠是常见的心理健康问题",
    "[食欲状况]指个体对食物的渴望或需求，食欲变化可反映抑郁、焦虑等情绪问题",
    "[躯体症状]指由心理因素引起的身体症状，如头痛、胸痛等",
    "[社交功能]指个体在社会互动中的适应能力，包括工作、家庭和朋友关系等方面",
    "[自杀倾向]指个体产生自杀念头、计划或行为的倾向，通常与严重的心理疾病相关",
    "[其他情况]个体不存在以上所有普遍心理健康问题的情况，可能是正常心理健康的个体或者是有着疑似症状被误诊但是实际上正常的个体"
]

# 患者背景知识生成
# 额外知识 + 输出格式
# setupPrompt = "你是一名心理健康领域的专家，以下是你需要理解的在与病患对话时的核心病症的对话主题种类，其中 [] 内的是核心病症主题的专有名词，后面接着的是该专有名词的解释，你所进行的所有的心理健康对话的核心病症的主题都可以被归类到以下的情况当中：{}。你生成的消息一定要按照以下的 json 格式输出，其中 () 内表示你需要生成的内容，生成时请去除 () 并且不要生成多余的回复内容或者多余的标点符号，外层是列表符号 []，内层是字典符号 {{}}，具体 json 格式如下：[{{\"年龄\": (), \"性别\": (), \"主题\": (), \"背景\": ()}}]。注意其中年龄为人类的正常年龄，只能是数字，不能小于 0 岁；性别只有'男'和'女'两种；主题是心理健康对话的核心病症的主题的其中一种类型，需要由你随机选择一种病症主题的专有名词，最终选择的名词必须完全来自前面所提供的核心病症的专有名词，只有 \"情绪表现、兴趣爱好、心理状态、睡眠情况、食欲状况、躯体症状、社交功能、自杀倾向、其他情况\" 可以被选择，并且保证每种主题都有同等的概率被选择到；背景是与你对话的病患所提供的病情背景信息，需要由你根据所选择的核心病症的主题进行生成，字数不能少于 20 字且不能多于 100 字，并且请不要生成涉及真实人员隐私的信息，例如人名、地名、建筑名、身份证等隐私信息。".format("；".join(dialogueTopics))
setupPrompt = "你是一名心理健康领域的专家，以下是你需要理解的在与病患对话时的核心病症的对话主题种类，其中 [] 内的是核心病症主题的专有名词，后面接着的是该专有名词的解释，你所进行的所有的心理健康对话的核心病症的主题都可以被归类到以下的情况当中：{}。你生成的消息一定要按照以下的 json 格式输出，其中 () 内表示你需要生成的内容，生成时请去除 () 并且不要生成多余的回复内容或者多余的标点符号，外层是列表符号 []，内层是字典符号 {{}}，具体 json 格式如下：[{{\"年龄\": (), \"性别\": (), \"主题\": (), \"背景\": ()}}]。注意其中年龄为人类的正常年龄，只能是数字，不能小于 0 岁；性别只有'男'和'女'两种；主题你只能选择\"其他情况\"，其他情况是指不存在以上心理健康问题的，正常心理的人类个体的情况；背景是与你对话的病患所提供的病情背景信息，需要由你根据所选择的主题的专有名词对应的核心病症解释进行生成，选择了\"其他情况\"那么请生成正常人类的一段生活背景，其中的内容不应该存在任何心理健康问题，字数不能少于 20 字且不能多于 100 字，并且请不要生成涉及真实人员隐私的信息，例如人名、地名、建筑名、身份证等隐私信息。".format("；".join(dialogueTopics))
# 情绪表现、兴趣爱好、心理状态、睡眠情况、食欲状况、躯体症状、社交功能、自杀倾向、其他情况

def background_synthesis(role, epoches, genNum, exampleNum, dataPath, outputPath):
    assert role == "patient"

    # 读取案例数据
    data = None
    with open(dataPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data != None

    # print(data[0]["案例简述"])  # test
    
    # 如果文件不存在，先写入一个空列表
    if not os.path.exists(outputPath):
        with open(outputPath, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    agent = Agent(globalModelPath, globalModelName, role)
    agent.model_init(setupPrompt)

    # responses = []
    for epoch in range(epoches):
        selects = random.sample(data, exampleNum)
        selectData = [dic["案例简述"] for dic in selects]

        genPrompt = "请生成 {} 条符合要求 json 格式的患者背景消息，可以参考下面提供的病患的信息丰富你生成的内容，生成的病患信息尽量不要相似或者重复，在丰富生成数据的同时保持数据分布符合现实世界常理：{}".format(genNum, selectData)   # 案例  
        response = agent.generate(genPrompt)

        print(epoch, response)
        # responses.append(response)
        
        # 读取原有数据
        with open(outputPath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        # 添加新数据
        dataset.append(response)
        # 写回文件
        with open(outputPath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    # # write 格式的问题
    # with open(outputPath, 'w', encoding='utf-8') as o:
    #     json.dump(responses, o, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    background_synthesis("patient", 100, 1, 10, "./data/CPsyCounR/CPsyCounR.json", "./data/PatientBackground/bg-others.json")