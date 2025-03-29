from utils.parameters import *
from utils.agent import Agent
import random
import json
import pandas as pd


# path
patientBackgroundPath = "data/PatientBackground/processed/scored_anno_pad_dupu_bg1.csv"
patientPersonaPath = "data/Persona/patient.json"
doctorPersonaPath = "data/Persona/doctor.json"
outputPath = "./data/Chats/Random/Chat"
# outputPath = "./data/Chats/scoreChat"


def one_chatdata_synthesis(maxChats, epoch):
    # 加载
    data = None
    pbdata = None
    ppdata = None
    dpdata = None

    # 随机选取患者背景数据
    df = pd.read_csv(patientBackgroundPath)
    pbdata = df.sample(n=1)

    with open(patientPersonaPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        ppdata = random.sample(data, 1)[0]

    with open(doctorPersonaPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dpdata = random.sample(data, 1)[0]

    # print(pbdata)
    # print(pbdata[0, '背景'].iloc[0])
    # print(pbdata[0, '主题'].iloc[0])
    # print(ppdata)
    # print(dpdata)
    
    # prompt
    patientSystemPrompt = "你是一位{}的希望向医生求助心理健康相关问题的患者，意味着{}。在与心理健康医生交流前，你的病患背景是：{}".format(ppdata['类型'], ppdata['解释'], pbdata['背景'].iloc[0])
    doctorSystemPrompt = "你是一位{}的{}专业心理健康医生，在与患者交流时会一直保持着以下的问诊习惯：你的交流风格很{}，{}在恰当的时间和患者进行共情交流的习惯，{}向患者解释一些心理健康领域的专业术语。在交流前，你了解到患者的背景是：{}".format(dpdata["年龄"], dpdata["性别"], dpdata["语言风格"], dpdata["同理心"], dpdata["解释"], pbdata['背景'].iloc[0])
    
    patient = Agent(globalModelPath, globalModelName, "patient", pbdata['主题'].iloc[0], ppdata)
    doctor = Agent(globalModelPath, globalModelName, "doctor", pbdata['主题'].iloc[0], dpdata)

    patient.model_init(patientSystemPrompt, False)
    doctor.model_init(doctorSystemPrompt, False)

    chats = []
    chatNum = random.randint(10, maxChats)

    history = []
    chats.append(pbdata.to_dict(orient='records')[0])

    for i in range(chatNum):
        # 对话
        patientResponse = patient.chat(history, 0.7, seed=42)
        history.append(patientResponse)
        
        doctorResponse = doctor.chat(history, 0.3, seed=128)
        history.append(doctorResponse)

        print('---')
        print(patientResponse)
        print(doctorResponse)

        if "EOF" in patientResponse:
            chats.append({"role": "user", "content": "我认为当前我自身的心理健康问题已得到改善"})
            break
        else:
            chats.append({"role": "user", "content": patientResponse})
            
        if "EOF" in doctorResponse:
            chats.append({"role": "assistant", "content": "我认为当前患者的心理健康问题已得到改善"})
            break
        else:
            chats.append({"role": "assistant", "content": doctorResponse})

    # 输出包括患者背景相关信息
    with open(outputPath + str(epoch) + '.json', 'w', encoding='utf-8') as o:
        json.dump(chats, o, ensure_ascii=False, indent=4)


def chatdatas(epoches=13, maxChats=20):
    for epoch in range(epoches):
        # if epoch % 10 == 0: 
        print('current epoch:', epoch)
        one_chatdata_synthesis(maxChats, epoch + 3)


if __name__ == '__main__':
    chatdatas()