from utils.parameters import *
from utils.agent import Agent
import random
import json

# path
patientBackgroundPath = "data/PatientBackground/bg-example.json"
patientPersonaPath = "data/Persona/patient.json"
doctorPersonaPath = "data/Persona/doctor.json"
outputPath = "./data/Chats/c1.json"


def one_chatdata_synthesis(maxChats):
    # 加载
    data = None
    pbdata = None
    ppdata = None
    dpdata = None

    with open(patientBackgroundPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pbdata = random.sample(data, 1)

    with open(patientPersonaPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ppdata = random.sample(data, 1)

    with open(doctorPersonaPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dpdata = random.sample(data, 1)

    # prompt
    patientSystemPrompt = "你是一位{}的希望向医生求助心理健康相关问题的患者，意味着{}。在与心理健康医生交流前，你的病患背景是：{}".format(ppdata['类型'], ppdata['解释'], pbdata['背景'])
    doctorSystemPrompt = "你是一位{}的{}专业心理健康医生，在与患者交流时会一直保持着以下的问诊习惯：你的交流风格很{}，{}在恰当的时间和患者进行共情交流的习惯，{}向患者解释一些心理健康领域的专业术语。在交流前，你了解到患者的背景是：{}".format(dpdata["年龄"], dpdata["性别"], dpdata["语言风格"], dpdata["同理心"], dpdata["解释"], pbdata['背景'])
    
    patient = Agent(globalModelPath, globalModelName, "patient", pbdata, ppdata)
    doctor = Agent(globalModelPath, globalModelName, "doctor", pbdata, dpdata)

    patient.model_init(patientSystemPrompt)
    doctor.model_init(doctorSystemPrompt)

    chats = []
    chatNum = random.randint(3, maxChats)

    history = []
    chats.append(pbdata)

    for i in range(chatNum):
        # 对话
        patientResponse = patient.chat(history)
        history.append("患者：" + patientResponse)
        doctorResponse = doctor.chat(history)
        history.append("医生：" + doctorResponse)

        chats.append({"role": "user", "content": patientResponse})
        if "EOF" in doctorResponse:
            chats.append({"role": "assistant", "content": "医生认为当前患者的心理健康问题已得到改善"})
            break
        else:
            chats.append({"role": "assistant", "content": doctorResponse})

    # 输出包括患者背景相关信息
    with open(outputPath, 'w', encoding='utf-8') as o:
        json.dump(chats, o, ensure_ascii=False, indent=4)


def chatdatas(epoches=3, maxChats=10):
    for epoch in range(epoches):
        one_chatdata_synthesis(maxChats)


if __name__ == '__main__':
    chatdatas()