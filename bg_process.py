import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.parameters import *
from utils.agent import Agent


# parameters
topics = ["情绪表现", "兴趣爱好", "心理状态", "睡眠情况", "食欲状况", "躯体症状", "社交功能", "自杀倾向", "其他情况"]
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

rawPath = "data/PatientBackground/raw/"
processedPath = "data/PatientBackground/processed/"


def basic_process(d):
    # 基础的字符串转换处理，去除不规范数据
    dataset = []
    error = []
    for i in d:
        try:
            data = json.loads(i)
            for j in data:
                # 规范年龄、性别、主题
                j["年龄"] = re.findall(r'\d+', str(j["年龄"]))[0]
                j["性别"] = "男" if "男" in j["性别"] else "女"
                j["主题"] = j["主题"] if j["主题"] in topics else "无"
                dataset.append(j)
        except Exception as e:
            error.append(i)
    
    return dataset, error


def first_load(datasetName, outputName, encoding='utf-8'):
    # parameters
    # datasetName = [
    #     "bg-appetite.json",
    #     "bg-example.json",
    #     "bg-interest.json",
    #     "bg-mental.json",
    #     "bg-mood.json",
    #     "bg-others.json",
    #     "bg-sleep.json",
    #     "bg-social.json",
    #     "bg-somatic.json",
    #     "bg-suicidal.json"
    # ]
    
    # datasetName = [
    #     "bg-appetite-pad.json",
    #     "bg-sleep-pad.json"
    # ]

    datasetName = datasetName
    
    datasets = []
    errors = []

    for name in datasetName:
        with open(rawPath + name, 'r', encoding='utf-8') as f:
            d = json.load(f)
            dataset, error = basic_process(d)
            datasets.extend(dataset)
            errors.extend(error)
    
    df = pd.DataFrame(datasets)
    print(df.head())
    df.to_csv(processedPath + outputName, index=False, encoding=encoding)

    with open(processedPath + "errors.json", "w", encoding='utf-8') as f:
        json.dump(errors, f, indent=4)


def label_deduplicate(name, threshold=0.1, encoding='utf-8'):
    # 使用去重
    df = pd.read_csv(processedPath + name, encoding=encoding)
    print(df.info())

    # 进行TF-IDF处理
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['背景'])
    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    keep_rows = set(range(len(df)))  # 所有行默认都被保留
    # 遍历相似度矩阵
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i, j] > threshold:
                # 查看相似文本
                print('\n-----')
                print(df['背景'][i])
                print(df['背景'][j])
                # 比较两个文本的长度，保留较长的
                if len(df['背景'][i]) < len(df['背景'][j]):
                    keep_rows.discard(i)
                else:
                    keep_rows.discard(j)

    # 保存
    df_retained = df.iloc[list(keep_rows)]
    print(df_retained.info())
    df_retained.to_csv(processedPath + "depu_" + name, index=False, encoding=encoding)

    # 使用 LLM + 人工填充缺失标签
    setupPrompt = "你是一位心理健康领域的专家学者，受邀参与一项关于共情对话主题的数据分类工作。你会得到病患的背景描述数据，请你通过分析病患背景描述数据来判断病患的核心病症。请注意，病患的核心病症只有以下 9 种类别：{}，其中它们对应的详细解释如下：{}。你的回答只能是你认为的 9 种核心病症中的最能概括和反映所提供的病患背景表现的那一种，注意你的回答只能是核心病症 9 种类别中的其中一个专有名词，只能有四个字，不需要你做任何的分析，不要说“根据您提供的信息”等敬语，如果你无法判断就返回“无”。".format('，'.join(topics), "；".join(dialogueTopics))
    
    padAgent = Agent(globalModelPath, globalModelName, 'pad')
    padAgent.model_init(setupPrompt)

    for idx, row in df_retained.iterrows():
        padTopic = "无"
        if row["主题"] == padTopic:
            genPrompt = "病患的背景描述数据为：{}。".format(row["背景"])
            response = padAgent.generate(genPrompt)
            print(response)

            if response in topics: padTopic = response
            df_retained.loc[idx, "主题"] = padTopic

    print(df_retained.info())
    df_retained.to_csv(processedPath + "pad_dupu_" + name, index=False, encoding=encoding)


def convert_to_int(val):
    try:
        # 尝试将数字字符串转换为整数
        return int(val)
    except Exception as e:
        # 如果转换失败则返回 0
        return 0


def label_score(name, encoding='utf-8'):
    # 大模型判断标签(是否正确 + 补齐) / 对数据质量打分 + 人工过滤
    # "标签一致性评分" "生成质量评分"
    df = pd.read_csv(processedPath + name, encoding=encoding)
    print(df.info())
    VconsistentScore = []
    VqualityScore = []
    PconsistentScore = []
    PqualityScore = []

    # Prompt
    volunteerPrompt = "你是一位心理健康专业的本科生志愿者，受邀参与一项关于共情对话主题的数据评分工作。你会得到可能存在心理健康问题的病患的背景和该病患的核心病症两个数据，你需要判断病患背景符合真实情况的程度来得到质量评分，并判断病患的核心病症符合病患背景描述的程度来得到一致性评分，最终你只能以以下的格式回答：\"(质量评分)|(一致性评分)\"。其中 (质量评分) 和 (一致性评分) 的括号和文字都要替换成你的两个评分数字，评分只能是 0 到 100 的整数，数字越大表示质量越高或者一致性越好，不能包括除了格式以外的内容。下面为你提供了辅助你评分的知识：1.质量评分的指标指病患背景数据和你所认为的真实情境下的病患之间的真实度差别；2.一致性评分的核心病症只有以下 9 种类型，对应的详细解释如下：{}。请结合核心病症的解释来判断病患背景的描述与核心病症的一致程度来进行一致性评分。".format("；".join(dialogueTopics))
    professionalPrompt = "你是一位心理健康领域的专家学者，受邀参与一项关于共情对话主题的数据评分工作。你会得到可能存在心理健康问题的病患的背景和该病患的核心病症两个数据，你需要判断病患背景符合真实情况的程度来得到质量评分，并判断病患的核心病症符合病患背景描述的程度来得到一致性评分，最终你只能以以下的格式回答：\"(质量评分)|(一致性评分)\"。其中 (质量评分) 和 (一致性评分) 的括号和文字都要替换成你的两个评分数字，评分只能是 0 到 100 的整数，数字越大表示质量越高或者一致性越好，不能包括除了格式以外的内容。下面为你提供了辅助你评分的知识：1.质量评分的指标指病患背景数据和你所认为的真实情境下的病患之间的真实度差别；2.一致性评分的核心病症只有以下 9 种类型，对应的详细解释如下：{}。请结合核心病症的解释来判断病患背景的描述与核心病症的一致程度来进行一致性评分。".format("；".join(dialogueTopics))

    # 加载模型
    volunteerAgent = Agent(globalModelPath, globalModelName, "volunteer")
    volunteerAgent.model_init(volunteerPrompt)

    professionalAgent = Agent(globalModelPath, globalModelName, "professional")
    professionalAgent.model_init(professionalPrompt)

    for idx, row in df.iterrows():
        # Prompt row["年龄"] row["性别"] row["主题"] row["背景"]
        genPrompt = "请结合你自己的身份和所提供的评分知识，严格遵守评分\"(质量评分)|(一致性评分)\"的评分格式进行两个分数的评分，其中 (质量评分) 和 (一致性评分) 的括号和文字都要替换成你的两个评分数字，评分只能是 0 到 100 的整数，数字越大表示质量越高或者一致性越好，不能包括除了格式以外的内容，不需要做任何分析，不要说“我认为分数是”这样的套话。提供的病患背景数据为：{}；提供病患核心病症为：{}".format(row["背景"], row["主题"])
        
        # "x|y"
        Vresponse = volunteerAgent.generate(genPrompt)
        score = Vresponse.replace("(", "").replace(")", "").split('|')
        # print(score)
        assert len(score) == 2
        VconsistentScore.append(score[0])
        VqualityScore.append(score[1])

        Presponse = professionalAgent.generate(genPrompt)
        score = Presponse.replace("(", "").replace(")", "").split('|')
        # print(score)
        assert len(score) == 2
        PconsistentScore.append(score[0])
        PqualityScore.append(score[1])

        if idx % 100 == 0: print("current idx: ", idx)

    df["志愿者一致性评分"] = VconsistentScore
    df["志愿者质量评分"] = VqualityScore
    df["专家一致性评分"] = PconsistentScore
    df["专家质量评分"] = PqualityScore
    df[["志愿者一致性评分", "志愿者质量评分", "专家一致性评分", "专家质量评分"]] = df[["志愿者一致性评分", "志愿者质量评分", "专家一致性评分", "专家质量评分"]].applymap(convert_to_int)

    print(df.info())
    df.to_csv(processedPath + "scored_" + name, index=False, encoding=encoding)


if __name__ == '__main__':
    # local_encoding = 'ansi'
    local_encoding = 'utf-8'
    
    # first_load(["bg-somatic-pad.json", "bg-somatic-pad2.json"], 'bg3.csv', local_encoding)

    # label_deduplicate("bg3.csv", 0.1, local_encoding)

    label_score('anno_pad_dupu_bg3.csv', local_encoding)

    # df = pd.read_csv(processedPath + "scored_anno_pad_dupu_bg2.csv", encoding=local_encoding)
    # df[["志愿者一致性评分", "志愿者质量评分", "专家一致性评分", "专家质量评分"]] = df[["志愿者一致性评分", "志愿者质量评分", "专家一致性评分", "专家质量评分"]].applymap(convert_to_int)
    # df.to_csv(processedPath + "scored_anno_pad_dupu_bg1-fix.csv", index=False, encoding=local_encoding)