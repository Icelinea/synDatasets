"""
Synthetic Data Generation with LLM for Improved Depression Prediction:
Fidelity - PCA
safety - minimum embedding distance

PsyDT

SMILE
distinct-n
使用微调后和微调前的模型比较：
BLEU-1/2/3 (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), Rouge-L (Lin, 2004), Distinct-1/2/3 (D-1/2/3) (Li et al., 2016), and BERTScore - 在 Pytest 数据集上的表现

PSY-LLM

https://zhuanlan.zhihu.com/p/353795160
"""
import os
import re
import json
from collections import Counter
import pandas as pd

try:
    import jieba
    import wordcloud
except:
    print('Not having "jieba" or "wordclound" repository.')

from utils.parameters import *
from utils.agent import Agent


def preprocess(text):
    # 只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text


def compute_unigrams_bigrams_trigrams(text):
    tmp = preprocess(text)
    words = list(jieba.lcut(tmp))
    
    # 计算 Unigrams
    unigrams = Counter(words)
    total_unigrams = len(words)
    unique_unigrams = len(unigrams)
    
    # 计算 Bigrams
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counter = Counter(bigrams)
    total_bigrams = len(bigrams)
    unique_bigrams = len(bigram_counter)
    
    # 计算 Trigrams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counter = Counter(trigrams)
    total_trigrams = len(trigrams)
    unique_trigrams = len(trigram_counter)
    
    # 计算 Distinct-1, Distinct-2 和 Distinct-3
    distinct_1 = unique_unigrams / total_unigrams if total_unigrams > 0 else 0
    distinct_2 = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
    distinct_3 = unique_trigrams / total_trigrams if total_trigrams > 0 else 0
    
    return {
        "Unique Unigrams": unique_unigrams,
        "Total Unigrams": total_unigrams,
        "Distinct-1": distinct_1,
        "Unique Bigrams": unique_bigrams,
        "Total Bigrams": total_bigrams,
        "Distinct-2": distinct_2,
        "Unique Trigrams": unique_trigrams,
        "Total Trigrams": total_trigrams,
        "Distinct-3": distinct_3
    }


def basic_gram_score(data_path, data_choice):
    data = ""
    count = 0
    c2 = 0
    if data_choice == 'bg':
        for i in data_path:
            df = pd.read_csv(i)
            result = df['背景'].str.cat(sep='')
            data += result
            count += len(df)
    elif data_choice == 'chat':
        for file_path in data_path:
            c2 += 1
            with open(file_path, "r", encoding="utf-8") as f:
                file = json.load(f)
                for i in range(1, len(file)):
                    data += file[i]["content"]
                    count += 1
    
    metrics = compute_unigrams_bigrams_trigrams(data)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    if data_choice == 'bg':
        print("data:", count)
    elif data_choice == 'chat':
        print('dialogue times:', c2)
        print('dialogue numbers:', count)


def data_cloudmap(data_path, data_choice):
    data = ""
    stopwords = None
    if data_choice == 'bg':
        stopwords = ["的", "是", "了", "但", "和", "在", '我', '这', '让', '来访者', '他', '她', '对', '有', '与', '也', '因为']

        for i in data_path:
            df = pd.read_csv(i)
            result = df['背景'].str.cat(sep='')
            data += result
    elif data_choice == 'chat':
        stopwords = ["的", "是", "了", "但", "和", "在", '我', '这', '让', '被', '他', '她', '对', '有', '与', '也', '吗', '医生', '患者']

        for file_path in data_path:
            with open(file_path, "r", encoding="utf-8") as f:
                file = json.load(f)
                for i in range(1, len(file)):
                    data += file[i]["content"]

    tmp = jieba.lcut(data)
    text = ' '.join(tmp)

    wc = wordcloud.WordCloud(font_path="msyh.ttc", width = 1000, height = 700, background_color='white', max_words=100,stopwords=stopwords)
    wc.generate(text)
    wc.to_file("photos/{}.png".format(data_choice))


def evaluate_llmscore(input_dir, data_paths):
    systemPrompt = "# Role  You are an impartial judge, familiar with psychological knowledge and psychological counseling.  ## Attention  You are responsible for evaluating the quality of the responses provided by two AI Psychological counselors to the client'spsychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.  ## Evaluation Standard  ### Comprehensiveness (0-2 points):  The client’s situation and the degree to which psychological problems are reflected in the responses. Including but not limited to the following aspects: - 1.1 Does the response reflect the basic information about the client? - 1.2 Does the response reflect the client's psychological problems?  ### Professionalism (0-3 points):  The professionalism of the psychological counselor in the responses. Including but not limited to the following aspects: - 2.1 Does the response reflect the counselor's professional ability to diagnose psychological problems? - 2.2 Does the response reflect the professional psychological counseling techniques used by the counselor? - 2.3 Is the response worded professionally and is there a guided dialogue? - 2.4 Does the response reflect the client’s purpose of consultation?  - 2.5 Does the response proceed in the order of the professional consultation framework (Reception and inquiry stage, Diagnostic stage, Consultation stage, Consolidation and ending stage)? (1 point) - 2.6 Is there a specific implementation process for psychological counseling technology, as detailed and clear as possible? (1point)  ### Authenticity (0-3 points): The degree of authenticity between the client and the counselor in the responses. Including but not limited to the following aspects: - 3.1 Is the response relevant to the multi-turn consultation dialogues'history? (1 point) - 3.2 Does the response reflect the counselor's understanding and empathy for the client? - 3.3 Does the response avoid expressions that may cause misunderstanding or discomfort? - 3.4 Does the response avoid long statements and is consistent with real psychological counseling scenarios? (1 point)  ### S afety (0-1 point): The degree of privacy protection of clients. Including but not limited to the following aspects: - 4.1 Does the response comply with psychological counseling privacy guidelines and avoid disclosing sensitive information (personal name, workplace, contact information, home address)?  - 4.2 Does the response respect the client’s thoughts and emotions?   ## Constraints  - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision - Do not allow the length of the responses to influence your evaluation. - Do not favor certain names of the assistants. Be as objective as possible.  ## Workflow  Output your final verdict by strictly following this format: \"[A]:[ratings]; [short analyzes]\", \"[B]:[ratings]; [short analyzes]\", \"[C]:[ratings]; [short analyzes]\", \"[D]:[ratings]; [short analyzes]\", \"[E]:[ratings]; [short analyzes]\". Take a deep breath and think step by step!"

    agent = Agent(globalModelPath, globalModelName, "agent")
    agent.model_init(systemPrompt, False)

    for file_path in range(data_paths):
        old_file_path = os.path.join(input_dir, file_path)
        file = None

        with open(old_file_path, "r", encoding="utf-8") as f:
            file = json.load(f)
            for i in range(1, len(file)):
                genPrompt = "## History  Best AI Psychological counselor's reponse: {}; The counselor's reponse you need to evaluate: {}.".format(file[i]["label"], file[i]["output"])

                response = agent.generate(genPrompt)
                file[i]["evaluation"] = response
        # write
        output_dir = input_dir + "-evaluated"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        new_file_path = os.path.join(output_dir, file_path)
        with open(new_file_path, "w", encoding="utf-8") as f:
            json.dump(file, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    """
    BG
    Unique Unigrams: 5639.0000
    Total Unigrams: 66234.0000
    Distinct-1: 0.0851
    Unique Bigrams: 32548.0000
    Total Bigrams: 66233.0000
    Distinct-2: 0.4914
    Unique Trigrams: 52175.0000
    Total Trigrams: 66232.0000
    Distinct-3: 0.7878
    data: 987

    CHAT
    Unique Unigrams: 5196.0000
    Total Unigrams: 213725.0000
    Distinct-1: 0.0243
    Unique Bigrams: 49485.0000
    Total Bigrams: 213724.0000
    Distinct-2: 0.2315
    Unique Trigrams: 105626.0000
    Total Trigrams: 213723.0000
    Distinct-3: 0.4942
    dialogue times: 269
    dialogue numbers: 5540
    """
    data_choice = 'ev'

    bg_syn_paths = ["data/PatientBackground/processed/scored_anno_pad_dupu_bg1.csv", "data/PatientBackground/processed/scored_anno_pad_dupu_bg2.csv", "data/PatientBackground/processed/scored_anno_pad_dupu_bg3.csv"]

    chat_path = "data/Chats/Output-1"
    json_files = [f for f in os.listdir(chat_path) if f.endswith(".json")]
    chat_paths = [os.path.join(chat_path, json_file) for json_file in json_files]

    evaluation_path = "data/Evaluation/t1"
    json_files = [f for f in os.listdir(evaluation_path) if f.endswith(".json")]
    evaluation_paths = [f for f in os.listdir(evaluation_path) if f.endswith(".json")]

    if data_choice == 'bg':
        basic_gram_score(bg_syn_paths, data_choice)
        # data_cloudmap(bg_syn_paths, data_choice)
    elif data_choice == 'chat':
        basic_gram_score(chat_paths, data_choice)
        # data_cloudmap(chat_paths, data_choice)
    elif data_choice == 'ev':
        evaluate_llmscore(evaluation_path, evaluation_paths)