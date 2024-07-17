import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np

def remove_punctuation_regex(text):
    # 使用正则表达式去除标点符号
    clean_text = re.sub(r'[^\w\s]', '', text)
    return clean_text

def contain_intro_text(txt):
    pattern = r"以上|这些|上述|以下|如下"
    return re.search(pattern, txt)

def norm_law_item(text):
    text = text.replace("和", ";")
    text = text.replace("以及", ";")
    text = text.replace("-", "\n")
    text = text.replace("*", "")
    text = split_multiple_books(text)
    # 使用正则表达式匹配每个条目，将其整理成所需格式
    pattern = r'《.*?》'

    formatted_text = []
    for line in text.split("\n"):
        if re.search(pattern, line.strip()):
            formatted_text.append([line.strip(), []])
        else:
            if len(formatted_text) != 0 and line.strip() != "" and not contain_intro_text(line.strip()):
                formatted_text[-1][-1].append(remove_punctuation_regex(line).strip())

    new_lines = []
    for item in formatted_text:
        new_line = item[0] + ";".join(item[1])
        new_lines.append(new_line)
    return "\n".join(new_lines)

def split_multiple_books(txt):
    pattern = r'《[^》]+》[^《]*'
    matches = re.findall(pattern, txt)
    matches = [re.sub(r'\d+\.\s*', '', match) for match in matches]
    return "\n".join(matches)

def norm_names(input_text):
    input_text = input_text.replace("*", "")
    input_text = input_text.replace(" ", "")
    # 定义分类的正则表达式模式
    patterns = {
        "人名": r"人名(.*?)(?=\s*(?:动词|普通名词|地名|单位/机构|法律专业名词|时间|数字)|\s*$)",
        "动词": r"动词(.*?)(?=\s*(?:人名|普通名词|地名|单位/机构|法律专业名词|时间|数字)|\s*$)",
        "普通名词": r"普通名词(.*?)(?=\s*(?:人名|动词|地名|单位/机构|法律专业名词|时间|数字)|\s*$)",
        "地名": r"地名(.*?)(?=\s*(?:人名|动词|普通名词|单位/机构|法律专业名词|时间|数字)|\s*$)",
        "单位/机构": r"单位/机构(.*?)(?=\s*(?:人名|动词|普通名词|地名|法律专业名词|时间|数字)|\s*$)",
        "法律专业名词及名词短语": r"法律专业名词及名词短语(.*?)(?=\s*(?:人名|动词|普通名词|地名|单位/机构|时间|数字)|\s*$)",
        "时间": r"时间(.*?)(?=\s*(?:人名|动词|普通名词|地名|单位/机构|法律专业名词|数字)|\s*$)",
        "数字(保留单位)": r"数字\(保留单位\)(.*?)(?=\s*(?:人名|动词|普通名词|地名|单位/机构|法律专业名词|时间)|\s*$)"
    }

    # 定义函数提取信息
    def extract_info(text, pattern):
        matches = re.findall(pattern, text, re.MULTILINE)
        return ';'.join([match.strip() for match in matches if match.strip() != ""])

    # 提取并分类信息
    extracted_info = {key: extract_info(input_text, pattern) for key, pattern in patterns.items()}

    re_dict = {}
    # 打印提取的信息
    temp = []
    for key, value in extracted_info.items():
        if len(value) > 0:
            if value[0] in [":", "："]:
                value = value[1:]
        temp.append(f"{key}:{value}")
        re_dict[key] = f"{key}:{value}"
    return "\n".join(temp), re_dict

def norm_story(text):
    text = text.replace("*", "")
    text = text.replace(" ", "")
    temp = []
    for i, line in enumerate(text.strip().split("\n")):
        if text.strip() == "":
            continue
        if not re.match(r'^\d+[\.\．]', line) and i != 0:
            # 如果不以数字加点开头，则添加前缀 '1.'
            line = re.sub(r'^[^a-zA-Z0-9\u4e00-\u9fa5]+', '', line.strip())
            line = '%d. '%(i+1) + line.strip()
            if i != len(text.strip().split("\n"))-1:
                line = line + '1. '
        temp.append(line)
    text = "\n".join(temp)
    # 正则表达式匹配每个数字开头的段落
    # + re.findall(r'\-.*?(?=- |$)', text, re.DOTALL)
    matches = re.findall(r'\d+[\.\．]?[^\n]*?(?=\d+[\.\．]|$)', text, re.DOTALL)  
    # matches = re.findall(r'-.*?(?=- |$)', text, re.DOTALL)
    temp = []
    # 输出结果
    for match in matches:
        if match.strip() != "" and len(match.strip()) > 5:
            temp.append(match.strip())
        
    if len(temp) == 1:
        if "。" in text:
            temp = text.split("。")
        elif "." in text:
            temp = text.split(".")
        elif ";" in text:
            temp = text.split(";")
        elif "；" in text:
            temp = text.split("；")
            
    return "\n".join(temp)

def norm_result(text):
    text = text.replace("*", "")
    text = text.replace(" ", "")
    temps = []
    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            continue
        
        temp = None
        if "。" in text:
            temp = line.split("。")
        elif "." in text:
            temp = line.split(".")
            
        if temp is not None:
            for item in temp:
                temps.append(item)
        else:
            temps.append(line)
    return "\n".join(temps)

if __name__ == "__main__":
    save_home = r"/Work20/2023/wangtianrui/codes/law/benchmark/datas/results_filted"
    result_csvs = [
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultchatgpt.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultchatlaw.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultclaude.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultdoubao.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resulternie_3.5_8k.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultglm.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultkimi.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultminimax.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultqianfan_llama2.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultqwen.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultxuanyuan.csv",
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/results/resultYi.csv",
    ]
    for result_csv_path in result_csvs:
        result_csv = pd.read_csv(result_csv_path)
        result_csv = result_csv.fillna('')
        result_csv.to_csv(os.path.join(save_home, "ori_"+os.path.basename(result_csv_path)), index=False, encoding='utf-8-sig')
        results_dict = {
            "title": [],
            "description": [],
            "A1": [],
            "A2": [],
            "A3": [],
            "A4": [],
            "A5": [],
            "people_name": [],
            "verbs": [],
            "nouns": [],
            "positions": [],
            "departments": [],
            "law_nouns": [],
            "dates": [],
            "numbers": [],
            "ori_paths": []
        }
        for i, title, a1, a2, a3, a4, a5 in np.array(result_csv):
            results_dict["title"].append(title)
            results_dict["description"].append("-")
            a1, n_dict = norm_names(a1)
            results_dict["A1"].append(a1)
            results_dict["A2"].append(norm_law_item(a2))
            results_dict["A3"].append(norm_story(a3))
            results_dict["A4"].append(norm_result(a4))
            results_dict["A5"].append(a5)
            results_dict["people_name"].append(n_dict["人名"])
            results_dict["verbs"].append(n_dict["动词"])
            results_dict["nouns"].append(n_dict["普通名词"])
            results_dict["positions"].append(n_dict["地名"])
            results_dict["departments"].append(n_dict["单位/机构"])
            results_dict["law_nouns"].append(n_dict["法律专业名词及名词短语"])
            results_dict["dates"].append(n_dict["时间"])
            results_dict["numbers"].append(n_dict["数字(保留单位)"])
            results_dict["ori_paths"].append("-")
        results_dict = pd.DataFrame(results_dict)
        results_dict.to_csv(os.path.join(save_home, os.path.basename(result_csv_path)), index=False, encoding='utf-8-sig')
    