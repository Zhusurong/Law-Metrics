from pathlib import Path
import pandas as pd
import re

def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    return normalized_text

infos = {
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

def unify_symbols(text):
    # 将中文标点替换为英文标点
    text = text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    return text

def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".txt"):
            files.append(str(p))
        for s in p.rglob('*.txt'):
            files.append(str(s))
    return list(set(files))

all_txt = get_all_wavs(r"/Work20/2023/wangtianrui/codes/law/benchmark/data_627")

for txt in all_txt:
    # print(txt)
    with open(txt, "r") as rf:
        lines = rf.readlines()
    title = None
    describe = []
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    
    Q_idx = 0 # 当前案例
    D_start = False # 案情部分
    A_start = False # 答案部分
    for idx, line in enumerate(lines):
        line = unify_symbols(line.strip())
        line = normalize_text(line)
        if idx == 0:
            title = line.strip()
            continue
        if "# 案情描述" in line:
            D_start = True
            A_start = False
            if Q_idx < 1:
                if line != "":
                    describe.append(line)
            Q_idx += 1
            continue
        if "# 参考答案" in line:
            D_start = False
            A_start = True
            # continue
        if "你现在是中国的一名法律行业工作者" in line or "# 问题" in line:
            D_start = False
            A_start = False
            continue
        # print(line)
        if A_start:
            assert not D_start
            if Q_idx == 1:
                if line != "":
                    A1.append(line)
            if Q_idx == 2:
                if line != "":
                    A2.append(line)
            if Q_idx == 3:
                if line != "":
                    A3.append(line)
            if Q_idx == 4:
                if line != "":
                    A4.append(line)
            if Q_idx == 5:
                if line != "":
                    A5.append(line)
        if D_start and Q_idx < 1:
            assert not A_start
            if line != "":
                describe.append(line)
    infos["A1"].append("\n".join(A1))
    infos["A2"].append("\n".join(A2))
    infos["A3"].append("\n".join(A3))
    infos["A4"].append("\n".join(A4))
    infos["A5"].append("\n".join(A5))
    infos["description"].append("\n".join(describe))
    infos["title"].append(title)
    infos["ori_paths"].append(txt)
    
    for row in A1:
        # print(row)
        if str(row).startswith("人名"):
            infos["people_name"].append(row)
        elif str(row).startswith("动词"):
            infos["verbs"].append(row)
        elif str(row).startswith("普通名词"):
            infos["nouns"].append(row)
        elif str(row).startswith("地名") or str(row).startswith("事实地名"):
            infos["positions"].append(row)
        elif str(row).startswith("单位"):
            infos["departments"].append(row)
        elif str(row).startswith("法律专"):
            infos["law_nouns"].append(row)
        elif str(row).startswith("时间"):
            infos["dates"].append(row)
        elif str(row).startswith("数字"):
            infos["numbers"].append(row)
    tot_num = len(infos["title"])
    if tot_num != len(infos["people_name"]) or \
        tot_num != len(infos["verbs"]) or \
        tot_num != len(infos["nouns"]) or \
        tot_num != len(infos["positions"]) or \
        tot_num != len(infos["departments"]) or \
        tot_num != len(infos["law_nouns"]) or \
        tot_num != len(infos["dates"]) or \
        tot_num != len(infos["numbers"]):
            print(tot_num)
            print(len(infos["people_name"]))
            print(len(infos["verbs"]))
            print(len(infos["nouns"]))
            print(len(infos["positions"]))
            print(len(infos["departments"]))
            print(len(infos["law_nouns"]))
            print(len(infos["dates"]))
            print(len(infos["numbers"]))
            print(A1)
            print(txt)
            break
    assert Q_idx == 5, str(Q_idx) +"\t"+ txt
    
df = pd.DataFrame(infos)
df.to_csv("/Work20/2023/wangtianrui/codes/law/benchmark/data_627/info.csv", index=False, encoding='utf-8-sig')
    
        