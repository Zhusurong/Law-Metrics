from pathlib import Path
import docx
import pandas as pd

infos = {
    "title": [],
    "description": [],
    "Q1": [],
    "Q2": [],
    "Q3": [],
    "Q4": [],
    "Q5": [],
    "A1": [],
    "A2": [],
    "A3": [],
    "A4": [],
    "A5": []
}

def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".doc"):
            files.append(str(p))
        for s in p.rglob('*.docx'):
            files.append(str(s))
    return list(set(files))

def convert_docx_to_txt(docx_file, txt_file):
    """
    将Word文档转换为纯文本文件
    
    参数:
    docx_file (str): Word文档的路径
    txt_file (str): 生成的纯文本文件的路径
    """
    doc = docx.Document(docx_file)
    with open(txt_file, 'w', encoding='utf-8') as f:
        for para in doc.paragraphs:
            f.write(para.text + '\n')

for path in get_all_wavs(r"/Work20/2023/wangtianrui/codes/law/benchmark/data_627"):
    if path.find("__MACOSX") != -1:
        continue
    save_tgt = path.replace(".docx", ".txt").replace(".doc", ".txt")
    convert_docx_to_txt(path, save_tgt)
    # print("%s\t%s"%(save_tgt, ))