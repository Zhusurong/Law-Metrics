from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import numpy as np
import hanlp
import re
import chinese2digits as c2d
import os
import time
import argparse
from copy import copy
from tqdm import tqdm
import spacy
import editdistance
from scipy.optimize import linear_sum_assignment
import re
import json
import random

MAX_ITER = 100000

def contain_intro_text(txt):
    pattern = r"注意|以上|这些|上述|根据|以下|内容"
    return re.search(pattern, txt)

def norm_law_item(text):
    text = text.replace("*", "")
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

def extract_bracketed_content(text):
    # 正则表达式模式，匹配《》中的内容
    pattern = r'《(.*?)》'
    # 使用 re.findall() 提取所有匹配的内容
    matches = re.findall(pattern, text)
    return matches

def extract_numbers(s: str) -> list:
    # 使用正则表达式查找字符串中的所有数字
    numbers = re.findall(r'\d+', s)
    return numbers

def extract_name_content(text):
    # 定义正则表达式模式
    pattern = r'.某.?'
    
    # 查找所有匹配的子串
    matches = re.findall(pattern, text)
    
    # 过滤掉非中文字符、标点符号和空格，并处理“x某”与“x某某”或“x某x”的情况
    filtered_matches = []
    seen = set()
    
    for match in matches:
        # 移除标点符号和空格
        clean_match = re.sub(r'[^\u4e00-\u9fff]', '', match)
        # 检查是否只包含中文字符并长度合适
        base = clean_match[:2]
        if base not in seen:
            filtered_matches.append(clean_match)
            seen.add(base)
    
    # 去重并按原始顺序排序
    # filtered_matches = sorted(set(filtered_matches), key=matches.index)
    
    return filtered_matches


def generate_mappings(list1, list2, mapping, too_long=False):
    # print(list1, list2)
    # if len(list1) == 0:  # 如果 list1 为空，表示已经找到一组可能的对应关系
    
    if len(list1) == 0:  # 如果 list1 为空，表示已经找到一组可能的对应关系
        return [mapping]
    else:
        result = []
        for idx, item in enumerate(list2):
            new_mapping = mapping + [[list1[0], item]]
            result.extend(generate_mappings(list1[1:], [x for i, x in enumerate(list2) if i != idx], new_mapping, too_long))
            if too_long:
                if len(result) > MAX_ITER:
                    return result
        return result

def all_possible_mappings(l1, l2, too_long=False):
    if len(l1) > len(l2):
        temp = copy(l2)
        l2 = copy(l1)
        l1 = temp
    # 递归函数，生成所有可能的一一对应关系
    # 调用递归函数并返回结果
    return generate_mappings(l1, l2, [], too_long)

def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    return normalized_text

def unify_symbols(text):
    # 将中文标点替换为英文标点
    text = text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'").replace('\"\"', '\'').replace("参考答案:", "").replace("参考答案", "").replace(" 参考答案", "").replace("#", "").replace(" ", "").strip()
    text = c2d.takeNumberFromString(text)["replacedText"]
    return text

def remove_numbered_prefix(text):
    # 使用正则表达式匹配以数字加点开头的文本，并替换为空
    cleaned_text = re.sub(r'^\d+\.\s*', '', text)
    return cleaned_text

def remove_punctuation_regex(text):
    # 使用正则表达式去除标点符号
    clean_text = re.sub(r'[^\w\s]', '', text)
    return clean_text

def endwith_fa(txt):
    for name in "法;法典;条例;通知;规定;规章;解释;条约;批复;修正案;规则;决定;意见;答复;方案;决议;告;规划;令;细则;通则;函;答复;通报;报告;标准;指示;纲要;大纲".split(";"):
        if txt.find(name) != -1:
            return True
    return False

def all_false(txt):
    return False

def add_dict(dict1, dict2):
    # 将dict1作为基础字典，将dict2中的相同key的值相加
    sum_dict = dict1.copy()  # 首先复制dict1
    # 遍历dict2中的每个key
    for key in dict2:
        if key in sum_dict:  # 如果key在sum_dict中
            for sub_key in dict2[key]:
                if isinstance(dict2[key][sub_key], (int, float)):  # 确保是数值类型
                    sum_dict[key][sub_key] += dict2[key][sub_key]  # 累加值
        else:
            sum_dict[key] = dict2[key]  # 如果key不在sum_dict中，添加这个key
    return sum_dict

def add_1layer_dict(dict1, dict2, divide=1):
    # 创建一个新的字典来存储结果
    sum_dict = dict1.copy()  # 首先复制dict1
    for key in sum_dict.keys():
        sum_dict[key] = sum_dict[key] / (divide+1e-12)

    # 遍历dict2中的每个key
    for key, value in dict2.items():
        if key in sum_dict:  # 如果key在sum_dict中
            if isinstance(value, (int, float)):  # 确保是数值类型
                sum_dict[key] += (value / (divide+1e-12))  # 累加值
        else:
            sum_dict[key] = value / (divide+1e-12)  # 如果key不在sum_dict中，添加这个key

    return sum_dict

def div_dict(sum_dict, div_fac):
    for key in sum_dict:
        for sub_key in sum_dict[key]:
            if isinstance(sum_dict[key][sub_key], (int, float)):  # 确保是数值类型
                sum_dict[key][sub_key] /= div_fac  # 除以除数
    return sum_dict

def compute_edit_distance_matrix(list1, list2):
    # 计算编辑距离矩阵
    len1 = len(list1)
    len2 = len(list2)
    distance_matrix = np.zeros((len1, len2), dtype=int)
    
    for i in range(len1):
        for j in range(len2):
            distance_matrix[i][j] = editdistance.eval(list1[i], list2[j])
    
    return distance_matrix

def precision_recall_f1_from_edit_distance(list1, list2, threshold=1, only_true_num=False):
    list1 = list(set(list1))
    list2 = list(set(list2))
    # 计算编辑距离矩阵
    distance_matrix = compute_edit_distance_matrix(list1, list2)
    
    # 使用匈牙利算法找到最小总编辑距离
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # 计算符合阈值的匹配数
    matches = sum(distance_matrix[row_ind[i], col_ind[i]] <= threshold for i in range(len(row_ind)))
    
    if only_true_num:
        return matches
    # 计算 precision, recall, f1 score
    precision = matches / (len(list1) if len(list1) > 0 else 0 + 1e-12)
    recall = matches / (len(list2) if len(list2) > 0 else 0 + 1e-12)
    f1_score = 2 * (precision * recall) / ((precision + recall) if (precision + recall) > 0 else 0 + 1e-12)
    
    return precision, recall, f1_score

def remove_numbers(s):
    """
    移除字符串中的数字部分
    """
    return re.sub(r'\d+', '', s)

def calculate_scaled_distance(num1, num2):
    """
    计算两个数字之间的距离，并根据数字的位数进行缩放
    """
    num1 = float(extract_numbers(num1)[0])
    num2 = float(extract_numbers(num2)[0])
    distance = abs(num1 - num2)
    # 根据数字的位数进行缩放
    # len_num1 = len(str(num1))
    # len_num2 = len(str(num2))
    # 根据最大位数来选择缩放因子
    # max_length = min(len_num1, len_num2)
    # scale_factor = 10 ** (max_length - 1)
    # return distance / scale_factor
    scale_factor = max(abs(num1), abs(num2))
    if scale_factor == 0:
        scale_factor = 1.0
    return distance / (scale_factor + 1e-12)

def calculate_edit_distance(s1, s2):
    """
    计算两个字符串的非数字部分之间的编辑距离
    """
    s1_non_numeric = remove_numbers(s1)
    s2_non_numeric = remove_numbers(s2)
    return editdistance.eval(s1_non_numeric, s2_non_numeric)

def number_distance(est_lines, label_lines):
    results = []
    possible_map = all_possible_mappings(est_lines, label_lines, too_long=True)
    if len(possible_map) > MAX_ITER:
        possible_map = random.sample(list(possible_map), MAX_ITER)
    for map_ in possible_map:
        cur_map_dis = 0
        num = 0
        for map_i in map_:
            num_distance = calculate_scaled_distance(map_i[0], map_i[1])
            # edit_distance = calculate_edit_distance(word1, word2)
            # 综合距离可以考虑两者的加权和
            # combined_distance = num_distance + edit_distance
            cur_map_dis += num_distance
            num += 1
        cur_map_dis = cur_map_dis / (num + 1e-12)
        results.append(cur_map_dis)
    return np.min(results)

def filter_names(input_list):
    # 定义一个正则表达式来匹配人名，不包括包含数字的字符串
    name_pattern = re.compile(r"^[\u4e00-\u9fa5]{2,}$|^[\u4e00-\u9fa5]{2,}\w+$")
    # 过滤出匹配正则表达式的字符串
    filtered_names = [item for item in input_list if name_pattern.fullmatch(item)]
    # 按照字符串长度排序，长度相同则保持原有顺序
    filtered_names.sort(key=lambda x: len(x), reverse=True)
    # 去除包含关系的部分
    result = []
    for name in filtered_names:
        if not any(name in longer_name for longer_name in result):
            result.append(name)
    transform_pattern = re.compile(r"^([\u4e00-\u9fa5])某[\u4e00-\u9fa5]$")
    for i in range(len(result)):
        if transform_pattern.fullmatch(result[i]):
            result[i] = result[i][:2] 
    return result

def extract_numeric_parts(input_list):
    # 定义一个正则表达式来匹配数字
    digit_pattern = re.compile(r"\d+")

    # 过滤出每个字符串中的数字部分
    numeric_parts = []
    for item in input_list:
        numbers = digit_pattern.findall(item)
        if numbers:
            numeric_parts.append("".join(numbers))
    
    return numeric_parts

def filter_out_numeric_strings(input_list):
    # 定义一个正则表达式来检测包含数字的字符串
    digit_pattern = re.compile(r'\d')

    # 过滤掉包含数字的字符串
    filtered_list = [item for item in input_list if not digit_pattern.search(item)]

    return filtered_list

class Metric():
    def __init__(self):
        super(Metric, self).__init__()
        self.tok = hanlp.load(r"/Work20/2023/wangtianrui/codes/law/coarse_electra_small_20220616_012050")
        self.nlp_pos = spacy.load("zh_core_web_sm")
    
    def nouns(self, estimation, label):
        estimation = unify_symbols(estimation).strip().replace("、", ";")
        label = unify_symbols(label).strip().replace("、", ";")
        if estimation.find(":") != -1:
            estimation = estimation.split(":")[-1]
        if label.find(":") != -1:
            label = label.split(":")[-1]
        if label == "":
            label = "无"
        if estimation == "":
            estimation = "无"
        precision, recall, f1_score = self.acc(
            [item.strip() for item in estimation.split(";")], 
            [item.strip() for item in label.split(";")]
        )
        return {
            "precision": precision,
            "recall": recall,
            "F1Score": f1_score,
        }
    
    def department_nouns(self, estimation, label):
        estimation = unify_symbols(estimation).strip().replace("、", ";")
        label = unify_symbols(label).strip().replace("、", ";")
        if estimation.find(":") != -1:
            estimation = estimation.split(":")[-1]
        if label.find(":") != -1:
            label = label.split(":")[-1]
        if label == "":
            label = "无"
        if estimation == "":
            estimation = "无"
        precision, recall, f1_score = self.acc_with_contain(
            [item.strip() for item in estimation.split(";")], 
            [item.strip() for item in label.split(";")]
        )
        return {
            "precision": precision,
            "recall": recall,
            "F1Score": f1_score,
        }

    def merge_information(self, info):
        merged_info = {}
        
        # 解析输入信息
        for line in info.split('\n'):
            line = remove_numbered_prefix(line.strip())
            temp = line.split('》', 1)
            if len(temp) == 1:
                title, section = temp[0], " "
            else:
                title, section = line.split('》', 1)
            title = title.strip('《》')
            
            if title in merged_info:
                merged_info[title].append(section)
            else:
                merged_info[title] = [section]
        
        # 格式化输出
        output = ''
        for title, sections in merged_info.items():
            sections_str = ';'.join(sections)
            output += f'《{title}》{sections_str}\n'
        
        return output.strip()
    
    def reference(self, estimation, label):
        # re.sub(r'\(.*?\)', '', input_string)
        # estimation = self.merge_information(unify_symbols(estimation)).replace("中华人民共和国", "")
        # label = self.merge_information(unify_symbols(label)).replace("中华人民共和国", "")
        estimation = self.merge_information(re.sub(r'\(.*?\)', '', unify_symbols(estimation))).replace("中华人民共和国", "")
        label = self.merge_information(re.sub(r'\(.*?\)', '', unify_symbols(label))).replace("中华人民共和国", "")
        # print(estimation)
        # print(label)
        # print("-")
        # precision, recall, F1Score
        labels = []
        for line in label.split("\n"):
            line_labels = []
            for item in self.tok(remove_numbered_prefix(line)):
                item = remove_punctuation_regex(item)
                # print(item)
                if item.strip() != "":
                    line_labels.append(item)
                    # print(item)
            labels.append(line_labels)
        # 提取est信息
        ests = []
        for line in estimation.split("\n"):
            ests_labels = []
            for item in self.tok(remove_numbered_prefix(line)):
                item = remove_punctuation_regex(item)
                if item.strip() != "":
                    ests_labels.append(item)
            ests.append(ests_labels)
        
        possible_map = all_possible_mappings(ests, labels, too_long=True)
        if len(possible_map) > MAX_ITER:
            possible_map = random.sample(list(possible_map), MAX_ITER)
        pos_num = len(possible_map)
        
        choose_flag = []
        scores = []
        for map_ in possible_map:
            cur_map_p, cur_map_re, cur_map_f1, num = 0, 0, 0, 0
            # print(map_)
            for item in map_:
                # for item in items:
                if len(item[0]) == 0 or len(item[1]) == 0:
                    continue
                else:
                    precision, recall, f1 = self.acc_with_contain(item[0], item[1], endwith_fa) # est, label
                cur_map_p += precision
                cur_map_re += recall
                cur_map_f1 += f1
                num += 1
            if num == 0:
                continue
            cur_map_p, cur_map_re, cur_map_f1 = cur_map_p / num, cur_map_re / num, cur_map_f1 / num
            choose_flag.append(cur_map_f1)
            scores.append([cur_map_p, cur_map_re, cur_map_f1])
        # print(possible_map, choose_flag)
        best_map_idx = np.argmax(choose_flag)
        best_map = possible_map[best_map_idx]
        best_map_score = scores[best_map_idx]
        for i in range(len(best_map)):
            # print(best_map[i])
            best_map[i][0] = "-".join(best_map[i][0])
            best_map[i][1] = "-".join(best_map[i][1])
        # return best_map_score, best_map
        return {
            "precision": best_map_score[0],
            "recall": best_map_score[1],
            "F1Score": best_map_score[2],
        }
        
    def result(self, estimation, label):
        estimation = unify_symbols(estimation)
        label = unify_symbols(label)
        # print(estimation, label)
        est_lines = []
        for line in estimation.strip().split("\n"):
            # line = remove_punctuation_regex(remove_numbered_prefix(normalize_text(line.strip())))
            line = remove_numbered_prefix(normalize_text(line.strip()))
            if line.startswith(" 参考答案") or line.strip() == "" or line.startswith("参考答案"):
                continue
            else:
                est_lines.append(self.tok(line))
                
        label_lines = []
        for line in label.strip().split("\n"):
            # line = remove_punctuation_regex(remove_numbered_prefix(normalize_text(line.strip())))
            line = remove_numbered_prefix(normalize_text(line.strip()))
            if line.startswith(" 参考答案") or line.strip() == "" or line.startswith("参考答案"):
                continue
            else:
                label_lines.append(self.tok(line))
        
        est_num = len(est_lines)
        label_num = len(label_lines)
        num_group = 2
        est_lines_split = [est_lines]
        while est_num ** label_num > 300000:
            est_lines_split = list(np.array_split(est_lines, len(est_lines)//num_group))
            est_num = len(est_lines_split[0])
            num_group += 1
            if len(est_lines)//num_group < 2:
                break
            # print(est_num)
        # print(est_lines_split)
        final_map = []
        final_score = []
        for est_group in est_lines_split:
            # print(est_group)
            # print(label_lines)
            # est_group = est_group.tolist()
            possible_map = all_possible_mappings(est_group, label_lines, too_long=True)
            pos_num = len(possible_map)
            choose_flag = []
            scores = []
            if len(possible_map) > MAX_ITER:
                possible_map = random.sample(list(possible_map), MAX_ITER)
            for map_ in tqdm(possible_map, mininterval=2.0):
                item_info = {}
                num = 0
                for item in map_: # est, lab
                    if len(item[0]) == 0 or len(item[1]) == 0:
                        continue
                    else:
                        result = {
                            "score": precision_recall_f1_from_edit_distance(item[1], item[0], only_true_num=True)
                            # "bleu": self.scorer_bleu.compute_score({"0": [" ".join(item[1])]}, {"0": [" ".join(item[0])]})[0][0]
                            # "bleu": self.pos_acc(" ".join(item[1]), " ".join(item[0]), threshold=2)["overall"]["F1Score"]
                            # "cer": self.sentence_simi(" ".join(item[1]), " ".join(item[0]))
                        }
                        # print(item, result)
                        for key in result:
                            if key not in item_info.keys():
                                item_info[key] = result[key]
                            else:
                                item_info[key] += result[key]
                        num += 1
                for key in result:
                    item_info[key] = item_info[key]/num
                choose_flag.append(item_info["score"])
                scores.append(item_info)
            # print(possible_map)
            best_map_idx = np.argmax(choose_flag)
            possible_map[best_map_idx] = possible_map[best_map_idx][0]
            # temp = np.array(possible_map[best_map_idx], dtype=object)
            # if len(temp.shape) != 2:
            #     # print(np.array(possible_map[best_map_idx], dtype=object).shape)
            #     possible_map[best_map_idx] = possible_map[best_map_idx][0]
            final_map += [possible_map[best_map_idx]]
            # final_map.append(possible_map[best_map_idx])
            final_score.append(scores[best_map_idx])
        
        # print(final_map)
        # final_map = np.concatenate(final_map).tolist()
        score = None
        num = 0
        for item in final_map: # est, lab
            result = self.pos_acc(" ".join(item[0]), " ".join(item[1]))
            # print(result)
            if score is None:
                score = result.copy()
            else:
                score = add_dict(score, result)
            num += 1
        score = div_dict(score, num)
        return score

    def conclusion(self, estimation, label, label_refs=None):
        estimation = unify_symbols(estimation)
        label = unify_symbols(label)
        
        refs_est = extract_bracketed_content(estimation)
        refs_label = extract_bracketed_content(label)
        if len(refs_label) == 0:
            refs_label = label_refs
            refs_label = extract_bracketed_content(refs_label)
        ref_p, ref_recall, ref_f1 = self.acc_with_contain(refs_est, refs_label)
        
        estimation = " ".join(self.tok(remove_numbered_prefix(normalize_text(estimation.strip()))))
        label = " ".join(self.tok(remove_numbered_prefix(normalize_text(label.strip()))))
        result = self.pos_acc(estimation, label, threshold=1)
        
        result["reference_metric"] = {
            "precision": ref_p,
            "recall": ref_recall,
            "F1Score": ref_f1,
        }
        
        result["overall"] = add_1layer_dict(result["overall"], result["reference_metric"], divide=2)
        
        return result
    
    def acc(self, est_list, label_list, fn=all_false):
        # print(est_list, label_list)
        while "" in est_list:
            est_list.remove("")
        while "" in label_list:
            label_list.remove("")
        est_list = set(est_list)
        label_list = set(label_list)
        for label_item in label_list:
            if fn(label_item):
                if label_item not in est_list:
                    return 0.0, 0.0, 0.0
        true_positives = len(set(est_list) & set(label_list))
        precision = true_positives / ((len(est_list) if true_positives < len(est_list) else true_positives)+1e-12)
        recall = true_positives / ((len(label_list) if true_positives < len(label_list) else true_positives)+1e-12)
        f1_score = 2 * (precision * recall) / ((precision + recall) if precision + recall > 0 else 0 + 1e-12)
        assert precision <= 1 and recall <= 1 and f1_score <= 1
        return precision, recall, f1_score
    
    def acc_with_contain(self, est_list, label_list, fn=all_false):
        # print(est_list, label_list)
        while "" in est_list:
            est_list.remove("")
        while "" in label_list:
            label_list.remove("")
            
        est_list = set(est_list)
        label_list = set(label_list)
        for label_item in label_list:
            if fn(label_item):
                if label_item not in est_list:
                    return 0.0, 0.0, 0.0
                
        true_positives = 0
        for est_item in est_list:
            for label_item in label_list:
                if bool(re.search(r'\d', est_item)):
                    if est_item == label_item:
                        true_positives += 1
                        break
                else:
                    if est_item in label_item or label_item in est_item:
                        true_positives += 1
                        break
                    
        precision = true_positives / ((len(est_list) if true_positives < len(est_list) else true_positives)+1e-12)
        recall = true_positives / ((len(label_list) if true_positives < len(label_list) else true_positives)+1e-12)
        f1_score = 2 * (precision * recall) / (((precision + recall) if precision + recall > 0 else 0) + 1e-12)
        assert precision <= 1 and recall <= 1 and f1_score <= 1
        return precision, recall, f1_score

    def pos_acc(self, est, label, threshold=1):
        est_names = extract_name_content(est.replace(" ", ""))
        label_names = extract_name_content(label.replace(" ", ""))
        
        est = self.nlp_pos(est.replace(" ", ""))
        label = self.nlp_pos(label.replace(" ", ""))
        
        ests = {
            "propn": list(set(est_names)),
            "verb": [],
            "noun": [],
            "num": [],
        }
        for token in est:
            if token.pos_.lower() in ests.keys():
                ests[token.pos_.lower()].append(token.text)
                
        labels = {
            "propn": list(set(label_names)),
            "verb": [],
            "noun": [],
            "num": [],
        }
        for token in label:
            if token.pos_.lower() in labels.keys():
                labels[token.pos_.lower()].append(token.text)
        scores = {}
        weight = {
            "propn": 0.4,
            "verb": 0.1,
            "noun": 0.3,
            "num": 0.2,
        }
        overall = {
            "precision": 0,
            "recall": 0,
            "F1Score": 0,
        }
        for key in labels.keys():
            if key == "propn":
                ests[key] = filter_names(ests[key])
                labels[key] = filter_names(labels[key])
            if key == "num":
                ests[key] = extract_numeric_parts(ests[key])
                labels[key] = extract_numeric_parts(labels[key])
            if key == "noun":
                ests[key] = filter_out_numeric_strings(ests[key])
                labels[key] = filter_out_numeric_strings(labels[key])
            if len(ests[key]) == 0 and len(labels[key]) == 0:
                continue
            
            if len(ests[key]) == 0 and len(labels[key]) != 0 or \
                len(labels[key]) == 0 and len(ests[key]) != 0 :
                     p, r, f = 0, 0, 0
            else:
                # p, r, f = self.acc(ests[key], labels[key])
                if key in ["num"]:
                    # p, r, f = self.acc(ests[key], labels[key])
                    score = number_distance(ests[key], labels[key])
                    p, r, f = 1 - score, 1 - score, 1 - score
                else:
                    p, r, f = precision_recall_f1_from_edit_distance(ests[key], labels[key], threshold=threshold)
            scores[key] = {
                "precision": p,
                "recall": r,
                "F1Score": f,
            }
            overall["precision"] += weight[key] * p
            overall["recall"] += weight[key] * r
            overall["F1Score"] += weight[key] * f
        scores["overall"] = overall
        return scores

    def sentence_simi(self, est, label, threshold=1):
        est = re.sub(r'[^\w\u4e00-\u9fff]', '', est) 
        label = re.sub(r'[^\w\u4e00-\u9fff]', '', label) 
        return - editdistance.eval(est, label)

if __name__ == "__main__":
    import pandas as pd
    parser=argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default="/Work20/2023/wangtianrui/codes/law/benchmark/data_627/info.csv")
    parser.add_argument('--est', type=str, default="/Work20/2023/wangtianrui/codes/law/benchmark/datas/results_filted/resulternie_3.5_8k.csv")
    args = parser.parse_args()
    
    m = Metric()
    est_csv_path = args.est
    
    ref_csv = np.array(pd.read_csv(args.ref).fillna('无描述'))
    cur_csv_results = {}
    est_csv = np.array(pd.read_csv(est_csv_path).fillna('无描述'))
    for idx in tqdm(range(len(ref_csv)), mininterval=2.0):
        # print(ref_csv[idx])
        scores = {}
        assert len(ref_csv[idx]) == 16
        assert len(est_csv[idx]) == 16
        assert ref_csv[idx][0] == est_csv[idx][0], ref_csv[idx]+"|"+est_csv[idx][0]
        people_name_ref = ref_csv[idx][7]
        people_name_est = est_csv[idx][7]
        
        scores["people_name_score"] = m.nouns(people_name_est, people_name_ref)
        
        verbs_ref = ref_csv[idx][8]
        verbs_est = est_csv[idx][8]
        scores["verbs_score"] = m.nouns(verbs_est, verbs_ref)
        
        nouns_ref = ref_csv[idx][9]
        nouns_est = est_csv[idx][9]
        scores["nouns_score"] = m.nouns(nouns_est, nouns_ref)
        
        position_ref = ref_csv[idx][10].replace("中国", "中华人民共和国")
        position_est = est_csv[idx][10].replace("中国", "中华人民共和国")
        scores["position_score"] = m.department_nouns(position_est, position_ref)
        
        departments_ref = ref_csv[idx][11]
        departments_est = est_csv[idx][11]
        scores["departments_score"] = m.department_nouns(departments_est, departments_ref)
        
        law_nouns_ref = ref_csv[idx][12]
        law_nouns_est = est_csv[idx][12]
        scores["law_nouns_score"] = m.nouns(law_nouns_est, law_nouns_ref)

        dates_ref = ref_csv[idx][13].replace("至", ";").replace("-", ";")
        dates_est = est_csv[idx][13].replace("至", ";").replace("-", ";")
        scores["dates_score"] = m.nouns(dates_est, dates_ref)
        
        numbers_ref = ref_csv[idx][14].replace("-", ";")
        numbers_est = est_csv[idx][14].replace("-", ";")
        scores["numbers_score"] = m.nouns(numbers_est, numbers_ref)
        
        reference_ref = ref_csv[idx][3]
        reference_est = est_csv[idx][3]
        scores["reference_score"] = m.reference(reference_est, reference_ref)
        
        story_ref = ref_csv[idx][4]
        story_est = est_csv[idx][4]
        scores["story"] = m.result(story_est, story_ref)
        
        result_ref = ref_csv[idx][5]
        result_est = est_csv[idx][5]
        scores["result"] = m.result(result_est, result_ref)
        
        desc_ref = ref_csv[idx][6]
        desc_est = est_csv[idx][6]
        scores["conclusion"] = m.conclusion(desc_est, desc_ref, label_refs=ref_csv[idx][3])
        
        cur_csv_results[idx] = scores
        # print("第%d个案例的结果: \n"%idx + json.dumps(scores, indent=4,))
        # input()
    json_save = os.path.join(
        "/Work20/2023/wangtianrui/codes/law/benchmark/datas/eval_json",
        os.path.basename(est_csv_path)
    )
    with open(json_save, 'w') as json_file:
        json.dump(cur_csv_results, json_file)