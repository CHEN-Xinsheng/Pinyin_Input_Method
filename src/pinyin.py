import json
import argparse
import math
from pathlib import Path
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
from utils import ROOT_DIR



def log_prob(word: str, total_count: Counter, head_count: Counter, coefficients: List[float],  max_word_len: int) -> float:
    """
    计算某一组合的条件概率（的对数）
    :param word: 需要计算的汉字或汉字组合
    :param total_count: 全文词频表
    :param head_count: 句首词频表
    :param coefficients: 模型使用的参数
    :param max_word_len: 使用基于字的 max_word_len 元模型
    :return: 某一组合的条件概率（的对数）
    """

    PENALTY = -math.inf  # 超参数，log(0) 的近似值。取一个绝对值较大的负数，用于惩罚不在词频表中的词（生僻词）
    PENALTY_FIRST = -math.inf  # 超参数，log(0) 的近似值。取一个绝对值较大的负数，用于惩罚不在词频表中的词（生僻词）

    x, y = coefficients[0], coefficients[1]

    # 第 1 个字
    if len(word) == 1:
        # 如果在句首词频表中出现，返回频率的对数
        try:
            return math.log(head_count[word] / head_count['sum'])
        # 如果不在句首词频表中出现，返回一个绝对值较大的负数
        except:
            return PENALTY_FIRST

    # 第 2 个字
    elif len(word) == 2:
        # 如果在全文/句首词频表中出现，返回频率的对数
        try:
            # 如果是二元模型，使用全文词频表
            if max_word_len == 2:
                return math.log(x * total_count[word] / total_count[word[0]]
                                + (1-x) * total_count[word[1]] / total_count['sum'])
            # 如果是三元模型，使用句首词频表
            else:
                return math.log(x * head_count[word] / head_count[word[0]]
                                + (1-x) * head_count[word[1]] / head_count['sum'])
        # 如果不在全文/句首词频表中出现，返回一个绝对值较大的负数
        except:
            return PENALTY

    # 第 3 个字
    else:
        # 如果在全文词频表中出现，返回频率的对数
        try:
            return math.log(x * total_count[word] / total_count[word[:2]]
                            + (1-x) * (y * total_count[word[1:]] / total_count[word[1]]
                                    + (1-y) * total_count[word[2]] / total_count['sum']))
        # 如果不在全文词频表中出现，返回一个绝对值较大的负数
        except:
            return PENALTY


def pinyin_to_hans(pinyin: str, total_count: Counter, head_count: Counter, 
                   dictionary: Dict[str, List[str]], coefficients: List[float], max_word_len: int) -> str:
    """
    将一段拼音转化为对应的简体汉字。
    :param pinyin: 输入全拼字符串
    :param total_count: 全文词频表
    :param head_count: 句首词频表
    :param dictionary: 拼音汉字对应表
    :param coefficients: 模型使用的参数
    :param max_word_len: 使用基于字的 max_word_len 元模型
    :return: 对应的简体汉字
    """

    class Node:
        def __init__(self, character: str):
            self.char = character
            self.prevs: List[Node] = [None]  # 在 Viterbi 图中的前驱节点（按概率从大到小排序取前若干个）
            self.log_prob: float = -math.inf  # 在 Viterbi 图中从起点至当前节点的最长路（与概率对应）
        
    def log_prob(word: str) -> float:
        """
        全局函数 log_prob 的包装。计算某一组合的条件概率（的对数）。
        :param word: 需要计算的汉字或汉字组合
        :return: 某一组合的条件概率（的对数）
        """
        global log_prob
        return log_prob(word, total_count, head_count, coefficients, max_word_len)

    graph = [[Node(char) for char in dictionary[each_pinyin]]
             for each_pinyin in pinyin.split()]
    MAX_PREV = 1  # 超参数。在基于字的三元模型中，最多考虑前一个字的路径最长（概率最大）的第 1 至 max_prev 个前驱


    # 基于字的二元模型
    if max_word_len == 2:
        for layer_idx in range(len(graph)):
            # 第 1 个字
            if layer_idx == 0:
                for node in graph[layer_idx]:
                    node.log_prob = log_prob(node.char)
            # 第 2 个字起
            else:
                for node in graph[layer_idx]:
                    # 键为当前节点的候选前驱；值为从起点出发，经过该候选前驱，到达当前节点的最长路径
                    candidates = {candidate: candidate.log_prob + log_prob(candidate.char + node.char)  
                                  for candidate in graph[layer_idx-1]}
                    node.prevs[0], node.log_prob = max(candidates.items(), key=lambda item: item[1])

    # 基于字的三元模型
    else:
        for layer_idx in range(len(graph)):
            # 第 1 个字
            if layer_idx == 0:
                for node in graph[layer_idx]:
                    node.log_prob = log_prob(node.char)
            # 第 2 个字
            elif layer_idx == 1:
                for node in graph[layer_idx]:
                    # 键为当前节点的候选前驱；值为从起点出发，经过该候选前驱，到达当前节点的最长路径
                    candidates = {candidate: candidate.log_prob + log_prob(candidate.char + node.char)
                                  for candidate in graph[layer_idx-1]}
                    candidates = dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True)[0:MAX_PREV]) # 保存前一个字的 max_prev 个最优前驱
                    node.prevs, node.log_prob = list(candidates.keys()), list(candidates.values())[0]
            # 第 3 个字起
            else:
                for node in graph[layer_idx]:
                    # 键为当前节点的候选前驱；值为从起点出发，经过该候选前驱，到达当前节点的最长路径
                    candidates = {candidate: 0 for candidate in graph[layer_idx-1]} 
                    for candidate in candidates.keys():
                        # 每个候选前驱与当前节点之间的边长度取如下计算的值：以该候选前驱的每个最优前驱分别作为第一个字计算，再取最大值
                        candidates[candidate] = candidate.log_prob + max([log_prob(prev.char + candidate.char + node.char) for prev in candidate.prevs])
                    candidates = dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True)[0:MAX_PREV])  # 保存前一个字的 max_prev 个最优前驱
                    node.prevs, node.log_prob = list(candidates.keys()), list(candidates.values())[0]
                    
    # 找出 Viterbi 图中第一层到最后一层的最长路
    candidates = {node: node.log_prob for node in graph[-1]}
    curr_node, _ = max(candidates.items(), key=lambda item: item[1])

    # 沿最长路径回溯
    answer = curr_node.char
    for _ in range(len(graph)-1):
        curr_node = curr_node.prevs[0] # 从每个节点回退到其最优的那个前驱
        answer += curr_node.char
    answer = ''.join(reversed(answer))

    return answer


def calc_word_accuracy(file: Path, std_file: Path) -> float:
    """
    计算字准确率。
    :param file: 拼音输入法计算结果（文件）
    :param std_file: 标准结果（文件）
    :return: 返回字准确率。
    """
    # 读入计算结果与标准结果文件
    with open(file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(std_file, 'r', encoding='utf-8') as f:
        std_sentences = f.readlines()
    # 逐行逐字统计正确字数
    total = 0
    correct = 0
    for (sentence, std_sentence) in zip(sentences, std_sentences):
        total += len(std_sentence)
        for (char, std_char) in zip(sentence, std_sentence):
            if char == std_char:
                correct += 1
    return correct / total


def calc_sentence_accuracy(file: Path, std_file: Path) -> float:
    """
    计算句准确率。
    :param file: 拼音输入法计算结果（文件）
    :param std_file: 标准结果（文件）
    :return: 返回句准确率。
    """
    # 读入计算结果与标准结果文件
    with open(file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    with open(std_file, 'r', encoding='utf-8') as f:
        std_sentences = f.readlines()
    # 逐行统计正确句数
    correct = 0
    for (sentence, std_sentence) in zip(sentences, std_sentences):
        if sentence == std_sentence:
            correct += 1
    return correct / len(std_sentences)


def get_arg():
    """
    获取命令行参数。
    :return: 返回一个元组，各项分别代表输入文件、输出文件、标准结果文件、参数、训练集、所使用的模型
    """

    universal_training_set = {'sina', 'baike', 'wiki'}
    coefficients_default = [0.9, 0.99]

    parser = argparse.ArgumentParser(description='Choose input file, output file, standard output file, training data set and model.')

    # 添加命令行参数
    parser.add_argument('-i', '--input_file', dest='input_file', type=Path, default= ROOT_DIR / 'data' / 'input.txt',
                        help='Input file')
    parser.add_argument('-o', '--output_file', dest='output_file', type=Path, default= ROOT_DIR / 'data' / 'output.txt',
                        help='Output file')
    parser.add_argument('-d', '--std_output_file', dest='std_output_file', type=Path, default= ROOT_DIR / 'data' / 'std_output.txt',
                        help='Standard output file')
    parser.add_argument('-c', '--coefficients', dest='coefficients', type=float, nargs='+', default=coefficients_default,
                        help='coefficients')
    parser.add_argument('-s', '--training_set', dest='training_set', type=lambda x: set(x.split('_')), default={'sina', 'baike'},
                        help=f'Choose training data set in {universal_training_set}.')
    parser.add_argument('-m', '--model', dest='model', type=int, default=3,
                        help='Choose training model, "2" for character-based binary model and "3" for character-based ternary model')

    # 从命令行中解析参数
    args = parser.parse_args()

    # 训练集参数必须在给定范围中
    try:
        assert args.training_set.issubset(universal_training_set)
    except:
        print(f'The argument "-s" or "--training-data-set" must in {universal_training_set}, but received {args.training_set}.')
        exit(1)
    try:
        assert args.model in [2, 3]
    except:
        print(f'The argument "-m" or "--model" must be "2" or "3", but received {args.model}.')
        exit(1)

    # 系数检查
    coefficients = args.coefficients
    try:
        assert len(coefficients) == 2
        assert 0 <= coefficients[0] <= 1 and 0 <= coefficients[1] <= 1
    except:
        print(f'The argument "-c" or "--coefficients" wrong.')
        exit(1)
    
    return args.input_file, args.output_file, args.std_output_file, coefficients, args.training_set, args.model


if __name__ == '__main__':
    # 处理命令行参数
    input_file, output_file, std_output_file, coefficients, training_set, max_word_len = get_arg()

    # 读入词频表
    total_count = Counter()
    head_count = Counter()
    word_count_path = ROOT_DIR / 'src' / 'word_count' / '_'.join(sorted(training_set))
    print(f'word_count_path = "{word_count_path}"')
    print(f'max_word_len = {max_word_len}')
    print(f'coefficients = {coefficients}')

    total_count_files = [word_count_path / f'total_count_{i}.json' for i in range(1, max_word_len + 1)]  # 全文词频文件
    head_count_files = [word_count_path / f'head_count_{i}.json' for i in range(1, max_word_len)]  # 句首词频文件
    for total_count_file in total_count_files:
        with open(total_count_file, 'r', encoding='utf-8') as f:
            total_count.update(Counter(json.load(f)))
    for head_count_file in total_count_files:
        with open(head_count_file, 'r', encoding='utf-8') as f:
            head_count.update(Counter(json.load(f)))

    # 读入拼音汉字对应表
    dictionary_path = ROOT_DIR / 'src' / 'dictionary.json'
    with open(dictionary_path , 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    # 输入拼音数据
    with open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
    
    # 计算拼音输入法结果
    output_list = [pinyin_to_hans(pinyin=line, total_count=total_count, head_count=head_count, 
                                  dictionary=dictionary, coefficients=coefficients, max_word_len=max_word_len)
                   for line in tqdm(lines)]

    
    # 输出输入法的计算结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_list:
            f.write(line + '\n')
    
    # 计算字准确率和句准确率，并输出
    word_accuracy = calc_word_accuracy(file=output_file, std_file=std_output_file)
    sentence_accuracy = calc_sentence_accuracy(file=output_file, std_file=std_output_file)
    print(f'字准确率：{word_accuracy}')
    print(f'句准确率：{sentence_accuracy}')
