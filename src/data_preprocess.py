import argparse
import json
import re
from pathlib import Path
from collections import Counter
from typing import List
from tqdm import tqdm
from utils import timer, ROOT_DIR



def is_chinese(word: str) -> bool:
    """
    判断一个字符串是否仅包含中文字符。
    """
    if not isinstance(word, str):  # 确保输入参数为字符串类型
        return False
    if len(word) == 0:  # 确保输入参数为非空字符串
        return False
    if re.search('[^\u4e00-\u9fa5]', word) is None:
        return True
    else:
        return False


@timer
def gen_word_count(input_files: List[Path], input_file_encoding: str, input_json_keys: List[str], 
                   output_dir: Path, max_word_len: int) -> None:
    """
    将原始数据中所有一元、二元和（或三元）的字组合加到词频表中
    :param input_files: Path 类对象组成的列表，代表原始数据文件
    :param input_file_encoding: 原始数据文件的编码方式
    :param input_json_keys: 原始数据文件中文本对应的键
    :param output_dir:  一个 Path 类对象，代表词频表所在的目录
    :param max_word_len: 每个词的长度（等于 2 或 3）
    :return: 
    """

    def is_all_target_chars(word: str, target_set: set) -> bool:
        """
        判断一个字符串是否只由 target_set 中的字符组成
        :param word: 待检测字符串
        :param target_set: 合法来源字符构成的集合
        :return: 返回值为 True 当且仅当该字符串只由 target_set 中的字符组成
        """
        if not isinstance(word, str):  # 确保输入参数为字符串类型
            return False
        for char in word:
            if char not in target_set:
                return False
        return True

    # 超参数
    MAX_HEAD_COUNTS = [4000, 2_000_000]  # 超参数，句首词频只保留前多少个
    MAX_TOTAL_COUNTS = [6000, 2_000_000, 2_000_000]  # 超参数，全文词频只保留前多少个

    # 输出文件预处理
    total_count_files = [output_dir / f'total_count_{i}.json' for i in range(1, max_word_len + 1)]  # 全文词频
    head_count_files = [output_dir / f'head_count_{i}.json' for i in range(1, max_word_len)]  # 每句句首词频
    for file in total_count_files + head_count_files:
        # 如果词频文件不存在，则创建该文件及其父目录
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()

    # 如果此函数首次被调用，从零开始统计词频
    total_counters = [Counter() for _ in range(len(total_count_files))]
    head_counters = [Counter() for _ in range(len(head_count_files))]
    if hasattr(gen_word_count, 'has_been_called'):
        # 如果此函数之前被调用过，那么需要在原统计结果之上追加
        for total_counter, total_count_file in zip(total_counters, total_count_files):
            with open(total_count_file, 'r') as f:
                total_counter.update(Counter(json.load(f)))
        for head_counter, head_count_file in zip(head_counters, head_count_files):
            with open(head_count_file, 'r') as f:
                head_counter.update(Counter(json.load(f)))
    # 打上已被调用标记
    gen_word_count.has_been_called = True


    # 逐个处理输入文件
    for input_file in input_files:
        with open(input_file, 'r', encoding=input_file_encoding) as f:
            for line in tqdm(f.readlines()):
                # 载入一行数据
                data = json.loads(line)
                # 统计该行结果，只考虑中文字符
                for key in input_json_keys:
                    text = data[key]
                    if len(text) >= 2:  # 只统计长度至少为 2 的句子
                        # 统计全文词频
                        for idx in range(len(total_counters)):
                            max_word_len = idx + 1
                            curr_counter = Counter([text[i:i + max_word_len] for i in range(0, len(text) - max_word_len + 1) if is_chinese(text[i:i + max_word_len])])
                            total_counters[idx].update(curr_counter)
                        # 统计句首词频
                        for idx in range(len(head_counters)):
                            max_word_len = idx + 1
                            if is_chinese(text[0:max_word_len]):
                                head_counters[idx].update({text[0:max_word_len] : 1})

    # 读入一二级汉字表，存储为集合
    with open(ROOT_DIR / 'raw_data' / '一二级汉字表.txt', 'r', encoding='gbk') as f:
        common_char = {char for char in f.read() if is_chinese(char)}

    # 只保留完全由一二级汉字构成的字符串，以及频率在前若干（由超参数指定）的结果
    for idx in range(len(total_counters)):
        total_counters[idx] = Counter({k:v for k,v in total_counters[idx].items() if is_all_target_chars(word=k, target_set=common_char)})
        total_counters[idx] = Counter({k:v for k,v in total_counters[idx].most_common(MAX_TOTAL_COUNTS[idx])})
    for idx in range(len(head_counters)):
        head_counters[idx] = Counter({k:v for k,v in head_counters[idx].items() if is_all_target_chars(word=k, target_set=common_char)})
        head_counters[idx] = Counter({k:v for k,v in head_counters[idx].most_common(MAX_HEAD_COUNTS[idx])})

    # 计算全文字数，并存入一元词频表中
    total_sum = sum(total_counters[0].values())  # 一元句首词频表总字数
    head_sum = sum(head_counters[0].values())  # 一元句首词频表总字数
    total_counters[0].update({'sum': total_sum})
    head_counters[0].update({'sum': head_sum})

    # 将新结果写入文件
    for idx in range(len(total_count_files)):
        with open(total_count_files[idx], 'w') as f:
            json.dump(total_counters[idx], f, ensure_ascii=False, indent=2)
    for idx in range(len(head_count_files)):
        with open(head_count_files[idx], 'w') as f:
            json.dump(head_counters[idx], f, ensure_ascii=False, indent=2)
  


def gen_dictionary(input_file: Path, input_file_encoding: str, output_file: Path) -> None:
    """
    生成拼音汉字对应表。
    :param input_file: 拼音汉字对应表的原始文件
    :param input_file_encoding: 原始文件的编码方式
    :param output_file: 清洗后的拼音汉字对应表
    """

    dic = {}
    with open(input_file, 'r', encoding=input_file_encoding) as f:
        for line in f.readlines():
            words = line.split()
            dic[words[0]] = words[1:]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


def get_arg():
    """
    获取命令行参数。
    :return: 返回一个二元组，第 1 项代表训练集，第 2 项代表所使用的模型
    """

    universal_training_set = {'sina', 'baike', 'webtext'}

    parser = argparse.ArgumentParser(description='Choose training data set and model.')

    # 添加命令行参数
    parser.add_argument('-s', '--training_set', dest='training_set', type=lambda x: set(x.split(',')), default={'sina', 'baike'},
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

    return args.training_set , args.model


if __name__ == '__main__':
    # 生成拼音汉字对应表
    gen_dictionary(input_file=ROOT_DIR / 'raw_data' / '拼音汉字表.txt',
                   input_file_encoding='gbk',
                   output_file=ROOT_DIR / 'src' / 'dictionary.json')

    # 生成词频表
    training_set, model = get_arg()
    output_dir = ROOT_DIR / 'src' / 'word_count' / '_'.join(sorted(training_set))

    # 从 sina（新浪新闻 2016）中训练
    if 'sina' in training_set:
        gen_word_count(input_files=[input_file for input_file in (ROOT_DIR / 'raw_data' / 'sina_news_gbk').glob('*') if "README" not in input_file.name],
                       input_file_encoding='gbk',
                       input_json_keys=['html', 'title'],
                       output_dir=output_dir,
                       max_word_len=model)

    # 从 baike（百科问答 2018）中训练
    if 'baike' in training_set:
        gen_word_count(input_files=[ROOT_DIR / 'raw_data' / 'baike2018qa' / 'baike_qa_train.json'],
                       input_file_encoding='utf-8',
                       input_json_keys=['title', 'desc', 'answer'],
                       output_dir=output_dir,
                       max_word_len=model)
