"""
本部分负责文档的读取和预处理
"""
import os
import random
from PyPDF2 import PdfReader, PdfWriter
import jieba
from jieba import analyse
import re
import json
# 用于加载计算机学生简答题评分数据集
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# pdf的读取和预处理
# 读取PDF文件
def read_pdf(file_path):
    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)  # 页数
    page = reader.pages[0]  # 读取第0页
    text = page.extract_text()  # 转化为文本
    return text, number_of_pages


# 流式读取pdf文档
def process_folder(folder_path):
    # 存储所有文本内容的列表
    text_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print("Processing file:", file_path)

            # 提取文本内容
            text = crop_pdf(file_path)  # 调用crop_pdf函数来处理PDF文件并返回文本内容
            text_list.append(text)

    # 合并所有文本内容
    merged_text = "\n".join(str(text_list))
    return merged_text


# PDF文件预处理
def crop_pdf(input_path, x1=40, y1=65, x2=480, y2=675, docu_kind=True, output_path=""):
    """
    本函数用于pdf文档的预处理，并返回处理后的txt文档。
    **********************************************************************************************
    计算坐标从左下角开始（0， 0），A4纸右顶点大概为595x842。但文件不一定是A4纸的大小。可以通过pdfplumber来获取纸张大小
    下面是一个示例：
    import pdfplumber
    def get_pdf_page_size(pdf_path, page_num):
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            return page.width, page.height

    pdf_path = 'path/to/your/pdf/file.pdf'
    page_width, page_height = get_pdf_page_size(pdf_path, 0)  # 获取第一页的纸张尺寸
    print(f"页面宽度：{page_width}，页面高度：{page_height}")
    *********************************************************************************************
    :param docu_kind: 文件存储类型，T为txt文本，F为pdf文档
    :param input_path:文件名称
    :param output_path:储存名称
    :param x1:左宽，0为不压缩，每20单位大小大概一行
    :param y1:高度，从底部开始计算
    :param x2:右宽
    :param y2:高度
    :return:处理后的文档
    """
    # 打开PDF文件
    text = []
    with open(input_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        pdf_writer = PdfWriter()

        # 读取所有页
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # 设置裁剪区域
            page.mediabox.lower_left = (x1, y1)
            page.mediabox.upper_right = (x2, y2)

            # 添加裁剪后的页面到新的PDF文件中
            pdf_writer.add_page(page)

            # 读取裁剪后的pdf
            new_page = pdf_writer.pages[page_num]
            text.append(new_page.extract_text())

    # 除去非数学特殊字符
    processed_text = []
    for item in text:
        processed_text.append(remove_punctuation(item))

    # save
    if output_path != "":
        if docu_kind:
            document_save(processed_text, docu_kind=docu_kind, output_path=output_path)
        else:
            document_save(pdf_writer, docu_kind=docu_kind, output_path=output_path)
    return processed_text


# 辅助函数，文档清洗，除去非数学标点
def remove_punctuation(text):
    # 定义正则表达式，匹配非数学字符、空格、换行、普通中文标点和英文标点
    punctuation_pattern = r'[^\w+\s+\-*/=<>≠≈≤≥∞∑∫∏√∝∀∃∈∉∩∪⊆⊂⊄⊇⊃⊅，。？！；：“”‘’【】《》（）…—]'
    space_pattern = r'\s+'

    # 使用正则表达式替换非数学字符、空格、换行、普通中文标点和英文标点为空字符
    text = re.sub(punctuation_pattern, '', text)
    text = re.sub(space_pattern, '', text)

    return text


# 辅助函数，文件存储
def document_save(docu, output_path, docu_kind=True):
    if docu_kind:
        # 保存txt文件
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for item in docu:
                output_file.write(str(item))
    else:
        # 保存新的PDF文件
        with open(output_path, 'wb') as output_file:
            docu.write(output_file)


# 加载json文件
def read_json_data(data_name):
    """
    json文件中存在多行，利用for循环读取
    :param data_name: 文件名称
    :return: 读取后的文件
    """
    json_data = []
    for line in open(data_name, 'r', encoding='utf-8'):
        json_data.append(json.loads(line))

    data = [{'question': item['question'],  # 问题、答案及其标签
             'answerKey': item['answerKey'],  # 正确答案标签
             'combinedfact': item['combinedfact'],  # 结合事实
             'fact1': item['fact1'],  # 事实
             'fact2': item['fact2']  # 事实
             } for item in json_data]

    stem_list = []  # 题目
    answer_list = []  # 标准答案
    label_list = []  # 标签
    for item in data:
        stem_list.append(item['question']['stem'])
        answer_list.append(item['combinedfact'])

        # 获取标签
        answer_key = item['answerKey']
        choices = item['question']['choices']
        for choice in choices:
            if choice['label'] == answer_key:
                label_list.append(choice['text'])
                break

    return stem_list, answer_list, label_list


# 建立词向量和训练
# 文本分词
def word_segmentation(input_data):
    """
    输入文件或者文件目录，返回读取后的所有信息
    :param input_data:
    :return:
    """
    # 运行环境设置
    file_PATH = "D:/myAImodel/my_automatic_grading/knowledge"  # 运行环境
    os.chdir(file_PATH)
    # knowledge_folder_path = "D:\\myAImodel\\my_automatic_grading\\knowledge\\sensor"  # 预处理文本储存的目录
    stop_words = "stopwords.txt"  # 停用词
    lines = []
    # 判断传进来的是文件目录还是文本
    if os.path.isdir(str(input_data)):
        merged_text = process_folder(input_data)
    else:
        merged_text = input_data

    # 加载jieba库默认停用词
    jieba.analyse.set_stop_words(stop_words)
    for line in merged_text:  # 分别对每段分词
        temp = jieba.lcut(line)  # 结巴分词 精确模式
        words = []
        for i in temp:
            # 过滤掉所有的中文标点符号
            i = remove_punctuation(i)
            if len(i) > 0:
                words.append(i)
        if len(words) > 0:
            lines.append(words)
    print("分词完成")
    return lines


"""
********************************计算机学生简答题评分数据集********************************
"""


def read_computer_data(file_path='1.1'):
    """
    读取单个计算机简答评分数据集
    :return:读取后的列表
    """
    file_PATH = 'D:\\myAImodel\\my_dataset\\计算机科学课程的大量简短学生答案和成绩集合.Datasets\\data\\'
    with open(file_PATH + file_path, 'r', encoding='utf-8') as file:
        data_str = file.read()

    # 使用"\n"进行拆分，得到语料列表
    data_list = data_str.split('\n')
    # 去除可能存在的空字符串元素并除去特殊字符"<STOP>"
    data_list = [item.replace('<STOP>', '').strip() for item in data_list if item.strip()]

    # 去除问题和回答中的序号
    processed_data_list = []
    if "scores" not in file_path:
        for sentence in data_list:
            index = 0
            while (index < len(sentence)) and not sentence[index].isalpha():
                index += 1
            if index < len(sentence):
                processed_data_list.append(sentence[index:])
            else:
                print("该条语句为纯数字或为空,请检查，位于{}".format(file_PATH + file_path))
    else:
        processed_data_list = data_list

    return processed_data_list


def file_path_listdir():
    """
    排序文件夹，以匹配问题、教师答案、学生答案
    :return: 数字目录和文字目录
    """
    data_dir = 'D:\\myAImodel\\my_dataset\\计算机科学课程的大量简短学生答案和成绩集合.Datasets\\data\\sent\\'
    file_path = os.listdir(data_dir)

    path_num = []
    path_word = []
    for a_fila in file_path:
        if check_string_contains_digit(a_fila):
            path_num.append(a_fila)
        else:
            path_word.append(a_fila)

    i = 0
    for num_str in path_num:
        path_num[i] = float(num_str)
        i += 1
    path_num.sort()
    i = 0
    for num_float in path_num:
        path_num[i] = str(num_float)
        i += 1

    return path_num, path_word


def check_string_contains_digit(string):
    """
    判断是否包含数字
    :param string:
    :return: 包含T，不包含F
    """
    for char in string:
        if char.isdigit():
            return True
    return False


# @save
# 加载数据集
def load_con_dataset():
    """
    加载计算机简答评分数据集然后合并
    :return:返回处理好的数据集，包含前提、假设和标签
    """
    path_num, path_word = file_path_listdir()

    premises = []  # 前提-标准答案(教师答案)
    hypotheses = []  # 假设——学生答案
    labels = []
    # 读取教师回答
    teacher_answers = read_computer_data('raw\\answers')

    # 读取学生回答
    # 遍历数据集文件夹中的所有文件
    num_que = 0  # 一个计数器，用来取教师答案
    for filename in path_num:
        # 读取学生回答，将老师回答和学生回答进行拼接，并赋值给前提、假设、标签
        student_answer = read_computer_data('sent\\' + filename)
        teacher_answer = []
        for i in range(len(student_answer)):
            teacher_answer.append(teacher_answers[num_que])
        num_que += 1

        # 读取标签
        label = []
        for grade in read_computer_data('\\scores\\' + filename + '\\ave'):
            label.append(float(grade))

        premises += teacher_answer
        hypotheses += student_answer
        labels += label

    # 将数据打乱
    data = list(zip(premises, hypotheses, labels))  # 将三个列表打包在一起
    random.shuffle(data)
    shuffled_sentence_teachers, shuffled_sentence_student, shuffled_labels = zip(*data)

    return [shuffled_sentence_teachers, shuffled_sentence_student, shuffled_labels]


# 检查数据标签分布
def data_distribution(labels, draw=True):
    data_labels = [i * 0.5 for i in range(11)]
    distribute = [0] * 11
    for i in labels:
        i_dir = int(i * 2)
        distribute[i_dir] += 1

    if draw:
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制图像
        fig, ax = plt.subplots()
        bars = ax.bar(data_labels, distribute, width=0.4)

        # 在每个柱状图的上方显示数值
        for bar in bars:
            height = bar.get_height()
            ax.annotate('%.2f' % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords='offset points',
                        ha='center', va='bottom')

        # 设置刻度
        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax.set_xlabel('标签')
        ax.set_ylabel('数量')
        ax.set_title('data distribution')
        plt.show()

    return distribute


# 使用例子
# @save
# 定义数据集
class MyDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('D:\\myAImodel\\bert-base-chinese')
        self.dataset = data
        self.student_answers = self.dataset[1]
        self.teacher_answers = self.dataset[0]
        self.labels = self.dataset[2]
        self.max_length = max_length

    def __len__(self):
        return len(self.student_answers)

    def __getitem__(self, index):
        student_answer = self.student_answers[index]
        teacher_answer = self.teacher_answers[index]
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        # 使用BertTokenizer编码学生和老师答案
        encoded_inputs = self.tokenizer(student_answer, teacher_answer,
                                        padding='max_length',
                                        truncation='do_not_truncate',
                                        max_length=self.max_length,
                                        return_tensors='pt'
                                        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()
        token_type_ids = encoded_inputs['token_type_ids'].squeeze()

        return input_ids, attention_mask, token_type_ids, label


def collate_fn(batch):
    # 从batch中获取input_ids、attention_mask、token_type_ids和labels
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    token_type_ids = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)

    # 计算batch中的最大长度
    max_length = max(len(ids) for ids in input_ids)

    # 动态填充句子，保证batch中的所有句子都具有相同长度（最大长度为128）
    padded_input_ids = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                        for ids in input_ids]
    padded_attention_mask = [torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
                             for mask in attention_mask]
    padded_token_type_ids = [torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
                             for ids in token_type_ids]

    # 将填充后的张量转换为tensor并返回
    return torch.stack(padded_input_ids), torch.stack(padded_attention_mask), torch.stack(padded_token_type_ids), labels


# 加载数据集
def dataloader():
    # 加载并准备数据集
    dataset = MyDataset(load_con_dataset())
    # 定义训练集和验证集的比例
    train_ratio = 0.8
    # 计算数据集的样本数量
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    # 划分训练集和验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    return train_dataloader, val_dataloader

# 用于计算数据集分布
# distribution_num = data_distribution(load_con_dataset()[2])
# print(distribution_num)
