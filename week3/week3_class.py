# -*- coding : utf-8 -*-
# coding: utf-8

# import code
import jieba
import pandas as pd
import os
import re


# 添加词表到jieba库中
def jieba_update(path):
    '''
    将停用词、情绪词加入jieba词库中
    :param path: 目录文件夹地址 -->str
    :return filePaths: 添加的词表全路径  -->list
    '''
    filePaths = []  # 存储目录下的所有文件名，含路径
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
            jieba.load_userdict(os.path.join(root, file))
    return filePaths

def read_data(path):
    """
    读取txt文件，将其转化为列表形式
    :param path: txt文件路径 -->str
    :return weibo_list: 微博内容列表 -->list
    """
    weibo_list = []

    with open(path, 'r', encoding='utf-8') as file:
        n=0
        for line in file.readlines():
            n=n+1
            # 为防止死机，这里仅处理500000条数据
            if n<=100:
                line = line.strip('\n')
                weibo_list.append(line)
            else:
                break
    return weibo_list

def clean_text(text):
    """
    将微博文本进行清洗操作
    :param text: 清洗前的微博文本 -->list
    :return: 清洗后的微博文本  -->list
    """
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除 @ 和转发链 //
    text = re.sub(r"\[\S+?]", "", text)  # 去除微博表情 [文字]
    text = re.sub(r"#\S+#", "", text) # 保留tag
    addr_URL = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(addr_URL, "", text)  # 去除地址和 URL
    text = re.sub("转发微博", "", text)
    text = re.sub("我在:", "", text)
    text = re.sub("我在这里:", "", text)  # 去除无意义的字样
    text = re.sub(r"\s+", " ", text)  # 去除多余的空格
    text = re.sub(r"#", "", text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)   # 去除手机自带表情
    text = emoji_pattern.sub(r'', text)
    return text.strip()

def stopwordslist():
    """
    创建停用词列表
    :return stopwords: 评论数列表 -->list
    """
    stopwords = [line.strip() for line in open(
        "D:\\学习\\大三\\大三上\\现代程序设计技术\\课程资料\\dataset\\week2\\stopwords_list.txt", encoding='utf-8').readlines()]
    return stopwords

def seg_words(sentence):
    """
    根据停用词进行分词
    :param sentence: 每个元素是一条弹幕的列表 --> list
    :return wordlist: 进行分词操作后的全部词语 -->list
    """
    sentence_depart = jieba.cut(str(sentence))
    stopwords = stopwordslist()
    wordlist = list()
    for word in sentence_depart:
        if word not in stopwords:
            wordlist.append(word)
    return wordlist

def weibo_info_list(weibo_list):
    """
    将每一条的微博信息进行分类处理，整理在字典中以列表形式输出
    :param weibo_list: 全体微博数据 --> list
    :return weibo_info_list: 每一条微博的信息以字典形式存储而形成的一个列表 -->list
    """
    weibo_info_list = []
    for weibo in weibo_list:
        weibo_split = weibo.split('\t')
        location = weibo_split[0]
        text = weibo_split[1]
        text_clean = clean_text(text)
        text_split = seg_words(text_clean)
        user_id = weibo_split[2]
        time = weibo_split[3]
        weibo_info = {'text':text,'text_clean':text_clean,'text_split':text_split,'location':location,'user_id':user_id,'time':time}
        weibo_info_list.append(weibo_info)
    return weibo_info_list

        
if __name__ == '__main__':
    #更新jieba词库
    filePaths=jieba_update(r"D:\学习\大三\大三上\现代程序设计技术\课程资料\dataset\week3\weibo.txt\emotion_lexicon")
    print(filePaths)
    #读入数据集
    weibo_list=read_data(r"D:\学习\大三\大三上\现代程序设计技术\课程资料\dataset\week3\weibo.txt\weibo.txt")[1:]

    #将数据集进行处理
    weibo_info_list=weibo_info_list(weibo_list)




