# -*- coding : utf-8 -*-
# coding: utf-8

# import code
import csv
import jieba
import numpy as np
from collections import Counter  # 统计词频
from wordcloud import WordCloud  # wordcloud可视化
import PIL.Image as image  # 导入图像处理库
import numpy as np  # 导入数据处理库
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from pylab import mpl
from gensim import corpora, models, similarities  # tf-dif调用gensim包
from scipy import spatial
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
jieba.load_userdict(
    "D:\\学习\\大三\\大三上\\现代程序设计技术\\课程资料\\dataset\\week2\\stopwords_list.txt")


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


def count(wordlist):
    """
    词频统计,高低频词筛选,特征词确定
    :param wordlist: 进行分词操作后的全部词语--> list
    :return wordcountsort:排序后的词频 -->tuple
    :return wordcountfilter:特征词(词频>100) -->dict
    :return high_frequency:高频词(50个) -->list
    :return low_frequency:低频词(50个) -->list
    """
    # 统计词频-1
    wordcount = dict(Counter(wordlist))

    # #统计词频-2
    # wordcount={}
    # for word in wordlist:
    #     count=wordcount.get(word,0)
    #     wordcount[word]=count+1

    # 排序
    wordcountsort = sorted(
        wordcount.items(), key=lambda item: item[1], reverse=True)

    # 选前50为高频词，后50为低频词
    keys = dict(wordcountsort).keys()
    high_frequency = list(keys)[:49]
    low_frequency = list(keys)[-50:]

    # 保留词频大于500的为特征词,并创建字典
    wordcountfilter = {k: v for k, v in wordcount.items() if v > 500}
    return wordcountsort, wordcountfilter, high_frequency, low_frequency


def draw_from_dict(wordcountsort, wordlist_space_split):
    """
    用柱形图,词云图进行词频可视化
    :param wordcountsort: 排序后的词频--> list
    :return wordlist_space_split:" ".join(list)形式的词语 -->list
    """
    # 用词语绘制条形图
    x = []
    y = []
    for i in wordcountsort:
        x.append(i[0])
        y.append(i[1])
    plt.xticks(rotation=300)
    plt.bar(x[0:50], y[0:50])
    plt.show()
    savefig("D:\\学习\\大三\\大三上\\现代程序设计\\作业补充\\bar.png")

    # 用词语绘制词云图
    bg_pic = np.array(image.open("D:\\学习\\大三\\大三上\\现代程序设计\\作业补充\\3.png"))
    wc = WordCloud(
        font_path="C:\\Windows\\Fonts\\simhei.ttf",  # 字体路径
        width=240,
        height=240,
        max_words=100,  # 最多词个数
        max_font_size=100,  # 最大字号
        background_color='white',  # 背景色
        mask=bg_pic
    ).generate(wordlist_space_split)

    # #使用原图片的色彩
    # image_color = ImageColorGenerator(bg_pic)  #提取图片的色彩分布
    # a = wc.recolor(color_func = image_color)   #替换默认字体颜色
    # plt.imshow(a, interpolation='bilinear')    #Bilinear：双线性插值算法，用来缩放显示图片。缩放就是把原图片的像素应用坐标系统，用坐标表示                                      # 关闭坐标轴显示

    # 随机生成色彩
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')  # 最终从窗口显示词云wc.to_file("pjl_cloud1.jpg")
    plt.show()
    wc.to_file("D:\\学习\\大三\\大三上\\现代程序设计\\作业补充\\wordcloud.png")
    return


def distance_calculate(column, wordcountfilter, test):
    """
    通过余弦相似度和欧几里得算法比较两条弹幕之间的距离
    :param column: 每一个元素是一条弹幕的列表--> list
    :param wordcountfilter: 特征词 -->dict
    :param test: 测试弹幕 -->str
    :return matrix: 行数为弹幕数，列数为特征词数的0-1矩阵 -->array
    """
    # one-hot方法计算矩阵
    matrix = np.zeros((len(column), len(wordcountfilter)))
    for i, sentence in enumerate(column):
        for j, word in list(enumerate(seg_words(sentence)))[:len(wordcountfilter)]:
            if word in sentence:
                matrix[i, j] = 1
    # 用余弦相似度计算最近和最远的
    index = column.index(test)
    cos_sim = []
    for i in range(len(matrix)):
        cos_simi = 1-spatial.distance.cosine(matrix[index], matrix[i])
        cos_sim.append(cos_simi)
    gc_indexmin = cos_sim.index(min(cos_sim))
    gc_indexmax = cos_sim.index(max(cos_sim))
    print("----通过余弦相似度进行计算----")
    msg_cosmin = "距离测试弹幕'%s'最近的弹幕是:'%s',余弦相似度为%f" % (
        test, column[gc_indexmax], cos_sim[gc_indexmax])
    print(msg_cosmin)
    msg_cosmax = "距离测试弹幕'%s'最远的弹幕是:'%s',余弦相似度为%f\n" % (
        test, column[gc_indexmin], cos_sim[gc_indexmin])
    print(msg_cosmax)

    # 用欧几里得距离计算
    euclid = []
    for i in range(len(matrix)):
        euclidi = np.sqrt(np.sum(np.square(matrix[index]-matrix[i])))
        euclid.append(euclidi)
    gc_indexminx = euclid.index(min(euclid))
    gc_indexmaxx = euclid.index(max(euclid))
    print("----通过欧几里得距离进行计算----")
    msg_eucmin = "距离测试弹幕'%s'最近的弹幕是:'%s',欧几里得距离为%f" % (
        test, column[gc_indexminx], euclid[gc_indexminx])
    print(msg_eucmin)
    msg_eucmax = "距离测试弹幕'%s'最远的弹幕是:'%s',欧几里得距离为%f\n" % (
        test, column[gc_indexmaxx], euclid[gc_indexmaxx])
    print(msg_eucmax)
    return matrix

def get_distance(matrix):
    """
    确定向量重心,计算向量与重心距离
    :param matrix: 行数为弹幕数,列数为特征词数的0-1矩阵 -->array
    :return gravity_center: 存储中心坐标的列表 -->list
    :return dis_list: 存储每条弹幕和中心距离的列表 -->list
    """
    # 确定向量重心
    count = [0] * len(matrix[0])
    print("特征词数量为%d" % len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            count[j] += matrix[i][j]
    for i in range(len(count)):
        count[i] = count[i] / len(matrix)
    gravity_center = count
    print("所有特征向量的重心为: %s\n" % gravity_center)
    # 计算各向量与重心的距离
    dis_list = []
    for i in range(len(matrix)):
        dis = np.sqrt(np.sum(np.square(gravity_center - matrix[i])))
        dis_list.append(dis)
    return gravity_center, dis_list


def get_gc_danmu(column, dis_list):
    """
    根据向量重心获取最近和最远的弹幕
    :param column: 每个元素是一条弹幕的列表 -->list
    :param dis_list: 存储中心坐标的列表 -->list
    """
    gc_indexmin = dis_list.index(min(dis_list))
    gc_indexmax = dis_list.index(max(dis_list))
    msg1 = "距离重心最近的弹幕是:'%s' \n" % column[gc_indexmin]
    print(msg1)
    msg2 = "距离重心最远的弹幕是:'%s' \n" % column[gc_indexmax]
    print(msg2)
    return


def tfdif(compare_doc, refer_doc):
    """
    使用gensim包,采用TF-IDF方法进行文本相似度计算
    :param compare_doc: 待比较弹幕 -->str
    :param refer_doc: 参考弹幕 -->list
    :return sims[0]: 最相似的弹幕 -->list
    """
    # 语料库
    refer_words = []
    placeholder_count = 0
    for refer_word in refer_doc:
        words = seg_words(refer_word)
        if words:
            refer_words.append(words)
        else:  # 保证顺序一定
            placeholder_count += 1
            refer_words.append(
                seg_words('placeholder' + str(placeholder_count)))
    dictionary = corpora.Dictionary(refer_words)
    doc_vectors = [dictionary.doc2bow(word) for word in refer_words]

    # 建立语料库TF-IDF模型
    tf_idf = models.TfidfModel(doc_vectors)
    tf_idf_vectors = tf_idf[doc_vectors]
    compare_vectors = dictionary.doc2bow(seg_words(compare_doc))
    index = similarities.MatrixSimilarity(
        tf_idf_vectors, num_features=len(dictionary))
    sims = index[compare_vectors]

    # 对结果按相似度由高到低排序
    sims = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
    """
    index = similarities.MatrixSimilarity(tf_idf_vectors, num_features=len(dictionary), num_best=1)
    # 对结果按相似度由高到低排序
    sims = index[compare_vectors]
    """
    return sims[0]


if __name__ == '__main__':
    with open("D:\\学习\\大三\\大三上\\现代程序设计技术\\课程资料\\dataset\\week2\\弹幕数据\\danmuku.csv", 'rt', encoding='gbk', errors='ignore') as file:
        reader = csv.reader(file)
        # 获取初始弹幕文本
        column = [row[0] for row in reader][1:500000]

        # 进行文本切词
        wordlist = [x.strip() for x in seg_words(column) if x.strip() != '']

        # 统计词频,寻找特征词
        wordcountsort, wordcountfilter, high_frequency, low_frequency = count(
            wordlist)
        print("----高频词为：----")
        print(high_frequency)
        print("\n----低频词为：----")
        print(low_frequency)

        # 词频可视化
        draw_from_dict(wordcountsort, " ".join(wordcountfilter))

        test1 = '切菜怎么能切到大拇指？'
        test2 = '我爱吃卤菜！！！'

        # 余弦相似度和欧几里得算法计算
        matrix = distance_calculate(column, wordcountfilter, test1)

        # 找到最具代表性的弹幕
        gravity_center, dis_list = get_distance(matrix)
        get_gc_danmu(column, dis_list)

        # 通过TF-IDF方法计算
        similarity = tfdif(test2, column)
        print("----通过TF-IDF进行计算----")
        msg = "测试弹幕 '%s' 和参照弹幕中的 '%s' 最相似，相似度为 %f" % (
            test2, column[similarity[0]], similarity[1])
        print(msg)
