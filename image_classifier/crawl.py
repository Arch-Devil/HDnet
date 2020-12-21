import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os

import re


# 评论列表
comments = []

# 提取评论，传入参数为网址url
def get_comment(url):

    global comments

    try:
        # 发送HTTP请求
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 \
                             (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'}
        r = requests.get(url=url, headers=headers)

        # 解析网页，定位到评论部分
        soup = BeautifulSoup(r.text, 'lxml')
        main_content = soup.find_all('div', class_='comment_single')

        # 提取评论
        for para in main_content:
            comment = para.find('span', class_='heightbox')
            print(comment.text)
            ttxt = comment.text.replace('&quot', '').split('http')[0].split('，请戳')[0]
            if len(ttxt)>20:
                comments.append(ttxt)

    except Exception as err:
        print(err)

def main():
    # 请求网址
    # urls = ["http://you.ctrip.com/sight/chongqing158/50223-dianping-p%d.html"%x for x in range(1,11)]
    urls = ["http://you.ctrip.com/sight/hangzhou14/49894-dianping-p%d.html"%x for x in range(1,150)]
    urls[0] = urls[0].replace('-p1', '')


    # 利用多线程爬取景点评论
    executor = ThreadPoolExecutor(max_workers=10)  # 可以自己调整max_workers,即线程的个数
    # submit()的参数： 第一个为函数， 之后为该函数的传入参数，允许有多个
    future_tasks = [executor.submit(get_comment, url) for url in urls]
    # 等待所有的线程完成，才进入后续的执行
    wait(future_tasks, return_when=ALL_COMPLETED)

    # 创建DataFrame并保存到csv文件
    comments_table = pd.DataFrame({'id': range(1, len(comments)+1),
                                   'comments': comments})
    print(comments_table)



    data_dir = "./data/comment"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    csv_dir = data_dir+"/hangzhou.csv"
    # dataf = open(data_dir, 'w', newline='')
    # dataf.close()
    comments_table.to_csv(csv_dir, encoding='utf-8-sig', index=False)
    # 读取csv文件
    df = pd.read_csv(csv_dir)['comments']
    # 将pandas中的评论修改后，写入txt文件
    for item in df:
        cmt = item.replace('\n', '').replace('&quot', '').replace(r'&gt;', '').replace(r'！！！', '！') \
            .replace(r' ', '').replace(r'#', '').replace(r'&', '').replace(r'...', '')  \
            .replace(r'＜', '《').replace(r'＞', '》').replace(r'！！', '！')  \
            .replace(r'↑', '').replace(r'[', '').replace(r']', '') \
            .replace(r'❤', '').replace(r'❄', '').replace(r'🌛', '') \
            .replace(r'✈', '').replace(r',,Ծ^Ծ,,', ',') \
            .replace(r'(●176;u176;●)​', '').replace(r'」', ',') \
            .replace(r'。。。', '。').replace(r'，，', '，') \
            .replace(r'🚣', '').replace(r',', '，').replace(r'，。', '') \
            .replace(r'🌧️', '').replace(r'🌞', '').replace(r'。。', '。').replace(r'！！！', '！')  \
            .replace(r'^_^', '').replace(r'☀', '').replace(r'　　', '').replace(r'🌟', '').replace(r'🈷️', '')

        cmt = re.sub('[a-zA-Z]', '', cmt)

        with open(data_dir + "/hangzhou.txt", 'a', encoding='utf-8') as f:
            f.write(cmt)
            f.write('\n')
        # print(comments)
def remove_repeat_char(str1):
    list1 = []
    # print(len(str1))
    max_cnt = 0
    max_char = ''
    if len(str1)>0:
        # 先把第一个字符加到列表中
        list1.append(str1[0])
        cnt = 1
        index = 0
        for i in range(1, len(str1)):
            if str1[i] != list1[index]:
                list1.append(str1[i])
                index += 1
                cnt = 1
            else:
                cnt += 1
                max_cnt = max(cnt,max_cnt)
                max_char = str1[i]
    return max_cnt, max_char

def read_dataset():
    data_dir = "./data/comment"
    # 读取txt文件
    filename = data_dir + "/hangzhou.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read().lower()


    # 创建一个包含整个文本的字符串
    text_string = ''
    for line in raw_text:
        text_string += line
        # text_string += line.strip()

    # 。创建一个字符数组
    text = list()
    for char in text_string:
        text.append(char)
    cnt = {}
    for i in text:
        cnt[i] = cnt.get(i,0) + 1
    sort_cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=False)

    print(sort_cnt)
    text_split = text_string.split('\n')
    print(text_split)
    print(len(text_split))
    tstring = ''
    for txt in text_split:
        flag = True
        max_cnt, max_char = remove_repeat_char(txt)
        if max_cnt>2 and '哈' not in max_char :
            flag = False
            continue
        print(txt)
        for i in cnt:
            if flag and cnt[i]<15 and txt.find(i)>-1:
                print(i)
                flag = False
        if flag:
            tstring += txt.strip()
    text = list()
    for char in tstring:
        text.append(char)
    # 创建文字和对应数字的字典
    chars = sorted(list(set(text)))
    # 对加载数据做总结
    n_chars = len(text)
    n_vocab = len(chars)
    print("总的文字数: ", n_chars)
    print("总的文字类别: ", n_vocab)
    with open(data_dir + "/txthangzhou.txt", 'a', encoding='utf-8') as f:
        f.write(tstring)
        f.write('\n')
    return text, n_vocab

main()
read_dataset()



