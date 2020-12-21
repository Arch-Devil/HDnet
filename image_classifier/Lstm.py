# 搭建简单的LSTM模型用于生成旅游评论
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from qdnet.models.CommentModel import CommentModel
from qdnet.models.CommentModel import TextGenerator
import torch.nn.functional as F
import os

def read_dataset():
    data_dir = "./data/comment"
    # 读取txt文件
    filename = data_dir + "/txthangzhou.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read().lower()

    # 创建文字和对应数字的字典
    chars = sorted(list(set(raw_text)))
    # print(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # 对加载数据做总结
    n_chars = len(raw_text)
    n_vocab = len(chars)
    # 创建一个包含整个文本的字符串
    text_string = ''
    for line in raw_text:
        text_string += line
        # text_string += line.strip()

    # 。创建一个字符数组
    text = list()
    for char in text_string:
        text.append(char)
    print("总的文字数: ", n_chars)
    print("总的文字类别: ", n_vocab)

    # cnt = {}
    # for i in text:
    #     cnt[i] = cnt.get(i,0) + 1
    # # cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=False)
    #
    # print(cnt)
    # text_split = text_string.split('\n')
    # print(text_split)
    # print(len(text_split))
    # tstring = ''
    # for txt in text_split:
    #     print(txt)
    #     flag = True
    #     for i in cnt:
    #         if flag and cnt[i]==1 and txt.find(i)>-1:
    #             print(i)
    #             flag = False
    #     if flag:
    #         tstring += txt.strip()
    # text = list()
    # for char in tstring:
    #     text.append(char)
    # print(len(text))
    # with open(data_dir + "/txthangzhou.txt", 'a', encoding='utf-8') as f:
    #     f.write(tstring)
    #     f.write('\n')


    return text, n_vocab


def create_dictionary(text):
    char_to_idx = dict()
    idx_to_char = dict()
    idx = 0
    for char in text:
        if char not in char_to_idx.keys():
            # 构建字典
            char_to_idx[char] = idx
            idx_to_char[idx] = char
            idx += 1
    return char_to_idx, idx_to_char

def build_sequences(text, char_to_idx, window):
    x = list()
    y = list()
    for i in range(len(text)):
        try:
            # 从文本中获取字符窗口
            # 将其转换为其idx表示
            sequence = text[i:i+window]
            sequence = [char_to_idx[char] for char in sequence]
            #得到target
            # 转换到它的idx表示
            target = text[i+window]
            target = char_to_idx[target]
            # 保存sequence和target
            x.append(sequence)
            y.append(target)
        except:
            pass
    x = np.array(x)
    y = np.array(y)
    return x, y

def text_train(arg):
    num_window = arg["num_window"]
    batch_size = arg["batch_size"]
    num_epochs = arg["num_epochs"]
    min_loss = arg["min_loss"]
    data_dir = arg["model_dir"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    text, n_vocab = read_dataset()
    char_to_idx, idx_to_char = create_dictionary(text)
    sequences, targets = build_sequences(text, char_to_idx, window=num_window)
    # # 正则化
    # sequences = sequences / n_vocab
    vocab_size = len(char_to_idx)
    print(vocab_size)
    # 模型初始化
    model = TextGenerator(arg, vocab_size)
    if arg["pretrained"]:
        model.load_state_dict(torch.load(data_dir+'/best_model.pt'))
    criterion = nn.CrossEntropyLoss()
    # 优化器初始化
    optimizer = optim.Adam(model.parameters(), lr=float(arg["lr"]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.66, patience=2, verbose=False)
    # 定义batch数
    num_batches = int(len(sequences) / batch_size)
    # 训练模型
    model.train()
    # 训练阶段
    for epoch in range(num_epochs):
        # Mini batches
        avg_loss = 0
        for i in range(num_batches):
            # Batch 定义
            try:
                x_batch = sequences[i * batch_size : (i + 1) * batch_size]
                y_batch = targets[i * batch_size : (i + 1) * batch_size]
            except:
                x_batch = sequences[i * batch_size :]
                y_batch = targets[i * batch_size :]
            # 转换 numpy array 为 torch tensors
            x = torch.from_numpy(x_batch).type(torch.LongTensor)
            y = torch.from_numpy(y_batch).type(torch.LongTensor)
            # 输入数据
            y_pred = model(x)
            # loss计算
            loss = criterion(y_pred, y.squeeze())
            # 清除梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            avg_loss+=loss.item()
            if (i)%10 ==0:
                print("Epoch: %d ,  loss: %.5f " % (epoch, loss.item()))
        avg_loss/=num_batches
        print("Epoch: %d ,  avg_loss: %.5f, lr: %.8f" % (epoch, avg_loss, optimizer.param_groups[0]["lr"]))
        if avg_loss < min_loss:
            min_loss = avg_loss
            print('saved!')
            torch.save(model.state_dict(), data_dir + '/best_model.pt')
        if (epoch+1)  % 1 == 0 :
            generator(arg)
            torch.save(model.state_dict(), data_dir + '/{}_model.pt'.format((epoch+1)))
        torch.save(model.state_dict(), data_dir + '/latest_model.pt')
        lr_scheduler.step(avg_loss)


def generator(arg):
    num_window = arg["num_window"]
    n_chars = arg["n_chars"]
    data_dir = arg["model_dir"]
    text, n_vocab = read_dataset()
    char_to_idx, idx_to_char = create_dictionary(text)
    sequences, targets = build_sequences(text, char_to_idx, window=num_window)

    vocab_size = len(char_to_idx)
    model = TextGenerator(arg, vocab_size)
    model.load_state_dict(torch.load(data_dir+'/best_model.pt'))
    model.eval()

    # Define the softmax function
    softmax = nn.Softmax(dim=1)

    # 示例的输入句子
    input = '杭州西湖天下闻名，西'
    pattern = [char_to_idx[value] for value in input]
    pattern = np.array(pattern)

    print("输入：")
    print(''.join([idx_to_char[value] for value in pattern]))

    # 在full_prediction中，我们将保存完整的预测
    full_prediction = pattern.copy()

    # 预测开始，它将被预测为一个给定的字符长度
    for i in range(n_chars):
        # 转换为tensor
        pattern = torch.from_numpy(pattern).type(torch.LongTensor)
        pattern = pattern.view(1, -1)

        # 预测
        prediction = model(pattern)
        # 将softmax函数应用于预测张量
        prediction = softmax(prediction)

        # 预测张量被转换成一个numpy数组
        prediction = prediction.squeeze().detach().numpy()
        # 取概率最大的idx
        arg_max = np.argmax(prediction)

        # 将当前张量转换为numpy数组
        pattern = pattern.squeeze().detach().numpy()
        # 窗口向右1个字符
        pattern = pattern[1:]
        # 新pattern是由“旧”pattern+预测的字符组成的
        pattern = np.append(pattern, arg_max)

        # 保存完整的预测
        full_prediction = np.append(full_prediction, arg_max)


    print("输出: ")
    print(''.join([idx_to_char[int(value * 1)] for value in full_prediction]), "\"")

