import torch
import torch.nn as nn

class TextGenerator(nn.ModuleList):
    def __init__(self, arg, vocab_size):
        super(TextGenerator, self).__init__()
        # self.batch_size = 128
        self.hidden_dim = arg["hidden_dim"]
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = arg["sequence_len"]
        self.num_layers = arg["num_layers"]
        # Dropout
        self.dropout = nn.Dropout(0.25)
        # Embedding 层
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        # LSTM 层
        self.lstm = nn.LSTM(self.hidden_dim , self.hidden_dim, num_layers=self.num_layers)
        # # Bi-LSTM
        # # 正向和反向
        # self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        # self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        # # LSTM 层
        # self.lstm_cell = nn.LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2)
        # Linear 层
        # self.fc1 = nn.Linear(self.hidden_dim * 2,self.hidden_dim * 4)
        # self.fc2 = nn.Linear(self.hidden_dim * 4,self.hidden_dim * 8)
        # self.fc3 = nn.Linear(self.hidden_dim * 8,self.num_classes)
        self.linear = nn.Linear(self.hidden_dim * self.sequence_len , self.num_classes)
    def forward(self, x):
        # # Bi-LSTM
        # # hs = [batch_size x hidden_size]
        # # cs = [batch_size x hidden_size]
        # hs_forward = torch.zeros(x.size(0), self.hidden_dim)
        # cs_forward = torch.zeros(x.size(0), self.hidden_dim)
        # hs_backward = torch.zeros(x.size(0), self.hidden_dim)
        # cs_backward = torch.zeros(x.size(0), self.hidden_dim)
        # # LSTM
        # # hs = [batch_size x (hidden_size * 2)]
        # # cs = [batch_size x (hidden_size * 2)]
        # hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
        # cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
        # # Weights initialization
        # torch.nn.init.kaiming_normal_(hs_forward)
        # torch.nn.init.kaiming_normal_(cs_forward)
        # torch.nn.init.kaiming_normal_(hs_backward)
        # torch.nn.init.kaiming_normal_(cs_backward)
        # torch.nn.init.kaiming_normal_(hs_lstm)
        # torch.nn.init.kaiming_normal_(cs_lstm)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        torch.nn.init.kaiming_normal_(h_0)
        torch.nn.init.kaiming_normal_(c_0)

        # From idx to embedding
        out = self.embedding(x)
        # Prepare the shape for LSTM Cells
        out = out.view(self.sequence_len, x.size(0), -1)
        h_0, c_0 = self.lstm(out, (h_0, c_0))
        # print(h_0.size())
        # forward = []
        # backward = []
        # # Unfolding Bi-LSTM
        # # Forward
        # for i in range(self.sequence_len):
        #     hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
        #     forward.append(hs_forward)
        # # Backward
        # for i in reversed(range(self.sequence_len)):
        #     hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
        #     backward.append(hs_backward)
        # # LSTM
        # for fwd, bwd in zip(forward, backward):
        #     input_tensor = torch.cat((fwd, bwd), 1)
        #     hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
        #     hs_lstm = self.dropout(hs_lstm)
        #     cs_lstm = self.dropout(cs_lstm)

        # output = torch.tanh(self.fc1(hs_lstm))
        # output = torch.tanh(self.fc2(output))
        # out = self.fc3(output)
        out = self.linear(self.dropout(h_0.view(x.size(0), -1)))
        # print(out.size())
        return out

class CommentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CommentModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        # print(input.shape)
        if hidden is None:
            h_0 = input.data.new(256, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(256, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embeddings(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden