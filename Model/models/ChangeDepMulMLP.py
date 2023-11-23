import torch
import torch.nn as nn


class ChangeDepMulMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, group_num):
        super(ChangeDepMulMLP, self).__init__()
        self.input_size = input_size
        self.mlps = [
            nn.Sequential(
                nn.Linear(input_size[0][i], hidden_size),
                nn.ReLU(),
            ) for i in range(len(input_size[0]))]

        if torch.cuda.is_available():
            for mlp in self.mlps:
                mlp.cuda()

        self.feature_attention = CustomAttention(hidden_size, 3, 2)
        self.dependency_attention = CustomAttention(hidden_size, group_num, 3)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        dependencies = []
        start = 0
        for group_size in self.input_size:
            output = []
            for subgroup_index in range(len(group_size)):
                subgroup_size = group_size[subgroup_index]
                one_feature = x[:, start: start + subgroup_size]
                one_feature = self.mlps[subgroup_index](one_feature)
                output.append(one_feature)
                start = start + subgroup_size

            dependency = torch.cat([x.unsqueeze(1) for x in output], dim=1)
            dependency = self.feature_attention(dependency)
            dependencies.append(dependency)
        x = torch.cat([x.unsqueeze(1) for x in dependencies], dim=1)
        x = self.dependency_attention(x)
        x = self.final_mlp(x)
        return x


class CustomAttention(nn.Module):
    def __init__(self, input_dim, group_num, multi_head):
        super(CustomAttention, self).__init__()
        self.multi_head = multi_head
        self.W_v = []
        self.weights = []
        self.softmax = []
        self.softmax_ = []
        for _ in range(multi_head):
            self.W_v.append(nn.Linear(input_dim, input_dim))
            self.weights.append(nn.Parameter(torch.randn(group_num, requires_grad=True)))
            self.softmax.append(nn.Softmax(dim=-1))
            self.softmax_.append(nn.ReLU())
        if torch.cuda.is_available():
            for i in range(multi_head):
                self.W_v[i].cuda()
                self.weights[i] = self.weights[i].cuda()
                self.softmax[i].cuda()
                self.softmax_[i].cuda()
        self.final_mlp = nn.Sequential(
            nn.Linear(input_dim * multi_head, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        output = []
        for i in range(self.multi_head):
            v = self.W_v[i](x)
            weights = self.softmax[i](self.weights[i])
            res = torch.sum(v * weights.view(-1, 1), dim=1)
            res = self.softmax_[i](res)
            output.append(res)

        res = torch.cat(output, dim=1)
        res = self.final_mlp(res)
        return res
