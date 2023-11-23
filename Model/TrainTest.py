import torch.nn.functional as F
from models.ChangeDepMulMLP import ChangeDepMulMLP
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, confusion_matrix, accuracy_score, f1_score)
import sys

argvs = sys.argv[1:]
project_name = argvs[0]
dep_num = int(argvs[1])
history_num = int(argvs[2])
approach = argvs[3]

group_num = 1 + dep_num
input_file = 'Dataset/' + project_name + '_500_concatenated.csv'
num_epochs = 400
batch_size = 10000
hidden_dim = 128
one_group = [17, 47, 14]
input_dim = [one_group] * group_num

print('project_name', project_name)
print('approach', approach)
print('dep_num', dep_num)
print('history_num', history_num)
print('input_file', input_file)
print('num_epochs', num_epochs)
print('batch_size', batch_size)
print('hidden_dim', hidden_dim)
print('one_group', one_group)


def split_train_test(all_data, release_to_predict):
    train_set = all_data[all_data['evo_release_index'] < release_to_predict]
    train_set = train_set[train_set['evo_release_index'] > release_to_predict - (history_num + 1)]
    scaler = MinMaxScaler()
    X_features = all_data.columns.tolist()[3:3 + sum(one_group) * group_num]
    X_train = train_set[X_features]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X_features)

    median = train_set['label_evo_file_commit_num'].median()
    y_train = train_set['label_evo_file_commit_num'].apply(lambda x: 1 if x > median else 0)
    test_set = all_data[all_data['evo_release_index'] == release_to_predict]
    X_test = test_set[X_features]
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X_features)
    y_test = test_set['label_evo_file_commit_num'].apply(lambda x: 1 if x > median else 0)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    X_train = torch.FloatTensor(X_train.values)
    y_train = torch.LongTensor(y_train.values)
    X_test = torch.FloatTensor(X_test.values)
    y_test = torch.LongTensor(y_test.values)
    return X_train, X_test, y_train, y_test


all_releases_df = pd.read_csv(input_file)
release_to_predict = all_releases_df['evo_release_index'].max()
X_train, X_test, y_train, y_test = split_train_test(all_releases_df, release_to_predict)

print(y_train.shape[0])
print(y_test.shape[0])

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if approach == 'my':
    model = ChangeDepMulMLP(input_dim, hidden_dim, 2, group_num=group_num)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

if torch.cuda.is_available():
    model = model.cuda()

def test():
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_predict = []
    all_label = []
    all_pre_pros = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.cpu().data, 1)
            pre_pros = F.softmax(outputs, dim=1)[:, 1]
            all_predict += predicted.tolist()
            all_label += labels.tolist()
            all_pre_pros += pre_pros.tolist()

    tn, fp, fn, tp = confusion_matrix(all_label, all_predict).ravel()
    print(tp, fp, tn, fn)
    print(sum(all_label))
    print('precision', precision_score(all_label, all_predict, zero_division=0))
    print('recall', recall_score(all_label, all_predict))
    print('f1', f1_score(all_label, all_predict))
    print('auc', roc_auc_score(all_label, all_predict))
    print('accuracy', accuracy_score(all_label, all_predict))


for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model(inputs)
        loss = criterion(outputs.cpu(), labels)
        loss.backward()
        optimizer.step()


test()
