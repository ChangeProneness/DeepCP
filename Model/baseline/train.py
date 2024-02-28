import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import torch.optim.lr_scheduler
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from model import Net
import deal_data
from pytorchtools import EarlyStopping
import sys

args = sys.argv

project_name = args[1]
epoch_sum = 100
deal_data.setup_seed(20)

all_data = pd.read_csv('Dataset/' + project_name + '_500_concatenated.csv')
release_to_predict = all_data['evo_release_index'].max()

train_set = all_data[all_data['evo_release_index'] < release_to_predict]
train_set = train_set[train_set['evo_release_index'] > release_to_predict - 2]

X_features = all_data.columns.tolist()[3:3 + 78]
X_train = train_set[X_features]
X_train = pd.DataFrame(X_train, columns=X_features)

median = train_set['label_evo_file_commit_num'].median()
y_train = train_set['label_evo_file_commit_num'].apply(lambda x: 1 if x > median else 0)
test_set = all_data[all_data['evo_release_index'] == release_to_predict]
X_test = test_set[X_features]
X_test = pd.DataFrame(X_test, columns=X_features)
y_test = test_set['label_evo_file_commit_num'].apply(lambda x: 1 if x > median else 0)

samples = SMOTE()
X_train, y_train = samples.fit_resample(X_train, y_train)
X_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.LongTensor(y_test.values)
train_dataset = TensorDataset(X_train, y_train)
trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
testloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

model = Net(len=78)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)
early_stopping = EarlyStopping(patience=15, verbose=True)
train_losses = []
for epoch in range(epoch_sum):
    model.train()
    loss_sum = 0.0
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = np.average(train_losses)

    epoch_len = len(str(epoch_sum))

    train_losses = []

    early_stopping(train_loss, model)
    if early_stopping.early_stop:
        break

model.load_state_dict(torch.load('checkpoint.pt'))

model.eval()

all_predict = []
all_label = []
all_pre_pros = []
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        output = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        pre_pros = F.softmax(outputs, dim=1)[:, 1]
        all_predict += predicted.tolist()
        all_label += labels.tolist()
        all_pre_pros += pre_pros.tolist()
tn, fp, fn, tp = confusion_matrix(all_label, all_predict).ravel()
print(project_name,
      roc_auc_score(all_label, all_predict),
      f1_score(all_label, all_predict),
      precision_score(all_label, all_predict, zero_division=0),
      recall_score(all_label, all_predict),
      accuracy_score(all_label, all_predict),
      tp,
      fp,
      tn,
      fn,
      sum(all_label))
