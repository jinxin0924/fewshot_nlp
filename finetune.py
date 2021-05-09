import torch.utils.data
from transformers import trainer_utils
from transformers import BertModel, BertForSequenceClassification, TrainingArguments, Trainer, AlbertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np


batch_size = 8

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# device = 'cpu'
device = torch.device('cuda')

from transformers import BertTokenizer

pretrained = 'voidful/albert_chinese_tiny'
# pretrained = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained)


def load_dataset(file_name, label_encoder=None):
    def prepare_data(file_name):
        data = []
        for line in open(file_name, 'r', encoding='utf-8'):
            data.append(json.loads(line))
        return data

    data_list = prepare_data(file_name)
    text = [data['sentence'] for data in data_list]
    label = [data['label'] for data in data_list]
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']

    if label_encoder:
        label_encoder = label_encoder
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(label)
    label = label_encoder.transform(label)
    labels = torch.tensor(label).long()
    return TensorDataset(input_ids, token_type_ids, attention_mask, labels), label_encoder


train_dataset, label_encoder = load_dataset(file_name='G:/FewCLUE/datasets/cecmmnt/train_few_all.json')

val_dataset, _ = load_dataset(file_name='G:/FewCLUE/datasets/cecmmnt/dev_1.json', label_encoder=label_encoder)

training_args = TrainingArguments(
    output_dir='output/',
    num_train_epochs=10,
    per_gpu_train_batch_size=16,
    per_gpu_eval_batch_size=16,
    warmup_steps=0,
    weight_decay=0.1,
    save_steps=20,
    overwrite_output_dir=True,
    label_names=['labels']
)

epochs = 4
batch_size = 8

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Load the pretrained BERT model
model = AlbertForSequenceClassification.from_pretrained(pretrained, num_labels=2)
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2, output_attentions=False,
#                                                       output_hidden_states=False)
# model.cuda()

# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

from sklearn.metrics import f1_score, accuracy_score

import numpy as np


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


model = model.to(device)
for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    print('epoch:', epoch, ', step_number:', len(train_dataloader))
    # 训练
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        loss, logits = model(input_ids=batch[0].to(device),
                             token_type_ids=batch[1].to(device),
                             attention_mask=batch[2].to(device),
                             labels=batch[3].to(device),
                             return_dict=False
                             )  # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 and step > 0:  # 每10步输出一下训练的结果，flat_accuracy()会对logits进行softmax
            model.eval()
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].to(device).data.cpu().numpy()
            avg_val_accuracy = flat_accuracy(logits, label_ids)
            print('step:', step)
            print(f'Accuracy: {avg_val_accuracy:.4f}')
            print('\n')
    # 每个epoch结束，就使用validation数据集评估一次模型
    model.eval()
    print('testing ....')
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            loss, logits = model(input_ids=batch[0].to(device),
                                 token_type_ids=batch[1].to(device),
                                 attention_mask=batch[2].to(device),
                                 labels=batch[3].to(device),
                                 return_dict=False
                                 )
            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cuda().data.cpu().numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

    print(f'Train loss     : {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Accuracy: {avg_val_accuracy:.4f}')
    print('\n')
    # save_model(self.saveModel_path + '-' + str(epoch))
