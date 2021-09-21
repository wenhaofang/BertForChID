import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertConfig, BertModel

from pytorch_transformers import AdamW

from logger import get_logger
from parser import get_parser

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)
result_folder = os.path.join(root_path, 'result', option.name)

logs_path = option.name + '.log'
save_path = option.name + '.bin'
sample_path = option.name + '.csv'
result_path = option.name + '.csv'

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logger = get_logger(option.name, os.path.join(logs_folder, logs_path))

logger.info('Prepare Data')

batch_size  = option.batch_size
max_seq_len = option.max_seq_len

train_input = option.train_input
valid_input = option.valid_input
test_input  = option.test_input

model_name_or_path = option.model_name_or_path
trained_model_path = os.path.join(save_folder, save_path)

sample_file = os.path.join(sample_folder, sample_path)
result_file = os.path.join(result_folder, result_path)

tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case = True)

class InputExample():
    def __init__(self, guid, source, target):
        self.guid = guid
        self.source = source
        self.target = target

class InputFeature():
    def __init__(self, guid, features, labels):
        self.guid = guid
        tokens, input_ids, input_mask, segment_ids = features
        self.features = {
            'tokens': tokens,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'labels': labels
        }

def read_examples(file_path):
    examples = []
    with open(file_path, 'rt', encoding = 'utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            data = json.loads(line)
            text = data['content']
            idioms = data['groundTruth']
            for idiom in idioms:
                text = text.replace('#idiom#', idiom, 1)
            source = list(text)
            target = ['O'] * len(text)
            for idiom in idioms:
                start_idx = text.index(idiom)
                target[start_idx: start_idx + len(idiom)] = ['I'] * len(idiom)
            example = InputExample(index, source, target)
            examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_len):
    features = []
    mapping = {'O': 0, 'I': 1, 'B': 2}
    for example in examples:
        tokens = []
        labels = []
        source = example.source
        target = example.target
        for s, t in zip(source, target):
            splits = tokenizer.tokenize(s)
            if len(splits) > 0:
                tokens.extend(splits)
                labels.extend([t] + ['O'] * (len(splits) - 1))
        tokens = tokens[:max_seq_len - 1]
        labels = labels[:max_seq_len - 1]
        tokens.insert(0, '[CLS]')
        labels.insert(0, 'O')
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        segment_ids = [0] * len(tokens)
        label_ids = [mapping[label] for label in labels]
        while len(input_ids) < max_seq_len:
            tokens.append('[PAD]')
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        feature = InputFeature(
            example.guid,
            (tokens, input_ids, input_mask, segment_ids),
            label_ids
        )
        features.append(feature)
    return features

def select_field(features, field):
    return [
        feature.features[field] for feature in features
    ]

train_examples = read_examples(train_input)
valid_examples = read_examples(valid_input)
test_examples  = read_examples(test_input )

train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len)
valid_features = convert_examples_to_features(valid_examples, tokenizer, max_seq_len)
test_features  = convert_examples_to_features(test_examples , tokenizer, max_seq_len)

train_tokens = select_field(train_features, 'tokens')
train_input_ids = np.array(select_field(train_features, 'input_ids'))
train_input_mask = np.array(select_field(train_features, 'input_mask'))
train_segment_ids = np.array(select_field(train_features, 'segment_ids'))
train_labels = np.array(select_field(train_features, 'labels'))

valid_tokens = select_field(valid_features, 'tokens')
valid_input_ids = np.array(select_field(valid_features, 'input_ids'))
valid_input_mask = np.array(select_field(valid_features, 'input_mask'))
valid_segment_ids = np.array(select_field(valid_features, 'segment_ids'))
valid_labels = np.array(select_field(valid_features, 'labels'))

test_tokens = select_field(test_features, 'tokens')
test_input_ids = np.array(select_field(test_features, 'input_ids'))
test_input_mask = np.array(select_field(test_features, 'input_mask'))
test_segment_ids = np.array(select_field(test_features, 'segment_ids'))
test_labels = np.array(select_field(test_features, 'labels'))

train_input_ids_tensor = torch.tensor(train_input_ids, dtype = torch.long)
train_input_mask_tensor = torch.tensor(train_input_mask, dtype = torch.long)
train_segment_ids_tensor = torch.tensor(train_segment_ids, dtype = torch.long)
train_label_tensor = torch.tensor(train_labels, dtype = torch.long)

valid_input_ids_tensor = torch.tensor(valid_input_ids, dtype = torch.long)
valid_input_mask_tensor = torch.tensor(valid_input_mask, dtype = torch.long)
valid_segment_ids_tensor = torch.tensor(valid_segment_ids, dtype = torch.long)
valid_label_tensor = torch.tensor(valid_labels, dtype = torch.long)

test_input_ids_tensor = torch.tensor(test_input_ids, dtype = torch.long)
test_input_mask_tensor = torch.tensor(test_input_mask, dtype = torch.long)
test_segment_ids_tensor = torch.tensor(test_segment_ids, dtype = torch.long)
test_label_tensor = torch.tensor(test_labels, dtype = torch.long)

train_dataset = torch.utils.data.TensorDataset(train_input_ids_tensor, train_input_mask_tensor, train_segment_ids_tensor, train_label_tensor)
valid_dataset = torch.utils.data.TensorDataset(valid_input_ids_tensor, valid_input_mask_tensor, valid_segment_ids_tensor, valid_label_tensor)
test_dataset  = torch.utils.data.TensorDataset(test_input_ids_tensor , test_input_mask_tensor , test_segment_ids_tensor , test_label_tensor )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True )
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_loader  = torch.utils.data.DataLoader(test_dataset , batch_size = batch_size, shuffle = False)

logger.info('Prepare Model')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hidden_size = option.hidden_size
num_classes = option.num_classes

class BertForTokenClassification(nn.Module):
    def __init__(self, model_name_or_path, hidden_size = 768, num_classes = 2):
        super(BertForTokenClassification, self).__init__()
        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.bert = BertModel.from_pretrained(model_name_or_path, config = self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, input_mask, segment_ids):
        last_hidden_states, _ = self.bert(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        output = self.linear(self.dropout(last_hidden_states))
        return output

model = BertForTokenClassification(model_name_or_path, hidden_size, num_classes)
model = nn.DataParallel(model)
model = model.to(device)

logger.info('Train & Valid')

n_epoch = option.n_epoch
learning_rate = option.learning_rate

parameters = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
grouped_parameters = [
    {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(grouped_parameters, lr = learning_rate, eps = 1e-6)

def extract(token, pred):
    pred_true = np.where(pred == 1)[0]
    pred_diff = np.diff (pred_true)
    seg_point = np.where(pred_diff != 1)[0] + 1
    seg_group = np.split(list(pred_true), seg_point)
    token = np.array(token)
    idioms = []
    while len(seg_group) > 0:
        group = seg_group.pop()
        if len(group) == 4:
            idom = token[group]
            idioms.append(idom)
        elif len(group) > 4:
            seg_group.append(group[:4])
            seg_group.append(group[4:])
        elif len(group) < 4:
            continue
    idioms.reverse()
    idioms = [''.join(idiom) for idiom in idioms]
    return idioms

def score(tokens, y_true, y_pred):
    A, B, C = 1e-10, 1e-10, 1e-10
    for i in range(len(tokens)):
        R = set(extract(tokens[i], y_pred[i]))
        T = set(extract(tokens[i], y_true[i]))
        A += len(R & T)
        B += len(R)
        C += len(T)
    precision, recall, f1 = A / B, A / C, 2 * A / (B + C)
    return f1, precision, recall

def save2file(tokens, y_true, y_pred, file_path):
    texts = []
    true_idioms = []
    pred_idioms = []
    for i in range(len(tokens)):
        texts.append(''.join(tokens[i]).lstrip('[CLS]').rstrip('[PAD]*'))
        true_idioms.append(';'.join(extract(tokens[i], y_true[i])))
        pred_idioms.append(';'.join(extract(tokens[i], y_pred[i])))
    with open(file_path, 'w', encoding = 'utf-8') as csv_file:
        csv_file.write('"text","true_idiom","pred_idiom"\n')
        for text, true_idiom, pred_idiom in zip(texts, true_idioms, pred_idioms):
            csv_file.write(
                '"' + text + '"' + ',' +
                '"' + true_idiom + '"' + ',' +
                '"' + pred_idiom + '"' + '\n'
            )

best_f1 = 0

patience = 5
early_stop = 0

for epoch in range(1, n_epoch + 1):
    model.train()
    train_loss = 0
    for _, batch in enumerate(train_loader):
        batch = [data.to(device) for data in batch]
        x_ids, x_mask, x_seg_ids, y_true = batch
        y_pred = model(x_ids, x_mask, x_seg_ids)
        length = torch.sum(x_mask, dim = 1).cpu()
        y_pred_pad = pack_padded_sequence(y_pred, length, batch_first = True, enforce_sorted = False).data
        y_true_pad = pack_padded_sequence(y_true, length, batch_first = True, enforce_sorted = False).data
        loss = criterion(y_pred_pad, y_true_pad)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    valid_loss = 0
    valid_pred_fold = np.zeros((valid_labels.shape[0], valid_labels.shape[1], num_classes))
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch = [data.to(device) for data in batch]
            x_ids, x_mask, x_seg_ids, y_true = batch
            y_pred = model(x_ids, x_mask, x_seg_ids)
            length = torch.sum(x_mask, dim = 1).cpu()
            y_pred_pad = pack_padded_sequence(y_pred, length, batch_first = True, enforce_sorted = False).data
            y_true_pad = pack_padded_sequence(y_true, length, batch_first = True, enforce_sorted = False).data
            loss = criterion(y_pred_pad, y_true_pad)
            valid_loss += loss.item()
            valid_pred_fold[i * batch_size: (i + 1) * batch_size] = F.softmax(y_pred, dim = 2).detach().cpu().numpy()

    f1, precision, recall = score(valid_tokens, valid_labels, np.argmax(valid_pred_fold, axis = 2))

    if  best_f1 < f1:
        best_f1 = f1
        early_stop = 0
        torch.save(model.state_dict(), trained_model_path)
        save2file(valid_tokens, valid_labels, np.argmax(valid_pred_fold, axis = 2), sample_file)
    else:
        early_stop += 1

    logger.info(
        'epoch: %d, train_loss: %.8f, valid_loss: %.8f, precision: %.8f, recall: %.8f, f1: %.8f, best_f1: %.8f' %
        (epoch, train_loss / len(train_loader), valid_loss / len(valid_loader), precision, recall, f1, best_f1)
    )

    torch.cuda.empty_cache()

    if early_stop > patience:
        break

model.load_state_dict(torch.load(trained_model_path))

model.eval()
test_preds_fold = np.zeros((test_labels.shape[0], test_labels.shape[1], num_classes))
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch = [data.to(device) for data in batch]
        x_ids, x_mask, x_seg_ids, y_true = batch
        y_pred = model(x_ids, x_mask, x_seg_ids)
        test_preds_fold[i * batch_size: (i + 1) * batch_size] = F.softmax(y_pred, dim = 2).detach().cpu().numpy()

f1, precision, recall = score(test_tokens, test_labels, np.argmax(test_preds_fold, axis = 2))
logger.info('epoch: best, precision: %.8f, recall: %.8f, f1: %.8f' % (precision, recall, f1))

save2file(test_tokens, test_labels, np.argmax(test_preds_fold, axis = 2), result_file)
