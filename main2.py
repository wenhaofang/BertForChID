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

def read_idioms(file_path):
    idioms = []
    with open(file_path, 'r', encoding = 'utf-8') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            for option in data['candidates']:
                idioms.extend(option)
    idioms = list(set(idioms))
    return idioms

idiom_list = read_idioms(train_input)
idiom_dict = { idiom: index for index, idiom in enumerate(idiom_list) }

class InputExample():
    def __init__(self, guid, text, answer, option):
        self.guid = guid
        self.text = text
        self.answer = answer
        self.option = option

class InputFeature():
    def __init__(self, tokens, input_ids, input_mask, segment_ids, answer_pos, option_ids, answer_id):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answer_pos = answer_pos
        self.option_ids = option_ids
        self.answer_id = answer_id

def read_examples(file_path):
    examples = []
    with open(file_path, 'r', encoding = 'utf-8') as jsonl_file:
        for outer_index, line in enumerate(jsonl_file):
            data = json.loads(line)
            content = data['content']
            options = data['candidates']
            answers = data['groundTruth']
            for answer in answers:
                content = content.replace('#idiom#', answer, 1)
            for inner_index, (option, answer) in enumerate(zip(options, answers)):
                guid = str(outer_index + 1) + '_' + str(inner_index + 1)
                text = content.replace(answer, '#idiom#', 1)
                example = InputExample(guid, text, answer, option)
                examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_len):
    features = []
    for example in examples:
        text = example.text
        s_part = tokenizer.tokenize(text.split('#idiom#')[0])
        e_part = tokenizer.tokenize(text.split('#idiom#')[1])
        half_len = max_seq_len // 2
        if len(s_part) >= half_len and len(e_part) >= half_len: # cut at both side
            s_idx = len(s_part) + 3 - half_len
            e_idx = len(s_part) + 1 + half_len
        elif len(s_part) < half_len: # cut at tail
            s_idx = 0
            e_idx = min(len(s_part) + 1 + len(e_part), max_seq_len - 2)
        elif len(e_part) < half_len: # cut at head
            e_idx = len(s_part) + 1 + len(e_part)
            s_idx = max(0, e_idx - (max_seq_len - 2))
        tokens = s_part + ['[MASK]'] + e_part
        tokens = ['[CLS]'] + tokens[s_idx:e_idx] + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        answer_pos = tokens.index('[MASK]')
        option_ids = [idiom_dict[idiom] for idiom in example.option]
        answer_id = option_ids.index(idiom_dict[example.answer])
        feature = InputFeature(tokens, input_ids, input_mask, segment_ids, answer_pos, option_ids, answer_id)
        features.append(feature)
    return features

train_examples = read_examples(train_input)
valid_examples = read_examples(valid_input)
test_examples  = read_examples(test_input )

train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_len)
valid_features = convert_examples_to_features(valid_examples, tokenizer, max_seq_len)
test_features  = convert_examples_to_features(test_examples , tokenizer, max_seq_len)

train_tokens = [f.tokens for f in train_features]
valid_tokens = [f.tokens for f in valid_features]
test_tokens  = [f.tokens for f in test_features ]

train_input_ids_tensor = torch.tensor([f.input_ids for f in train_features], dtype = torch.long)
train_input_mask_tensor = torch.tensor([f.input_mask for f in train_features], dtype = torch.long)
train_segment_ids_tensor = torch.tensor([f.segment_ids for f in train_features], dtype = torch.long)
train_answer_pos_tensor = torch.tensor([f.answer_pos for f in train_features], dtype = torch.long)
train_option_ids_tensor = torch.tensor([f.option_ids for f in train_features], dtype = torch.long)
train_answer_id_tensor = torch.tensor([f.answer_id for f in train_features], dtype = torch.long)

valid_input_ids_tensor = torch.tensor([f.input_ids for f in valid_features], dtype = torch.long)
valid_input_mask_tensor = torch.tensor([f.input_mask for f in valid_features], dtype = torch.long)
valid_segment_ids_tensor = torch.tensor([f.segment_ids for f in valid_features], dtype = torch.long)
valid_answer_pos_tensor = torch.tensor([f.answer_pos for f in valid_features], dtype = torch.long)
valid_option_ids_tensor = torch.tensor([f.option_ids for f in valid_features], dtype = torch.long)
valid_answer_id_tensor = torch.tensor([f.answer_id for f in valid_features], dtype = torch.long)

test_input_ids_tensor = torch.tensor([f.input_ids for f in test_features], dtype = torch.long)
test_input_mask_tensor = torch.tensor([f.input_mask for f in test_features], dtype = torch.long)
test_segment_ids_tensor = torch.tensor([f.segment_ids for f in test_features], dtype = torch.long)
test_answer_pos_tensor = torch.tensor([f.answer_pos for f in test_features], dtype = torch.long)
test_option_ids_tensor = torch.tensor([f.option_ids for f in test_features], dtype = torch.long)
test_answer_id_tensor = torch.tensor([f.answer_id for f in test_features], dtype = torch.long)

train_dataset = torch.utils.data.TensorDataset(train_input_ids_tensor, train_input_mask_tensor, train_segment_ids_tensor, train_answer_pos_tensor, train_option_ids_tensor, train_answer_id_tensor)
valid_dataset = torch.utils.data.TensorDataset(valid_input_ids_tensor, valid_input_mask_tensor, valid_segment_ids_tensor, valid_answer_pos_tensor, valid_option_ids_tensor, valid_answer_id_tensor)
test_dataset  = torch.utils.data.TensorDataset(test_input_ids_tensor , test_input_mask_tensor , test_segment_ids_tensor , test_answer_pos_tensor , test_option_ids_tensor , test_answer_id_tensor )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True )
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_loader  = torch.utils.data.DataLoader(test_dataset , batch_size = batch_size, shuffle = False)

logger.info('Prepare Model')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hidden_size = option.hidden_size
num_choices = option.num_choices

class BertForMultipleChoice(nn.Module):
    def __init__(self, model_name_or_path, hidden_size = 768, num_choices = 7):
        super(BertForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.embedding = nn.Embedding(len(idiom_dict), hidden_size)
        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.bert = BertModel.from_pretrained(model_name_or_path, config = self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, input_mask, segment_ids, answer_pos, option_ids):
        last_hidden_states, _ = self.bert(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        answer_encoded = last_hidden_states[[i for i in range(len(answer_pos))], answer_pos]
        option_encoded = self.embedding(option_ids)
        attention = torch.einsum('abc,ac->abc', option_encoded, answer_encoded)
        output = self.linear(self.dropout(attention))
        output = output.view(-1, self.num_choices) # (batch_size, num_choices)
        return output

model = BertForMultipleChoice(model_name_or_path, hidden_size, num_choices)
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

def score(y_ture, y_pred):
    y_ture = list(y_ture)
    y_pred = list(y_pred)
    min_value = min(y_ture + y_pred)
    max_value = max(y_ture + y_pred)
    diff = max_value - min_value + 1
    confusion_matrix = np.zeros((diff, diff), dtype = np.int)
    for row, col in zip(y_ture, y_pred):
        r_offset = row - min_value
        c_offset = col - min_value
        confusion_matrix[r_offset][c_offset] += 1
    FNs = confusion_matrix.sum(axis = 0)
    FPs = confusion_matrix.sum(axis = 1)
    TPs = confusion_matrix.diagonal()
    P = TPs / (TPs + FPs) # precision
    R = TPs / (TPs + FNs) # recall
    macro_P = sum(P) / len(P)
    macro_R = sum(R) / len(R)
    macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
    return macro_F1, macro_P, macro_R

def save2file(tokens, option_ids, y_ture, y_pred, file_path):
    mapping = dict(zip(idiom_dict.values(), idiom_dict.keys()))
    texts = []
    options = []
    true_choices = []
    pred_choices = []
    for i in range(len(tokens)):
        texts.append(''.join(tokens[i]).lstrip('[CLS]').rstrip('[SEP]'))
        options.append(';'.join([mapping[idiom] for idiom in option_ids[i]]))
        true_choices.append(mapping[option_ids[i][y_ture[i]]])
        pred_choices.append(mapping[option_ids[i][y_pred[i]]])
    with open(file_path, 'w', encoding = 'utf-8') as csv_file:
        csv_file.write('"text","option","true_choice","pred_choice"\n')
        for text, option, true_choice, pred_choice in zip(texts, options, true_choices, pred_choices):
            csv_file.write(
                '"' + text + '"' + ',' +
                '"' + option + '"' + ',' +
                '"' + true_choice + '"' + ',' +
                '"' + pred_choice + '"' + '\n'
            )

best_f1 = 0

patience = 5
early_stop = 0

for epoch in range(1, n_epoch + 1):
    model.train()
    train_loss = 0
    for _, batch in enumerate(train_loader):
        batch = [data.to(device) for data in batch]
        input_ids, input_mask, segment_ids, answer_pos, option_ids, answer_id = batch
        logits = model(input_ids, input_mask, segment_ids, answer_pos, option_ids)
        loss = criterion(logits, answer_id)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    valid_loss = 0
    valid_pred_fold = np.zeros((len(valid_loader.dataset), num_choices))
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch = [data.to(device) for data in batch]
            input_ids, input_mask, segment_ids, answer_pos, option_ids, answer_id = batch
            logits = model(input_ids, input_mask, segment_ids, answer_pos, option_ids)
            loss = criterion(logits, answer_id)
            valid_loss += loss.item()
            valid_pred_fold[i * batch_size: (i + 1) * batch_size] = F.softmax(logits, dim = 1).detach().cpu().numpy()

    f1, precision, recall = score(valid_answer_id_tensor.numpy(), np.argmax(valid_pred_fold, axis = 1))

    if  best_f1 < f1:
        best_f1 = f1
        early_stop = 0
        torch.save(model.state_dict(), trained_model_path)
        save2file(valid_tokens, valid_option_ids_tensor.numpy(), valid_answer_id_tensor.numpy(), np.argmax(valid_pred_fold, axis = 1), sample_file)
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
test_preds_fold = np.zeros((len(test_loader.dataset), num_choices))
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch = [data.to(device) for data in batch]
        input_ids, input_mask, segment_ids, answer_pos, option_ids, answer_id = batch
        logits = model(input_ids, input_mask, segment_ids, answer_pos, option_ids)
        test_preds_fold[i * batch_size: (i + 1) * batch_size] = F.softmax(logits, dim = 1).detach().cpu().numpy()

f1, precision, recall = score(test_answer_id_tensor.numpy(), np.argmax(test_preds_fold, axis = 1))
logger.info('epoch: best, precision: %.8f, recall: %.8f, f1: %.8f' % (precision, recall, f1))

save2file(test_tokens, test_option_ids_tensor.numpy(), test_answer_id_tensor.numpy(), np.argmax(test_preds_fold, axis = 1), result_file)
