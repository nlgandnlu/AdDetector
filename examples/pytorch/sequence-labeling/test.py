# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import pickle
import segeval as seg
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DefaultDataCollator,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_metric
import segeval as seg
metric1 = load_metric("metric/accuracy.py")
metric2 = load_metric("metric/precision.py")
metric3 = load_metric("metric/recall.py")
metric4 = load_metric("metric/f1.py")


import  numpy as np
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class Accuracy:
    def __init__(self, threshold=0.3):
        self.pk_to_weight = []
        self.windiff_to_weight = []
        self.threshold = threshold

    def update(self, h, gold, sentences_length = None):
        h_boundaries = self.get_seg_boundaries(h, sentences_length)
        gold_boundaries = self.get_seg_boundaries(gold, sentences_length)
        pk, count_pk = self.pk(h_boundaries, gold_boundaries)
        windiff, count_wd = -1, 400# self.win_diff(h_boundaries, gold_boundaries)

        if pk != -1:
            self.pk_to_weight.append((pk, count_pk))
        else:
            print ('pk error')

        if windiff != -1:
            self.windiff_to_weight.append((windiff, count_wd))

    def get_seg_boundaries(self, classifications, sentences_length = None):
        """
        :param list of tuples, each tuple is a sentence and its class (1 if it the sentence starts a segment, 0 otherwise).
        e.g: [(this is, 0), (a segment, 1) , (and another one, 1)
        :return: boundaries of segmentation to use for pk method. For given example the function will return (4, 3)
        """
        curr_seg_length = 0
        boundaries = []
        for i, classification in enumerate(classifications):
            is_split_point = bool(classifications[i])
            add_to_current_segment = 1 if sentences_length is None else sentences_length[i]
            curr_seg_length += add_to_current_segment
            if (is_split_point):
                boundaries.append(curr_seg_length)
                curr_seg_length = 0

        return boundaries

    def pk(self, h, gold, window_size=-1):
        """
        :param gold: gold segmentation (item in the list contains the number of words in segment)
        :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
        :param window_size: optional
        :return: accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.pk(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.pk(h, gold, return_parts=True)

        if total_count == 0:
            # TODO: Check when happens
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)

        return false_prob, total_count

    def win_diff(self, h, gold, window_size=-1):
        """
        :param gold: gold segmentation (item in the list contains the number of words in segment)
        :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
        :param window_size: optional
        :return: accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.window_diff(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.window_diff(h, gold, return_parts=True)

        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)

        return false_prob, total_count

    def calc_accuracy(self):
        pk = sum([pw[0] * pw[1] for pw in self.pk_to_weight]) / sum([pw[1] for pw in self.pk_to_weight]) if len(
            self.pk_to_weight) > 0 else -1.0
        windiff = sum([pw[0] * pw[1] for pw in self.windiff_to_weight]) / sum(
            [pw[1] for pw in self.windiff_to_weight]) if len(self.windiff_to_weight) > 0 else -1.0

        return pk, windiff

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    gcn_hidden_size: int = field(
        default=1547, metadata={"help": "gcn_hidden_size"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_concept_file: Optional[str] = field(
        default='train_concept.txt', metadata={"help": "The concept training data file (a text file)."}
    )
    train_index_file: Optional[str] = field(
        default='train_index.txt', metadata={"help": "The concept training index file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    out_info_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    eval_concept_file: Optional[str] = field(
        default='dev_concept.txt', metadata={"help": "The concept eavl data file (a text file)."}
    )
    eval_index_file: Optional[str] = field(
        default='dev_index.txt', metadata={"help": "The concept dev index file (a text file)."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    output_information: bool = field(
        default=False,
        metadata={"help": "Whether output attention figures."},
    )
    con_loss: bool = field(
        default=False,
        metadata={"help": "Whether output attention figures."},
    )
    add_entity_type: bool = field(
        default=False,
        metadata={"help": "Whether use entity type."},
    )
    add_graph_data: bool = field(
        default=False,
        metadata={"help": "Whether use graph data."},
    )

    graph_data_path: Optional[str] = field(
        default='ZHIHU-16K/',
        metadata={"help": "An optional input graph_data_path"},
    )
    eval_label2: bool = field(
        default=False,
        metadata={"help": "Whether eval_label2."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
class LMDataset(torch.utils.data.Dataset):
    def __init__(self, block_size, data_path,graph_data_path, tokenizer: PreTrainedTokenizer, con_loss, entity_type, eval_label2,add_graph_data):

        self.block_size=block_size
        self.data_path=data_path
        self.con_loss=con_loss
        self.add_entity_type=entity_type
        self.add_graph_data=add_graph_data
        self.eval_label2=eval_label2
        self.text_path = data_path+'text/'
        #存放断点label
        self.label1_path=data_path+'label1/'
        # 存放真实label
        self.label2_path = data_path + 'label2/'
        # 存放提前检测好的entity_type
        self.entity_type_path = data_path + 'name_entity_file/'
        #存放graph_data的位置
        self.graph_path = graph_data_path+'edge/'
        self.graph_feature_path =graph_data_path+'feature/'
        self.tokenizer=tokenizer
        self.textfiles = self._get_files()
        # self.memory_mask = list(self._get_mask(memory_mask_path))
        # self.zhutis_mask = list(self._get_mask(zhutis_mask_path))
        self.predefine_dic = {'品牌名': 8, '组织机构类': 24, '组织机构类_企事业单位': 30}

        #self.predefine_dic={'人物类_实体': 0, '物体类': 1, '生物类_动物': 2, '链接地址': 4, '人物类_概念': 6, '品牌名': 8, '作品类_实体': 12, '作品类_概念': 18,'组织机构类': 24, '组织机构类_企事业单位': 30, '世界地区类': 32, '事件类': 33,'组织机构类_医疗卫生机构': 36, '文化类_奖项赛事活动': 37, '饮食类': 38, '时间类': 39, '饮食类_菜品': 44, '时间类_特殊日': 45, '组织机构类_体育组织机构': 48, '饮食类_饮品': 50,  '组织机构类_教育组织机构': 54, '生物类': 55, '药物类': 56, '组织机构类_军事组织机构': 60, '生物类_植物': 61, '药物类_中药': 62}
    def _get_files(self):
        files=[]
        for filename in os.listdir(self.text_path):
            files.append(filename)
        return files
    def clean_paragraph(self, paragraph):
        cleaned_paragraph = paragraph.strip('\n')
        return cleaned_paragraph
    def transfer_label(self,label):
        if (label==1).any():
            label=torch.tensor([1],dtype=torch.long,device=label.device)
        else:
            label=torch.tensor([0],dtype=torch.long,device=label.device)
        return label
    def _get_exmples(self, path1, tokenizer, block_size):

        #logger.info("Creating features from dataset file at %s", path1)

        with open(self.text_path+path1,'r',encoding='utf-8') as f:
            sentence_list = f.readlines()
        if self.eval_label2:
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]
        else:
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]

        # 把每一个句子的类型转换为tensor，总体还是一个list
        token_list = [torch.tensor(x, dtype=torch.long) for x in
                      tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, max_length=block_size)[
                          "input_ids"]]
        type_list=None
        if self.add_entity_type:
            with open(self.entity_type_path+path1,'r',encoding='utf-8') as f:
                entity_type = [[int(y) for y in x.split()] for x in f.readlines()]
            # 计算出每个token对应在原始字符串中的位置，从而能够对应传入entity_type tensor
            position_list = []
            # 用来临时存放token，方便调试观察
            # temp_list=[]
            for sentence_tokens in token_list:
                # token的首尾都是none_type，因为是两个特殊token
                sentence_position_list = [-1, 0]  # 第一个是cls -1，第二个是从0开始的
                # sentence_temp_list=['-1','0']
                for token in sentence_tokens:
                    if token.item() not in [101, 102]:
                        # 上一个token对应在字符串中的位置加上当前token对应字符串的长度就是下一个token对应在字符串中的位置
                        # print(self.text_path+path1)
                        s = tokenizer.convert_ids_to_tokens(token.unsqueeze(0))[0]
                        s = s.replace('##', '')
                        length = len(s)
                        # 如果s是一个特殊token
                        if s.find('[') != -1 and s.find(']') != -1:
                            length = 1
                        sentence_position_list.append(sentence_position_list[-1] + length)
                        # sentence_temp_list.append(s)
                sentence_position_list[-1] = -1
                # sentence_temp_list[-1]='-1'
                position_list.append(sentence_position_list)
                # temp_list.append(sentence_temp_list)
            # 将对应位置的索引转化为对应的entity_type tensor
            type_list = []
            for i in range(len(position_list)):
                sentence_type_list = []
                for j in range(len(position_list[i])):
                    if position_list[i][j] == -1:
                        # 如果位置不在原始序列中，那么类型就是未知
                        sentence_type_list.append(66)
                    else:
                        if entity_type[i][position_list[i][j]] not in self.predefine_dic.values():
                            sentence_type_list.append(66)
                        else:
                            sentence_type_list.append(entity_type[i][position_list[i][j]])
                type_list.append(torch.tensor(sentence_type_list, dtype=torch.long))
        graph=None
        graph_feature=None
        if self.add_graph_data:
            name=path1.replace('.txt','.pkl')
            with open(self.graph_path+name, 'rb') as f: 
                graph = pickle.load(f)
            with open(self.graph_feature_path+name, 'rb') as f: 
                graph_feature = pickle.load(f)
        return token_list, label1, label2, type_list,graph,graph_feature


    def __len__(self):
        return len(self.textfiles)

    def __getitem__(self, i):
        path = self.textfiles[i]
        #print(path)

        token_list,label1,label2, type_list,graph,graph_feature=self._get_exmples(path, self.tokenizer, self.block_size)
        label1=torch.tensor(label1, dtype=torch.long)
        label1=torch.cat([self.transfer_label(label1),label1],dim=0)
        if self.con_loss:
            if self.add_entity_type:
                return [token_list, label1, torch.tensor(label2, dtype=torch.long), type_list, path,None,graph,graph_feature]
            else:
                return [token_list,label1,torch.tensor(label2, dtype=torch.long),None, path,None,graph,graph_feature]
        elif self.add_entity_type:
            return [token_list, label1, None, type_list, path,None,graph,graph_feature]
        else:
            return [token_list, label1, None, None, path,None,graph,graph_feature]

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        # return LineByLineTextDataset(
        #     tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        # )
        return LMDataset(
            tokenizer=tokenizer,  data_path=file_path, graph_data_path=args.graph_data_path,block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2,add_graph_data=args.add_graph_data
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, graph_data_path=args.graph_data_path,block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2,add_graph_data=args.add_graph_data
        )
#删除文件夹中之前残留的文件
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            try:
                os.remove(path_file)
            except:
                pass
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    try:
                        os.remove(path_file2)
                    except:
                        pass
def pk(h,gold):
    h_new=''
    gold_new=''
    for n,x in enumerate(h):
        h_new+=str(int(x))
    for n,x in enumerate(gold):
        gold_new+=str(int(x))
    h_new = seg.convert_nltk_to_masses(h_new)
    gold_new = seg.convert_nltk_to_masses(gold_new)
    dic={}
    pk = seg.pk(h_new, gold_new)
    return pk
yuzhi=0.5
def compute_metrics(eval_pred,all_length):
    global yuzhi
    all_start=[0]
    for i in range(len(all_length)):
        all_start.append(all_start[i]+all_length[i])
    logits, labels = eval_pred
    logits = logits.reshape(-1, 2)
    labels = labels.reshape(-1)
    lines = []
    lines_logits=[]
    # 去掉因为padding补充进来的-100
    for line in range(labels.shape[0]):
        if labels[line] != -100:
            lines.append(line)
    for line in range(logits.shape[0]):
        if logits[line][0] != -100:
            lines_logits.append(line)
    logits = logits[lines_logits]
    labels = labels[lines]
    predictions = np.argmax(logits, axis=-1)
    idx=0
    pre_idx=0
    labels_up = []
    labels_down = []
    predictions_up = []
    predictions_gcn= []
    predictions_down = []

    for i in range(len(all_length)):
        if all_length[i]==0:
            continue
        labels_up.append(labels[idx])
        labels_down = np.append(labels_down,labels[idx+1:idx + all_length[i]])
        predictions_up.append(predictions[pre_idx])
        predictions_gcn.append(predictions[pre_idx+1])
        predictions_down = np.append(predictions_down,predictions[pre_idx+2:pre_idx + all_length[i]+1])
        idx = idx + all_length[i]
        pre_idx= pre_idx + all_length[i] +1
    labels_up = np.array(labels_up)
    labels_down = np.array(labels_down)
    predictions_up = np.array(predictions_up)
    predictions_down = np.array(predictions_down)
    predictions_gcn= np.array(predictions_gcn)
    dic1 = metric1.compute(predictions=predictions_up, references=labels_up)
    dic2 = metric2.compute(predictions=predictions_up, references=labels_up)
    dic3 = metric3.compute(predictions=predictions_up, references=labels_up)
    dic4 = metric4.compute(predictions=predictions_up, references=labels_up)
    dic5 = metric1.compute(predictions=predictions_down, references=labels_down)
    dic5 = {'down_accuracy':dic5['accuracy']}
    dic6 = metric2.compute(predictions=predictions_down, references=labels_down)
    dic6 = {'down_precision':dic6['precision']}
    dic7 = metric3.compute(predictions=predictions_down, references=labels_down)
    dic7 = {'down_recall':dic7['recall']}
    dic8 = metric4.compute(predictions=predictions_down, references=labels_down)
    dic8 = {'down_f1':dic8['f1']}
    dic9 = metric1.compute(predictions=predictions_gcn, references=labels_up)
    dic9 = {'gcn_accuracy':dic9['accuracy']}
    dic10 = metric2.compute(predictions=predictions_gcn, references=labels_up)
    dic10 = {'gcn_precision':dic10['precision']}
    dic11 = metric3.compute(predictions=predictions_gcn, references=labels_up)
    dic11 = {'gcn_recall':dic11['recall']}
    dic12 = metric4.compute(predictions=predictions_gcn, references=labels_up)
    dic12 = {'gcn_f1':dic12['f1']}
    dic1.update(dic2)
    dic1.update(dic3)
    dic1.update(dic4)
    dic1.update(dic5)
    dic1.update(dic6)
    dic1.update(dic7)
    dic1.update(dic8)
    dic1.update(dic9)
    dic1.update(dic10)
    dic1.update(dic11)
    dic1.update(dic12)
    #dic1={'eval_accuracy': 0.9047566249029826, 'eval_precision': 0.0, 'eval_recall': 0.0, 'eval_f1': 0.0}
    return dic1
def delfile(path):
    for f in os.listdir(path):  # os.listdir(path) :返回path指定的文件夹包含的文件或文件夹的名字的列表
        file_path = os.path.join(path, f)  # 拼接文件路径
        os.remove(file_path)  # 删除文件（如果 file_path 是一个文件夹，将抛出OSError）
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.label_num = training_args.label_num
    config.gcn_hidden_size=model_args.gcn_hidden_size
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # Get datasets
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )

    data_collator =DataCollatorWithPadding(tokenizer)

    #
    # # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluation
    folder_path = 'embedding_ad/'
    path1 = 'true/'
    path2 = 'true_result/'
    path3 = 'false/'
    path4 = 'false_result/'

    # 不存在就创建目录
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹
        os.makedirs(folder_path+path1)  # 创建文件夹
        os.makedirs(folder_path+path2)  # 创建文件夹
        os.makedirs(folder_path+path3)  # 创建文件夹
        os.makedirs(folder_path+path4)  # 创建文件夹
    else:  # 否则就删除目录下的文件
        delfile(folder_path+path1)
        delfile(folder_path + path2)
        delfile(folder_path + path3)
        delfile(folder_path + path4)
    logger.info("*** Evaluate ***")
    eval_output = trainer.evaluate(output_figures=data_args.output_information)
    print(eval_output)
    import pandas as pd
    info_path='batch_results/'+data_args.out_info_file
    header = ['name',"accuracy","precision","recall","f1","down_accuracy","down_precision","down_recall","down_f1"]
    outData = pd.DataFrame(columns=header)
    outData.loc[0]=[data_args.eval_data_file,eval_output['eval_accuracy'],eval_output['eval_precision'],eval_output['eval_recall'],eval_output['eval_f1'],
                    eval_output['eval_down_accuracy'],eval_output['eval_down_precision'],eval_output['eval_down_recall'],eval_output['eval_down_f1']]
    #如果之前有其他主题的结果，需要把新结果存在之后
    if os.path.exists(info_path):
        df1 = pd.read_excel(info_path)
        outData=pd.concat([df1,outData])
    else:
        pass
    outData.to_excel(info_path, index=False)

if __name__ == "__main__":
    main()
