#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import re
import random
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional

# import wandb
import numpy as np
from datasets import load_dataset, load_metric
from utils import EXTERNAL_TOKENS, MAX_NUM_EVENTS, MAX_NUM_ARGUMENTS


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertTokenizer,
    RobertaTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import json
import torch
import collections
from tqdm import tqdm
from my_model import MyBertmodel
from trainer import Mytrainer

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    span_len: int = field(default=8)
    max_len: int = field(default=512)
    meta_file: str = field(default="")
    task_name: str = field(default="")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    pos_loss_weight: float = field(default=1.0)
    span_len_embedding_range: int = field(default=50)
    span_len_embedding_hidden_size: int = field(default=20)
    not_bert_learning_rate: float = field(default=0.0001)
    lambda_boundary: float = field(default=0.0)
    event_embedding_size: int = field(default=200)


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=sys.argv[1]
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check task
    assert data_args.task_name in [
        # 'rams',
        "wikievent"
    ]
    assert (
        data_args.train_file is not None
        and data_args.validation_file is not None
        and data_args.test_file is not None
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        logger.info("hi")
    # Check cuda setting
    logger.info(torch.cuda.is_available())
    logger.info(torch.version)
    logger.info(torch.version.cuda)
    logger.info(torch.backends.cudnn.enabled)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_dataset(
        "text",
        data_files={
            "train": data_args.train_file,
            "validation": data_args.validation_file,
            "test": data_args.test_file,
        },
    )
    # ======== construct meta schema ===========
    with open(data_args.meta_file) as f:
        meta = json.load(f)
    event2id = {}
    id2event = {}
    eventid2role2id = {}
    eventid2id2role = {}
    role_id = 1
    num_labels = 1
    for i, d in enumerate(meta):
        event2id[d[0]] = i
        id2event[i] = d[0]
        roles = d[1]
        eventid2role2id[i] = {}
        eventid2id2role[i] = collections.OrderedDict()
        for role in roles:
            eventid2role2id[i][role] = role_id
            eventid2id2role[i][role_id] = role
            role_id += 1
            num_labels += 1
    event_num = len(event2id)
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # assert data_args.max_len <= config.max_position_embeddings
    # ======== make some additional setting ==============
    setattr(config, "pos_loss_weight", model_args.pos_loss_weight)
    setattr(config, "len_size", model_args.span_len_embedding_range)
    setattr(config, "len_dim", model_args.span_len_embedding_hidden_size)
    setattr(config, "event_num", event_num)
    logger.info(f"model_args: {model_args}")
    logger.info(f"event_num: {event_num}")

    # ======== make some additional setting ==============

    if model_args.model_name_or_path.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        new_token_list = copy.deepcopy(EXTERNAL_TOKENS)

        # r = tokenizer.add_tokens(new_token_list)

        model = MyBertmodel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            lambda_boundary=model_args.lambda_boundary,
            event_embedding_size=model_args.event_embedding_size,
        )
        TRIGGER_LEFT = 5  # special token for trigger
        TRIGGER_RIGHT = 6
        EVENT_START = 104  # special token for event
        ROLE_START = 400  # special token for roles
        # TODO implement roberta
    # elif model_args.model_name_or_path.startswith("roberta"):
    #     tokenizer = RobertaTokenizer.from_pretrained(
    #         (
    #             model_args.tokenizer_name
    #             if model_args.tokenizer_name
    #             else model_args.model_name_or_path
    #         ),
    #         cache_dir=model_args.cache_dir,
    #         use_fast=model_args.use_fast_tokenizer,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    #     model = MyRobertamodel.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         lambda_boundary=model_args.lambda_boundary,
    #         event_embedding_size=model_args.event_embedding_size,
    #     )
    #     TRIGGER_LEFT = tokenizer("[")["input_ids"][1]
    #     TRIGGER_RIGHT = tokenizer("]")["input_ids"][1]
    #     ROLE_START = len(tokenizer) + len(event2id)
    #     EVENT_START = len(tokenizer)
    #     event_nums = len(event2id)
    #     role_nums = num_labels
    #     special_tokens_dict = {"additional_special_tokens": []}
    #     for i_e in range(event_nums):
    #         special_tokens_dict["additional_special_tokens"].append(
    #             "[unused" + str(EVENT_START + i_e) + "]"
    #         )
    #     for i_r in range(role_nums):
    #         special_tokens_dict["additional_special_tokens"].append(
    #             "[unused" + str(ROLE_START + i_r) + "]"
    #         )
    #     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #     model.resize_token_embeddings(len(tokenizer))  # update the embedding
    else:
        assert False

    def preprocess_function(example, idx, split):
        example = json.loads(example["text"])
        doc_id = example["doc_id"]
        sentences = [token[0] for item in example["sentences"] for token in item[0]]
        snt2span = []
        start, end = 0, 0
        for idx, sen in enumerate(sentences):
            end = start + len(sen[0]) - 1
            snt2span.append([start, end])
            start = start + len(sen[0])

        def which_snt(snt2span, span, original):
            for snt in range(len(snt2span)):
                snt_spans = snt2span[snt]
                if span[0] >= snt_spans[0] and span[1] <= snt_spans[1]:
                    return snt

            assert False

        triggers_begin = []
        triggers_end = []
        events = []
        events_id = []
        triggers_snt_id = []

        for i, (event_mention) in enumerate(example["event_mentions"]):

            # by event
            # logger.info(f"event_mention: {event_mention}")
            trigger_begin, trigger_end, event = (
                event_mention["trigger"]["start"],
                event_mention["trigger"]["end"] - 1,
                event_mention["event_type"],
            )
            # by event
            trigger_snt_id = event_mention["trigger"]["sent_idx"]
            # by event
            eventid = event2id[event]

            triggers_begin.append(trigger_begin)
            triggers_end.append(trigger_end)
            events.append(event)
            events_id.append(eventid)
            triggers_snt_id.append(trigger_snt_id)

        now_snt_idx = 0
        input_ids = [tokenizer.cls_token_id]
        subwords_snt2span = []
        wordidx2subwordidx = []
        exclude_words = []
        if data_args.task_name == "wikievent":
            exclude_symbols = [
                ",",
                "!",
                "?",
                ":",
            ]  # We select some normal symols that can not appear in the middle of a argument span. For different datasets, you can choose different symbols.
        else:
            exclude_symbols = [",", ".", "!", "?", ":"]

        current_event_cnt = 0
        for i, sentence in enumerate(sentences):
            subwords_snt2span_st = len(input_ids)
            for j, word in enumerate(sentence):
                if now_snt_idx in triggers_begin:
                    trig_sub_s = len(input_ids)
                    exclude_words.append(trig_sub_s)
                    input_ids.extend(
                        tokenizer(
                            f"<t-{current_event_cnt}>",
                            add_special_tokens=False,
                            return_attention_mask=False,
                        )["input_ids"]
                    )  # Special token
                if now_snt_idx in triggers_end:
                    trig_sub_e = len(input_ids)
                    exclude_words.append(trig_sub_e)
                    input_ids.extend(
                        tokenizer(
                            f"</t-{current_event_cnt}>",
                            add_special_tokens=False,
                            return_attention_mask=False,
                        )["input_ids"]
                    )

                subwords_ids = tokenizer(
                    word, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]

                if word in exclude_symbols:
                    exclude_idx = []
                    for kk in range(len(subwords_ids)):
                        exclude_idx.append(len(input_ids) + kk)
                    exclude_words.extend(exclude_idx)
                wordidx2subwordidx.append(
                    (len(input_ids), len(input_ids) + len(subwords_ids) - 1)
                )
                input_ids.extend(subwords_ids)
                now_snt_idx += 1
            subwords_snt2span.append([subwords_snt2span_st, len(input_ids) - 1])

        model_max_len = 768
        max_role_token_len = 36

        spans = []
        span_lens = []
        subwords_span2snt = []

        labels_mask = []
        # spans_labels = []

        span_labels = []
        for i, (trigger_b, trigger_e, event, trigger_snt_id, event_id) in enumerate(
            zip(triggers_begin, triggers_end, events, triggers_snt_id, events_id)
        ):
            label_mask = [0] * num_labels
            label_mask[0] = 1
            label_mask = np.array(label_mask)

            for argument in example["event_mentions"][i]["arguments"]:
                # logger.info(f"argument: {argument}")
                role_begin, role_end = argument["start"], argument["end"] - 1
                role = argument["role"]
                if role not in eventid2role2id[event_id]:
                    continue

                roleid = eventid2role2id[event_id][role]
                base_roleid = list(eventid2id2role[event_id].keys())[0]
                upper_roleid = list(eventid2id2role[event_id].keys())[-1]
                label_mask[base_roleid : upper_roleid + 1] = 1
                role_subword_start_idx = wordidx2subwordidx[role_begin][0]

                role_subword_end_idx = wordidx2subwordidx[role_end][-1]
                if role_subword_end_idx < model_max_len:
                    spans.append([role_subword_start_idx, role_subword_end_idx])
                    # MEMO: 오류. 그런데 snt 구하는 부분인데 word랑은 무관해야 함
                    # subwords_span2snt.append(
                    #     which_snt(subwords_snt2span, spans[-1], argument)
                    # )
                    span_lens.append(
                        min(
                            role_subword_end_idx - role_subword_start_idx,
                            config.len_size - 1,
                        )
                    )
                    span_labels.append(roleid)

            labels_mask.append(label_mask)
            # spans_labels.append(span_labels)

        role_nums = 0
        for label_mask in labels_mask:
            role_nums += label_mask.sum() - 1

        if len(input_ids) > model_max_len - 1:
            input_ids = input_ids[
                : model_max_len - (len(event)) * 6 - max_role_token_len - role_nums * 6
            ]

        info_dict = {
            "event_ids": [],
            "event_idx": [],
            "role_idxs": [],
            "words_num": 0,
        }
        ari_len = len(input_ids)
        info_dict["words_num"] = ari_len + 1

        argument_cnt = 0
        roles_idx = []
        for i, (trigger_b, trigger_e, event, trigger_snt_id, event_id) in enumerate(
            zip(triggers_begin, triggers_end, events, triggers_snt_id, events_id)
        ):
            role_list = list(eventid2id2role[event_id].values())
            role_id_list = list(eventid2id2role[event_id].keys())
            if data_args.task_name == "rams":
                for itt in range(len(role_list)):
                    role_list[itt] = role_list[itt].split("arg")[-1][2:]

            input_ids.append(tokenizer.sep_token_id)

            event_split_list = event.split(".")
            event_tok = EVENT_START + event_id

            input_ids.extend(
                tokenizer(
                    f"<e-{i}>",
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            )

            info_dict["event_idx"].append(len(input_ids) - 1)
            info_dict["event_ids"].append(role_id_list)

            for item in event_split_list:
                event_subwords_ids = tokenizer(
                    item, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                input_ids.extend(event_subwords_ids)

            input_ids.extend(
                tokenizer(
                    f"</e-{i}>",
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            )

            role_idx = []
            for rr, role_t in enumerate(role_list):
                if argument_cnt > MAX_NUM_ARGUMENTS:
                    logger.warning(
                        f"Error: Max_Num_ARGUMENT exceed. argument_cnt: {argument_cnt}:"
                    )
                    break
                role_subwords_ids = tokenizer(
                    role_t, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                input_ids.extend(
                    tokenizer(
                        f"<r-{argument_cnt}>",
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                )
                # input_ids.append(ROLE_START + role_id_list[rr])
                info_dict["role_idxs"].append(len(input_ids) - 1)
                input_ids.extend(role_subwords_ids)

                input_ids.extend(
                    tokenizer(
                        f"</r-{argument_cnt}>",
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                )
                # input_ids.append(ROLE_START + role_id_list[rr])
                argument_cnt += 1
            role_idx.append(len(input_ids) - 1)
            roles_idx.append(role_idx)
        input_ids.append(tokenizer.sep_token_id)
        info_dict["role_idxs"] = roles_idx

        triggers_index = []
        starts_label = []
        ends_label = []

        for i, (trigger_b, trigger_e, event, trigger_snt_id, event_id) in enumerate(
            zip(triggers_begin, triggers_end, events, triggers_snt_id, events_id)
        ):
            trigger_index = wordidx2subwordidx[trigger_b][0] - 1
            trigger_index = min(trigger_index, len(input_ids) - 1)
            triggers_index.append(trigger_index)

        start_label = [0 for _ in range(len(input_ids))]
        end_label = [0 for _ in range(len(input_ids))]
        words_num = info_dict["words_num"]
        for start_end_span in spans:
            start, end = start_end_span
            if (
                (start > words_num)
                or (end > words_num)
                or (start > len(input_ids))
                or (end > len(input_ids))
            ):
                print(start, end, len(input_ids), words_num)

            if start < len(input_ids) and end < len(input_ids):
                start_label[start] = 1
                end_label[end] = 1

        # construct negative examples
        all_non_spans = []
        for i in range(len(sentences)):
            start_idx, end_idx = subwords_snt2span[i]
            end_idx = min(end_idx, model_max_len - 1)
            for s in range(start_idx, end_idx + 1):
                for e in range(s, end_idx + 1):
                    if s >= words_num:
                        continue
                    if e >= words_num:
                        continue
                    flag = 0
                    if e - s + 1 <= data_args.span_len:
                        for kkk in range(s, e + 1):
                            if kkk in exclude_words:
                                flag = 1
                                break
                        if [s, e] not in spans and flag == 0:
                            all_non_spans.append([s, e])
                            subwords_span2snt.append(i)
        spans.extend(all_non_spans)
        span_lens.extend([x[1] - x[0] for x in all_non_spans])
        span_labels.extend([0] * len(all_non_spans))
        span_num = len(spans)
        result = {
            "idx": idx,
            "split": split,
            "input_ids": input_ids,
            "label": span_labels,
            "spans": spans,
            "event_id": eventid,
            "span_lens": span_lens,
            "label_mask": labels_mask,
            "trigger_index": triggers_index,
            "span_num": span_num,
            "subwords_span2snt": subwords_span2snt,
            "subwords_snt2span": subwords_snt2span,
            "trigger_snt_id": trigger_snt_id,
            "info_dict": info_dict,
            "snt2span": snt2span,
            "wordidx2subwordidx": wordidx2subwordidx,
            "start_label": start_label,
            "end_label": end_label,
        }

        if len(input_ids) > 1024:
            logger.warning(
                f"Input_ids Exceed maximum context: {doc_id}, len: {len(input_ids)}"
            )

        if idx == 0:
            # write result as json
            with open("new_after_preprocess.json", "w") as f:
                json.dump(result, f, indent=4)

        return result

    # train_dataset = datasets["train"].map(
    #     preprocess_function,
    #     batched=False,
    #     with_indices=True,
    #     load_from_cache_file=not data_args.overwrite_cache,
    #     fn_kwargs={"split": "train"},
    #     num_proc=6,
    # )

    eval_dataset = datasets["validation"].map(
        preprocess_function,
        batched=False,
        with_indices=True,
        load_from_cache_file=not data_args.overwrite_cache,
        fn_kwargs={"split": "dev"},
        num_proc=6,
    )
    test_dataset = datasets["test"].map(
        preprocess_function,
        batched=False,
        with_indices=True,
        load_from_cache_file=not data_args.overwrite_cache,
        fn_kwargs={"split": "test"},
        num_proc=6,
    )
    train_dataset = test_dataset

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )  # bsz * span_num * labelsize
        preds = np.argmax(preds, axis=-1)  # bsz * span_num
        spans = (
            p.predictions[1] if isinstance(p.predictions, tuple) else None
        )  # bsz * span_num * 2
        labels = p.label_ids  # bsz * span_num

        tp = 0
        fp = 0
        fn = 0
        NA_LABEL = 0

        for i, (pred, label, span) in enumerate(zip(preds, labels, spans)):
            for pe, l, sp in zip(pred, label, span):
                if l == -100:
                    continue
                span_len = sp[1] - sp[0] + 1
                # This is the limitation of the model
                # We cannot predict spans with large length!
                if span_len > data_args.span_len:
                    pe = NA_LABEL
                if pe == l:
                    if pe != NA_LABEL:
                        tp += 1
                else:
                    if pe != NA_LABEL:
                        fp += 1
                    else:
                        fn += 1
            # log
            if i % 100 == 0:
                predict = []
                gold = []
                for pe, l, span in zip(pred, label, span):
                    if l == -100:
                        continue
                    span_len = span[1] - span[0] + 1
                    if span_len > data_args.span_len:
                        pe = NA_LABEL
                    if pe != 0:
                        predict.append((tuple(span), pe))
                    if l != 0:
                        gold.append([tuple(span), l])
                print("pred: {}".format(predict))
                print("gold: {}".format(gold))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print("tp: {}, fn: {}, fp: {}".format(tp, fn, fp))
        return {
            "p": p,
            "r": r,
            "f1": f1,
        }

    def collator_fn(examples):
        VOCAB_PAD = tokenizer.pad_token_id
        LABEL_PAD = -100
        input_ids = []
        labels = []
        spans = []
        event_ids = []
        span_lens = []
        label_masks = []
        trigger_index = []
        span_nums = []
        info_dicts = []
        start_labels = []
        end_labels = []
        subwords_span2snts = []
        subwords_snt2spans = []
        trigger_snt_ids = []

        for example in examples:
            input_ids.append(torch.LongTensor(example["input_ids"]))
            event_ids.append(example["event_id"])
            label_masks.append(example["label_mask"])
            trigger_index.append(example["trigger_index"])
            labels.append(example["label"])
            spans.append(example["spans"])
            span_lens.append(example["span_lens"])
            span_nums.append(example["span_num"])
            subwords_span2snts.append(example["subwords_span2snt"])
            subwords_snt2spans.append(example["subwords_snt2span"])
            trigger_snt_ids.append(example["trigger_snt_id"])
            info_dicts.append(example["info_dict"])

            start_labels.append(torch.LongTensor(example["start_label"]))
            end_labels.append(torch.LongTensor(example["end_label"]))

        max_span_num = max(span_nums)
        max_sent_num = max([len(x) for x in subwords_snt2spans])

        pad_spans = []
        pad_span_lens = []
        pad_labels = []
        pad_subwords_span2snts = []
        pad_subwords_snt2spans = []
        for span, span_len, label, subwords_span2snt, info_dict in zip(
            spans, span_lens, labels, subwords_span2snts, info_dicts
        ):
            pad_num = max_span_num - len(span)
            pad_spans.append(span + [[0, 0]] * pad_num)
            pad_span_lens.append(span_len + [1] * pad_num)
            # label = [[ 1, 2,], [2, 3]]
            # pad_labels = [[[1, 2, 0, 0], [2, 3, 0, 0]]]
            # pad_labels.append([sublist + [LABEL_PAD] * pad_num for sublist in label])
            pad_labels.append(label + [LABEL_PAD] * pad_num)
            # print("pad_labels:", pad_labels)
            pad_subwords_span2snts.append(subwords_span2snt + [0] * pad_num)
        for subwords_snt2span in subwords_snt2spans:
            pad_num = max_sent_num - len(subwords_snt2span)
            pad_subwords_snt2spans.append(subwords_snt2span + [[0, 0]] * pad_num)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=VOCAB_PAD
        )
        info_dicts = info_dicts
        spans = torch.LongTensor(pad_spans)
        event_ids = torch.LongTensor(event_ids)
        span_lens = torch.LongTensor(pad_span_lens)
        labels = torch.LongTensor(pad_labels)
        label_masks = torch.LongTensor(label_masks)
        trigger_index = torch.LongTensor(trigger_index)
        start_labels = torch.nn.utils.rnn.pad_sequence(
            start_labels, batch_first=True, padding_value=LABEL_PAD
        )
        end_labels = torch.nn.utils.rnn.pad_sequence(
            end_labels, batch_first=True, padding_value=LABEL_PAD
        )

        print("wow:", labels.shape, spans.shape)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids != VOCAB_PAD,
            "spans": spans,
            "span_lens": span_lens,
            "label_masks": label_masks,
            "trigger_index": trigger_index,
            "info_dicts": info_dicts,
            "start_labels": start_labels,
            "end_labels": end_labels,
        }

        return result

    # Initialize our Trainer
    trainer = Mytrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=collator_fn,
        not_bert_learning_rate=model_args.not_bert_learning_rate,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if (
                AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels
                == num_labels
            ):
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:

        def extract_word_level_result(preds, labels, spans, dataset):
            all_result = []
            for i, (pred, label, span, example) in enumerate(
                zip(preds, labels, spans, dataset)
            ):
                example = json.loads(example["text"])
                doc_key = example["doc_key"]
                result = {
                    "doc_key": doc_key,
                    "predictions": [[]],
                }
                sentences = example["sentences"]
                trigger = example["evt_triggers"][0]
                trigger_b, trigger_e, event = trigger[0], trigger[1], trigger[2][0][0]
                eventid = event2id[event]
                start_subwordidx2wordidx = [-1]
                end_subwordidx2wordidx = [-1]
                word_idx = 0
                for i, sentence in enumerate(sentences):
                    for j, word in enumerate(sentence):
                        if word_idx == trigger_b:
                            start_subwordidx2wordidx.append(-1)  # Special token
                            end_subwordidx2wordidx.append(-1)
                        if word_idx == trigger_e + 1:
                            start_subwordidx2wordidx.append(-1)  # Special token
                            end_subwordidx2wordidx.append(-1)
                        subwords_ids = tokenizer(
                            word, add_special_tokens=False, return_attention_mask=False
                        )["input_ids"]
                        start_subwordidx2wordidx.append(word_idx)
                        for _ in range(1, len(subwords_ids)):
                            start_subwordidx2wordidx.append(-1)
                        for _ in range(len(subwords_ids) - 1):
                            end_subwordidx2wordidx.append(-1)
                        end_subwordidx2wordidx.append(word_idx)
                        word_idx += 1
                start_subwordidx2wordidx.append(-1)
                end_subwordidx2wordidx.append(-1)

                result["predictions"][0].append([trigger_b, trigger_e])
                all_ready_in_result = set()
                for pe, l, sp in zip(pred, label, span):
                    if l == -100 or sp[1] - sp[0] + 1 > data_args.span_len or pe == 0:
                        continue
                    role_name = eventid2id2role[eventid][pe]
                    if data_args.task_name == "rams":
                        role_name = role_name[role_name.find("arg0") + 5 :]
                    if (
                        start_subwordidx2wordidx[sp[0]] == -1
                        or end_subwordidx2wordidx[sp[1]] == -1
                    ):
                        continue
                    final_r = tuple(
                        [
                            start_subwordidx2wordidx[sp[0]],
                            end_subwordidx2wordidx[sp[1]],
                            role_name,
                            1.0,
                        ]
                    )
                    if final_r not in all_ready_in_result:
                        all_ready_in_result.add(final_r)
                        result["predictions"][0].append(list(final_r))
                all_result.append(result)
            return all_result

        # on dev
        p = trainer.predict(test_dataset=eval_dataset)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        spans = p.predictions[1] if isinstance(p.predictions, tuple) else None
        labels = p.label_ids
        logger.info(preds)
        logger.info(spans)
        logger.info(labels)
        output_validation_file = os.path.join(
            training_args.output_dir, "validation_predictions_span.jsonlines"
        )
        if trainer.is_world_process_zero():
            result = extract_word_level_result(
                preds=preds, labels=labels, spans=spans, dataset=eval_dataset
            )
            with open(output_validation_file, "w") as f:
                for r in result:
                    f.write(json.dumps(r) + "\n")

        # on test
        p = trainer.predict(test_dataset=test_dataset)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        spans = p.predictions[1] if isinstance(p.predictions, tuple) else None
        labels = p.label_ids
        output_test_file = os.path.join(
            training_args.output_dir, "test_predictions_span.jsonlines"
        )
        if trainer.is_world_process_zero():
            result = extract_word_level_result(
                preds=preds, labels=labels, spans=spans, dataset=test_dataset
            )
            with open(output_test_file, "w") as f:
                for r in result:
                    f.write(json.dumps(r) + "\n")

    # log settings
    with open(os.path.join(training_args.output_dir, "myconfig.json"), "w") as f:
        f.write(json.dumps(vars(model_args), indent=4))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # wandb.init(project="event-extraction-tsar")
    os.environ["WANDB_DISABLED"] = (
        "true"  # Wandb cannot be used due to our internal servers. If you have any requirements, you can use Wandb.
    )
    main()
