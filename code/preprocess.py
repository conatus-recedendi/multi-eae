import json

"""
IN:
    - /data/wikievents/{}.jsonl
    - /data/wikievents/coref/{}.jsonlines
    - with meta data

OUT:

"""


def preprocess_function(example, idx, split):
    example = json.loads(example["text"])
    doc_key = example["doc_key"]
    sentences = example["sentences"]
    snt2span = []
    start, end = 0, 0

    # snt2span
    # [[문장 시작 span idx, 문장 마지막 span idx],]
    for idx, sen in enumerate(sentences):
        end = start + len(sen) - 1
        snt2span.append([start, end])
        start = start + len(sen)

    def which_snt(snt2span, span):
        for snt in range(len(snt2span)):
            snt_spans = snt2span[snt]
            if span[0] >= snt_spans[0] and span[1] <= snt_spans[1]:
                return snt
        assert False

    trigger = example["evt_triggers"][0]
    # trigger_b = trigger 시작하는 span: 155
    # trigger_e = trigger 끝나는 span: 155
    # event = trigger에 대한 event type: 'Contact.Contact.Broadcast'
    trigger_b, trigger_e, event = trigger[0], trigger[1], trigger[2][0][0]
    # trigger_snt_id = trigger가 속한 문장의 idx: 0
    trigger_snt_id = which_snt(snt2span, [trigger_b, trigger_e])

    # ["Cognitive.IdentifyCategorize.Unspecified", ["Place", "IdentifiedRole", "IdentifiedObject", "Identifier"]]
    # event2id = { 'Contact.Contact.Broadcast': 0, ... } -> from meta file. ontology마다 새로 정의되지만, 데이터셋에 따라 정의되지는 않는 정보들.
    # eventid = event 자체 : 0
    eventid = event2id[event]

    now_snt_idx = 0
    input_ids = [tokenizer.cls_token_id]
    subwords_snt2span = []
    wordidx2subwordidx = []

    # 제거할 토큰들. but span 정보는 남겨둬야 하므로...?
    exclude_words = []  # non-argument spans exclusion
    if data_args.task_name == "wikievent":
        exclude_symbols = [
            ",",
            "!",
            "?",
            ":",
        ]  # We select some normal symols that can not appear in the middle of a argument span. For different datasets, you can choose different symbols.
    else:
        exclude_symbols = [",", ".", "!", "?", ":"]

    for i, sentence in enumerate(sentences):
        subwords_snt2span_st = len(input_ids)
        for j, word in enumerate(sentence):

            if now_snt_idx == trigger_b:
                trig_sub_s = len(input_ids)
                exclude_words.append(trig_sub_s)
                input_ids.append(TRIGGER_LEFT)  # Special token
            if now_snt_idx == trigger_e + 1:
                trig_sub_e = len(input_ids)
                exclude_words.append(trig_sub_e)
                input_ids.append(TRIGGER_RIGHT)  # Special token
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
            )  # [a, b]
            input_ids.extend(subwords_ids)
            now_snt_idx += 1
        subwords_snt2span.append([subwords_snt2span_st, len(input_ids) - 1])

    model_max_len = 1024
    max_role_token_len = 30  # We set the max length of role list 30

    spans = []
    span_lens = []
    span_labels = []
    label_mask = [0] * num_labels
    label_mask[0] = 1
    label_mask = np.array(label_mask)
    subwords_span2snt = []
    for link in example["gold_evt_links"]:
        role_b, role_e = link[1]
        role = link[-1]
        if role not in eventid2role2id[eventid]:
            continue
        roleid = eventid2role2id[eventid][role]
        base_roleid = list(eventid2id2role[eventid].keys())[0]
        upper_roleid = list(eventid2id2role[eventid].keys())[-1]
        label_mask[base_roleid : upper_roleid + 1] = 1
        role_subword_start_idx = wordidx2subwordidx[role_b][0]
        role_subword_end_idx = wordidx2subwordidx[role_e][-1]
        if role_subword_end_idx < model_max_len:
            spans.append([role_subword_start_idx, role_subword_end_idx])
            subwords_span2snt.append(which_snt(subwords_snt2span, spans[-1]))
            span_lens.append(
                min(role_subword_end_idx - role_subword_start_idx, config.len_size - 1)
            )
            span_labels.append(roleid)

    role_nums = label_mask.sum() - 1
    role_list = list(eventid2id2role[eventid].values())
    role_id_list = list(eventid2id2role[eventid].keys())
    if data_args.task_name == "rams":
        for itt in range(len(role_list)):
            role_list[itt] = role_list[itt].split("arg")[-1][2:]
    if len(input_ids) > model_max_len - 1:
        input_ids = input_ids[
            : model_max_len - max_role_token_len - role_nums
        ]  # 这里默认角色
    input_ids.append(tokenizer.sep_token_id)
    ari_len = len(input_ids)

    event_split_list = event.split(".")
    event_tok = EVENT_START + eventid

    input_ids.append(event_tok)
    info_dict = {}
    info_dict["words_num"] = ari_len
    info_dict["event_idx"] = len(input_ids) - 1
    info_dict["event_ids"] = role_id_list

    for item in event_split_list:
        event_subwords_ids = tokenizer(
            item, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        input_ids.extend(event_subwords_ids)
    input_ids.append(event_tok)
    role_idx = []

    for rr, role_t in enumerate(role_list):
        role_subwords_ids = tokenizer(
            role_t, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        input_ids.append(ROLE_START + role_id_list[rr])
        role_idx.append(len(input_ids) - 1)
        input_ids.extend(role_subwords_ids)
        input_ids.append(ROLE_START + role_id_list[rr])

    input_ids.append(ROLE_START)
    role_idx.append(len(input_ids) - 1)  # 空类标签
    info_dict["role_idxs"] = role_idx

    trigger_index = (
        wordidx2subwordidx[trigger_b][0] - 1
    )  # 这里用special token 代表触发词
    trigger_index = min(
        trigger_index, len(input_ids) - 1
    )  # very few times it would be out of bound so we have to ...

    # construct start label and end label
    start_label = [0 for _ in range(len(input_ids))]
    end_label = [0 for _ in range(len(input_ids))]
    for start_end_span in spans:
        start, end = start_end_span
        start_label[start] = 1
        end_label[end] = 1

    # construct negative examples
    all_non_spans = []
    for i in range(len(sentences)):
        start_idx, end_idx = subwords_snt2span[i]
        end_idx = min(end_idx, model_max_len - 1)
        for s in range(start_idx, end_idx + 1):
            for e in range(s, end_idx + 1):
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
        # desc: tookneizer 결과
        # 변경전: sentence + event + role
        # 변경후: sentence + [event + role] + ...
        "input_ids": input_ids,
        # desc: w
        "label": span_labels,
        # desc: 모든 spans 조합(role도 포함)
        "spans": spans,
        # desc: eventid
        # num
        # num[]
        "event_id": eventid,
        # desc: "spans"의 각 스판의 길이
        "span_lens": span_lens,
        # desc: label의 위치
        "label_mask": label_mask,
        "trigger_index": trigger_index,
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
    return result
