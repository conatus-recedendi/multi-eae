import json
import os
from tqdm import tqdm


def transfer_multi(split: str, output_dir: str = "./", MAX_NUM: int = 600):

    data = []
    output = []
    coref = {}
    with open("{}.jsonl".format(split), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    with open("coref/{}.jsonlines".format(split), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            coref[d["doc_key"]] = d["clusters"]

    for d in tqdm(data):
        # 파일 읽기

        doc_id = d["doc_id"]
        sentences = d["sentences"]
        tokens = d["tokens"]
        text = d["text"]
        entity_mentions = d["entity_mentions"]
        event_mentions = d["event_mentions"]

        # 초기화
        current_length = 0
        current_sentences = []
        split_data = []
        current_tokens = []
        current_text_parts = []
        current_entity_mentions = []
        current_event_mentions = []
        split_index = 1
        sentence_start_index = 0
        current_sent_idx = 0

        for idx, sentence_data in enumerate(sentences):
            sentence_tokens = sentence_data[0]
            sentence_text = sentence_data[1]

            sentence_length = sum([len(word[0]) for word in sentence_tokens])
            if current_length + sentence_length > MAX_NUM:
                # 새 JSON으로 저장
                split_doc_id = f"{doc_id}-X{str(split_index).zfill(2)}"
                split_data.append(
                    {
                        "doc_id": split_doc_id,
                        "tokens": current_tokens,
                        "text": " ".join(current_text_parts),
                        "sentences": current_sentences,
                        "entity_mentions": current_entity_mentions,
                        "event_mentions": current_event_mentions,
                    }
                )
                # 초기화
                split_index += 1
                current_length = 0
                current_sentences = []
                current_tokens = []
                current_text_parts = []
                current_entity_mentions = []
                current_event_mentions = []

            # 현재 데이터에 추가
            current_sentences.append(sentence_data)
            current_tokens.extend([token[0] for token in sentence_tokens])
            current_text_parts.append(sentence_text)
            current_length += sentence_length

            # Entity_mentions 추가
            for entity in entity_mentions:
                if entity["sent_idx"] == idx and (
                    entity["start"] < len(current_tokens)
                    or entity["end"] > len(current_tokens)
                ):
                    entity["start"] -= sentence_start_index
                    entity["end"] -= sentence_start_index
                    if entity["sent_idx"] != current_sent_idx:
                        print("ERROR 002: sent_idx mismatch idx", entity, idx)
                    current_entity_mentions.append(entity)

            coref_entities = coref.get(d["doc_id"])

            for related_entity in entity_mentions:
                if related_entity and related_entity not in [
                    e for e in current_entity_mentions
                ]:
                    for coref_related_entity_ids in coref_entities:
                        for coref_related_entity_id in coref_related_entity_ids:
                            if related_entity[
                                "id"
                            ] == coref_related_entity_id and related_entity not in [
                                e for e in current_entity_mentions
                            ]:
                                related_entity["start"] -= sentence_start_index
                                related_entity["end"] -= sentence_start_index
                                if related_entity["sent_idx"] != current_sent_idx:
                                    print(
                                        "ERROR 000 : sent_idx mismatch idx",
                                        related_entity,
                                        idx,
                                    )
                                related_entity["sent_idx"] = current_sent_idx
                                current_entity_mentions.append(related_entity)

            # for related_entity_ids in coref_entities:
            #     for related_entity_id in related_entity_ids:

            #         if related_entity_id == entity["id"]:
            #             for related_entity in entity_mentions:
            #                 if related_entity and related_entity not in [
            #                     e for e in current_entity_mentions
            #                 ]:
            #                     related_entity["start"] -= sentence_start_index
            #                     related_entity["end"] -= sentence_start_index
            #                     if (
            #                         related_entity["sent_idx"]
            #                         != current_sent_idx
            #                     ):
            #                         print(
            #                             "ERROR: sent_idx mismatch idx",
            #                             related_entity,
            #                             idx,
            #                         )
            #                     related_entity["sent_idx"] = current_sent_idx
            #                     current_entity_mentions.append(related_entity)
            # for related_entity in related_entities:
            #     if related_entity and related_entity not in [
            #         e for e in current_entity_mentions
            #     ]:
            #         current_entity_mentions.append(related_entity)

            # Event_mentions 추가
            # print(current_entity_mentions)
            for event in event_mentions:
                # print(event)
                trigger_in_range = event["trigger"]["sent_idx"] == idx

                # entity[id] is str
                arguments_in_range = any(
                    arg["entity_id"]
                    in [entity["id"] for entity in current_entity_mentions]
                    for arg in event["arguments"]
                )

                if trigger_in_range or arguments_in_range:
                    if event["id"] not in [e["id"] for e in current_event_mentions]:
                        event["trigger"]["start"] -= sentence_start_index
                        event["trigger"]["end"] -= sentence_start_index
                        if event["trigger"]["sent_idx"] != current_sent_idx:
                            print("ERROR 001: sent_idx mismatch idx", event, idx)
                        event["trigger"]["sent_idx"] = current_sent_idx
                        current_event_mentions.append(event)

            sentence_start_index += len(sentence_tokens)
            current_sent_idx += 1

        # 마지막 조각 저장
        if current_sentences:
            split_doc_id = f"{doc_id}-E{str(split_index).zfill(2)}"
            split_data.append(
                {
                    "doc_id": split_doc_id,
                    "tokens": current_tokens,
                    "text": " ".join(current_text_parts),
                    "sentences": current_sentences,
                    "entity_mentions": current_entity_mentions,
                    "event_mentions": current_event_mentions,
                }
            )

        output.extend(split_data)

    # JSON 파일로 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"transfer-{split}-multi.jsonl"), "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    transfer_multi("train")
    transfer_multi("dev")
    transfer_multi("test")
