import json
import os
from tqdm import tqdm

MAX_SNT_LEN = 6
MAX_SNT_TOKEN_LEN = 480


def transfer_multi(split: str, output_dir: str = "./", MAX_NUM: int = 600):
    data = []
    output = [
        # {
        #     "doc_id": "doc_id",
        #     "tokens": ["tokens"],
        #     "text": "text",
        #     "entity_mentions": [],
        #     "event_mentions": [],
        # }
    ]
    coref = {}

    with open("{}.jsonl".format(split), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    with open("coref/{}.jsonlines".format(split), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            coref[d["doc_key"]] = d["clusters"]

    for d in tqdm(data):
        doc_id = d["doc_id"]
        sentences = d["sentences"]
        tokens = d["tokens"]
        text = d["text"]
        entity_mentions = d["entity_mentions"]
        event_mentions = d["event_mentions"]
        sentences = d["sentences"]

        # 같은 argument를 공유하는 이벤트끼리 묶습니다.
        cluster_event_id = [
            # ["1", "2", "3"], ["4", "5"], ["6", "7", "8"]
        ]
        cluster_event_data = [
            # {
            #     "start_snt_idx": 0,
            #     "end_snt_idx": 0,
            #     "start_idx": 0,
            #     "end_idx": 0,
            # }
        ]

        for idx, event in enumerate(event_mentions):
            # 1. 그 이벤트에 해당하는 sentence를 앞뒤로 잘라오기
            # 2. 만약 그 sentence 범위 내에 동일한 argument entity를 argument로 가진 이벤트가 있다면, 같은 데이터에 포함하기
            # 3. 만약 그 sentence 범위 내에 동일한 argument entity를 argument로 가진 이벤트가 coref로 연결되어 있다면, 같은 데이터에 포함하기
            # 4. 만약, 이미 다른 곳에서 포함된 event라면 skip

            # 추가 1) entity 위치랑 trigger의 위치는 잘린 sentence에서 재조정해야함
            # 추가 2) 합친 데이터에서는 coref 정보를 따로 표현하지 않게끔, id 대신에 start/end 데이터만 남겨두 무방
            # 추가 3) clustervent_id로만 데이터 셋을 만들진 않을 수도 있음. 예를 들어 하나의 클러스터가 5개의 이벤트로 구성한다면, 5C2 = 10개의 데이터로 표현할 수도..?

            # 1. 만약, 이미 다른 곳에서 포함된 event라면 skip
            # cluster_event 안에 event_mention이 포함되어있는지 확인
            # is_event_included = False
            # for idx, cluster in cluster_event_id:
            #     if event["id"] in cluster:

            #         is_event_included = True
            #         break

            # if is_event_included:
            #     continue

            # 2. 그 이벤트에 해당하는 sentence를 앞뒤로 잘라오기
            # sentence의 범위는 다음과 같은 규칙을 따름
            # 2-1 반드시 이벤트를 포함하는 문장은 포함되어야 함
            # 2-2 이벤트를 포함하는 문장의 앞뒤로 N개의 문장을 포함해야 함
            # 2-3. 단, 문장의 시작이나 끝에 단어가 잘리지 않아야 함
            # 2-4. 문장의 길이가 MAX_SNT_LEN을 넘어서는 안됨
            # 2-5. start_snt_idx는 이런 문장들의 조합의 시작 문장의 idx

            # start_snt_idx = 9999999
            # end_snt_idx = -1
            # trigger_start_idx = event["trigger"]["start"]
            # trigger_end_idx = event["trigger"]["end"]

            # min_start_idx = trigger_start_idx
            # max_end_idx = trigger_end_idx

            # event_mentions.arguments의 argument에 start, end, sent_idx 값을 추가하려고해
            # 이때, start, end, sent_idx는 entity_mention에 있어. entity_mentions은 아래와 같은 구조의 List야.
            #             {
            #   "id": "road_ied_8-T35",
            #   "sent_idx": 4,
            #   "start": 120,
            #   "end": 121,
            #   "entity_type": "ORG",
            #   "mention_type": "UNK",
            #   "text": "agency"
            # }
            # event_mentions[idx] = {
            #     "arguments": event_mentions[idx]["arguments"].map(
            #         lambda y: y.update(
            #             {
            #                 "sent_idx": next(
            #                     (
            #                         entity
            #                         for entity in entity_mentions
            #                         if entity["id"] == y["entity_id"]
            #                     ),
            #                     None,
            #                 )["sent_idx"],
            #                 "start": next(
            #                     (
            #                         entity
            #                         for entity in entity_mentions
            #                         if entity["id"] == y["entity_id"]
            #                     ),
            #                     None,
            #                 )["start"],
            #                 "end": next(
            #                     (
            #                         entity
            #                         for entity in entity_mentions
            #                         if entity["id"] == y["entity_id"]
            #                     ),
            #                     None,
            #                 )["end"],
            #             }
            #         )
            #     )
            # }
            event_mentions[idx]["arguments"] = [
                {
                    **arg,  # 기존 값 유지
                    **{
                        "sent_idx": next(
                            (
                                entity["sent_idx"]
                                for entity in entity_mentions
                                if entity["id"] == arg["entity_id"]
                            ),
                            None,
                        ),
                        "start": next(
                            (
                                entity["start"]
                                for entity in entity_mentions
                                if entity["id"] == arg["entity_id"]
                            ),
                            None,
                        ),
                        "end": next(
                            (
                                entity["end"]
                                for entity in entity_mentions
                                if entity["id"] == arg["entity_id"]
                            ),
                            None,
                        ),
                    },
                }
                for arg in event["arguments"]  # 리스트의 각 요소에 대해 적용
            ]

        for idx, event in enumerate(event_mentions):

            is_event_included = False
            if len(cluster_event_id) != 0:
                for cluster in cluster_event_id:
                    if event["id"] in cluster:
                        is_event_included = True
                        break

            if is_event_included:
                continue

            cluster_event_id.append([event["id"]])

            start_snt_idx = event["trigger"]["sent_idx"]
            end_snt_idx = event["trigger"]["sent_idx"]
            trigger_start_idx = event["trigger"]["start"]
            trigger_end_idx = event["trigger"]["end"]

            min_start_idx = trigger_start_idx
            max_end_idx = trigger_end_idx

            for argument in event["arguments"]:
                entity_id = argument["entity_id"]

                min_start_idx = min(min_start_idx, argument["start"])
                max_end_idx = max(max_end_idx, argument["end"])
                start_snt_idx = min(start_snt_idx, argument["sent_idx"])
                end_snt_idx = max(end_snt_idx, argument["sent_idx"])

            if max_end_idx - min_start_idx > MAX_SNT_TOKEN_LEN:
                print(
                    f"ERROR 002: sentence length is too long: {max_end_idx} - {min_start_idx} = {max_end_idx - min_start_idx}"
                )
                continue

            if max_end_idx > len(tokens):
                print("ERROR 000: sentence is out of range")
                continue

            if min_start_idx < 0:
                print("ERROR 001: sentence is out of range")
                continue

            # 3. 만약 그 sentence 범위 내에 동일한 argument entity를 argument로 가진 이벤트가 있다면, 같은 데이터에 포함하기
            for idxs, nested_event in enumerate(event_mentions):
                is_event_included = False
                for cluster in cluster_event_id:
                    if nested_event["id"] in cluster:
                        is_event_included = True
                        continue

                if is_event_included:
                    continue
                # 2. 만약 그 sentence 범위 내에 동일한 argument entity를 argument로 가진 이벤트가 있다면, 같은 데이터에 포함하기

                # nested_event와 event의 argument entity가 동일한지 확인
                is_same_argument = False
                for argument in nested_event["arguments"]:
                    entity_id = argument["entity_id"]
                    if entity_id in [arg["entity_id"] for arg in event["arguments"]]:
                        min_start_idx = min(min_start_idx, argument["start"])
                        max_end_idx = max(max_end_idx, argument["end"])
                        start_snt_idx = min(start_snt_idx, argument["sent_idx"])
                        end_snt_idx = max(end_snt_idx, argument["sent_idx"])
                        is_same_argument = True
                        break

                # 3. 만약 그 sentence 범위 내에 동일한 argument entity를 argument로 가진 이벤트가 coref로 연결되어 있다면, 같은 데이터에 포함하기
                # TODO: Implement this
                is_same_coref = False

                if is_same_argument or is_same_coref:
                    cluster_event_id[-1].append(event["id"])

            # min_ = min_start_idx
            # max = max_end_idx
            token_idx_cnt = 0
            for idx, sentence in enumerate(sentences):
                sentence_tokens = sentence[0]
                if (
                    min_start_idx >= token_idx_cnt
                    and min_start_idx < token_idx_cnt + len(sentence_tokens)
                ):
                    start_snt_idx = idx
                    min_start_idx = token_idx_cnt

                if max_end_idx >= token_idx_cnt and max_end_idx < token_idx_cnt + len(
                    sentence_tokens
                ):
                    end_snt_idx = idx
                    max_end_idx = token_idx_cnt + len(sentence_tokens) - 1
                token_idx_cnt += len(sentence_tokens)
            cluster_event_data.append(
                {
                    "start_snt_idx": start_snt_idx,
                    "end_snt_idx": end_snt_idx,
                    "start_idx": min_start_idx,
                    "end_idx": max_end_idx,
                }
            )

        # clusteR_event_data 기준으로 sentence parse. index 값 조절
        # 1. 일단 모든 이벤트들을 포함하는 snt가 MAX_SNT_LEN을 넘지 않는지 확인
        # 2. 넘지 않는다면, 앞 뒤로 N개의 문장을 하나씩 추가하면서 MAX_SNT_LEN을 넘지 않는지 확인
        # 3. 넘는다면, 그때까지의 문장을 하나의 데이터로 저장
        for idx, (event_ids, event_snt_data) in enumerate(
            zip(cluster_event_id, cluster_event_data)
        ):
            # Log
            # print(
            #     f"idx:{idx}\nevent_ids:{event_ids}\nevent_snt_data:{event_snt_data}\ncluster_event_id:{cluster_event_id}\ncluster_event_data:{cluster_event_data}"
            # )
            cluster_output = {
                "doc_id": f"{doc_id}-X{str(idx).zfill(3)}",
                "tokens": [],
                "text": "",
                "sentences": [],
                "entity_mentions": [],
                "event_mentions": [],
            }
            current_snt_length = (
                event_snt_data["end_snt_idx"] - event_snt_data["start_snt_idx"] + 1
            )
            current_snt_token_length = (
                event_snt_data["end_idx"] - event_snt_data["start_idx"] + 1
            )

            if current_snt_length > MAX_SNT_LEN:
                print("ERROR 003 : sentence length is too long")
                continue
            if current_snt_token_length > MAX_SNT_TOKEN_LEN:
                print(current_snt_token_length, doc_id)
                print("ERROR 004: sentence token length is too long")
                continue

            if (
                current_snt_length >= MAX_SNT_LEN
                or current_snt_token_length >= MAX_SNT_TOKEN_LEN
            ):
                print("ERROR 005: sentence length is too long")
                continue
                # end
                # TODO

                # sentence - 1
                # sentence + 1
            iteration = 0
            left_done = False
            right_done = False

            while True:

                if event_snt_data["end_snt_idx"] == len(sentences) - 1:
                    right_done = True
                if event_snt_data["start_snt_idx"] == 0:
                    left_done = True

                if left_done and right_done:
                    break

                if (
                    iteration % 2 == 0 and right_done is False
                ):  # 짝수 회차에서는 max + 1

                    event_snt_data["end_snt_idx"] += 1

                    next_sentence = sentences[event_snt_data["end_snt_idx"]]

                    event_snt_data["end_idx"] += len(next_sentence[0])
                elif (
                    iteration % 2 != 0 and left_done is False
                ):  # 홀수 회차에서는 min - 1

                    event_snt_data["start_snt_idx"] -= 1
                    prev_sentence = sentences[event_snt_data["start_snt_idx"]]
                    event_snt_data["start_idx"] -= max(len(prev_sentence[0]), 0)

                # max - min 차이 계산
                difference = (
                    event_snt_data["end_snt_idx"] - event_snt_data["start_snt_idx"] + 1
                )
                # 중지 조건 확인
                if difference >= MAX_SNT_LEN:
                    break
                # 반복 횟수 증가
                iteration += 1

            first_token_idx = event_snt_data["start_idx"]
            first_snt_idx = event_snt_data["start_snt_idx"]
            # print(first_token_idx, first_snt_idx)
            # start_snt_idx부터 iteraction 돌면서 sentence를 추가

            cluster_output["sentences"] = sentences[
                event_snt_data["start_snt_idx"] : event_snt_data["end_snt_idx"] + 1
            ]

            cluster_output["tokens"] = tokens[
                event_snt_data["start_idx"] : event_snt_data["end_idx"] + 1
            ]
            # cluster_output["text"] = cluster_output["tokens"].join(" ")
            cluster_output["text"] = " ".join(cluster_output["tokens"])
            cluster_output["event_mentions"] = [
                {
                    **event,  # 기존 이벤트 복사
                    "arguments": [
                        {
                            **arg,  # 기존 값 유지
                            "start": arg["start"] - first_token_idx,
                            "end": arg["end"] - first_token_idx,
                            "sent_idx": arg["sent_idx"] - first_snt_idx,
                        }
                        for arg in event["arguments"]
                    ],
                    "trigger": {
                        **event["trigger"],  # 기존 trigger 복사
                        "start": event["trigger"]["start"] - first_token_idx,
                        "end": event["trigger"]["end"] - first_token_idx,
                        "sent_idx": event["trigger"]["sent_idx"] - first_snt_idx,
                    },
                }
                for event in event_mentions
                if event["id"] in event_ids
            ]

            cluster_output["entity_mentions"] = [
                {
                    **entity,  # 기존 값을 복사
                    "start": entity["start"] - first_token_idx,
                    "end": entity["end"] - first_token_idx,
                    "sent_idx": entity["sent_idx"] - first_snt_idx,
                }
                for entity in entity_mentions
                if entity["id"]
                in [
                    argument["entity_id"]
                    for event in cluster_output["event_mentions"]
                    for argument in event["arguments"]
                ]
            ]
            # input()
            output.append(cluster_output)
            # end

        # input()
    #
    with open(os.path.join(output_dir, f"transfer-{split}-new-multi.jsonl"), "w") as f:
        for d in output:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    transfer_multi("dev")
    transfer_multi("train")
    transfer_multi("test")
#
