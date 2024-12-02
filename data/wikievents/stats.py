import json
from collections import defaultdict, Counter


# case1: event 간에 coref 이 존재하는 갯수
# case2: 전체 event 수
def stats_multi(split: str):
    event_coref_count = 0
    event_count = 0
    data = []
    coref = {}
    # JSON 파일 읽기
    with open("{}.jsonl".format(split), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    with open("coref/{}.jsonlines".format(split), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            coref[d["doc_key"]] = d["clusters"]

    for d in data:
        event_count += len(d["event_mentions"])

        for event in d["event_mentions"]:
            for arg in event["arguments"]:
                entity_id = arg["entity_id"]
                # print(coref[d["doc_id"]], entity_id)

                for cluster in coref[d["doc_id"]]:
                    if entity_id in cluster:
                        event_coref_count += 1
                        break

    print(
        f"{split} event_coref_count: {event_coref_count} / event_count: {event_count}"
    )


if __name__ == "__main__":
    stats_multi("train")
    stats_multi("dev")
    stats_multi("test")
