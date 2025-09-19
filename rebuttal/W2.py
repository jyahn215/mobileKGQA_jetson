import json
import numpy as np
import ast

with open("./../data/preprocessed_data/cwq/qid2triple_num.json", "r") as rf:
    qid2triple_num = json.load(rf)

# with open("./../data/preprocessed_data/cwq/_domains/total/test.json", "r") as rf:
#     test_data = [json.loads(line) for line in rf.readlines()]

# # calculate ReaRev
# best_ReaRev_CWQ_config = "2025-01-16-19-05-05"
# with open(f"./../ckpts/ReaRev/{best_ReaRev_CWQ_config}_test.info", "r") as rf:
#     cnt = 0
#     f1_list = [[], [], []]
#     null_f1_list = []
#     total_f1_list = []
#     for line in rf.readlines():
#         f1 = json.loads(line)["f1"]
#         total_f1_list.append(f1)
#         qid = test_data[cnt]["id"]
#         hop_num = qid2triple_num[qid]
#         if hop_num == None:
#             null_f1_list.append(f1)
#         else:
#             if hop_num == 1:
#                 f1_list[0].append(f1)
#             elif hop_num == 2:
#                 f1_list[1].append(f1)
#             elif hop_num >= 3:
#                 f1_list[2].append(f1)
#         cnt += 1

#     for idx, f1s in enumerate(f1_list):
#         print(f"hop_num: {idx} / {sum(f1s) / len(f1s)}")

# calculate mobileKGQA
result = np.loadtxt("./../logs_2025-02-12-14-14-02_gemma2:2b.txt", dtype=str, delimiter="\t")

f1_list = [[], [], []]
for row in result:
    qid, f1, hit = row
    f1 = ast.literal_eval(f1)[0]

    hop_num = qid2triple_num[qid]
    if hop_num != None:
        hop_num = int(hop_num)
        if hop_num == 1:
            f1_list[0].append(f1)
        elif hop_num == 2:
            f1_list[1].append(f1)
        elif hop_num >= 3:
            f1_list[2].append(f1)

for idx, f1s in enumerate(f1_list):
    print(f"hop_num: {idx} / {sum(f1s)/len(f1s)}")

