import sys
import argparse
gpu = sys.argv[1]
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gemma2:2b")
parser.add_argument("--max_token", type=int, default=2000)
parser.add_argument("--port", type=int, default=11000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--method", type=str, choices=["mobileKGQA", "CoT", "RLM"])
args, _ = parser.parse_known_args()

import numpy as np
from tqdm import tqdm
import json
from bert_score import score
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gen_annotations import generate_question_mobileKGQA, generate_question_RLM, generate_question_CoT
from src.module.ollama_LLM import OLLAMA_LLM
from src.utils.evaluate import rouge_l_score


def evaluate_gen_methods(args):
    with open("./data/preprocessed_data/webqsp/qid2sampled_triples.json", "r") as rf:
        qid2sampled = json.load(rf)

    model = OLLAMA_LLM(
        model_name=args.model_name, 
        port=args.port, 
        max_token=args.max_token, 
        seed=args.seed
        )
    
    wf = open(f"./data/preprocessed_data/webqsp/sentences_{args.method}_{args.model_name}_{args.seed}.txt", "wt")
    for qid, data in tqdm(qid2sampled.items(), desc="generating question", ncols=100):
        triples, a_entity, q_entity, ref_question = data["chosen_triples"], data["a_entity"], data["q_entity"], data["question"]

        if args.method == "mobileKGQA":
            gen_question = generate_question_mobileKGQA(triples, a_entity, q_entity, model)
        elif args.method == "RLM":
            gen_question = generate_question_RLM(triples, a_entity, q_entity, model)
        elif args.method == "CoT":
            gen_question = generate_question_CoT(triples, a_entity, q_entity, model)
        else:
            raise ValueError("Invalid method")
        gen_question = gen_question[:-1] if gen_question.endswith("?") else gen_question
        gen_question = gen_question.replace("\n", " ") if "\n" in gen_question else gen_question

        wf.write(f"{qid}##{len(triples)}##{ref_question}##{gen_question}\n")
        wf.flush()
    wf.close()

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    eval_model = AutoModel.from_pretrained("roberta-large")

    rouge_l, bert_score = [[], [], []], [[], [], []]
    total_rouge_l, total_bert_score = [], []
    wf = open(f"./data/preprocessed_data/webqsp/gen_vs_ref_evaluation_{args.method}_{args.model_name}_{args.seed}.txt", "wt")
    with open(f"./data/preprocessed_data/webqsp/sentences_{args.method}_{args.model_name}_{args.seed}.txt", "r") as rf:
        for line in tqdm(rf.readlines(), desc="evaluating question", ncols=100):
            qid, triple_len, ref_question, gen_question = line.strip().split("##")
            qid = qid.strip()
            ref_question = ref_question.strip()
            gen_question = gen_question.strip()

            idx = 2 if int(triple_len) >=3 else int(triple_len)-1

            rouge_score = rouge_l_score(gen_question, ref_question)["f1"]
            rouge_l[idx].append(rouge_score)
            total_rouge_l.append(rouge_score)
            _, _, bert_score_f1 = score(
                [gen_question],
                [ref_question],
                lang="en",
                model_type=eval_model,
                tokenizer=tokenizer,
                rescale_with_baseline=True
                )
            bert_score[idx].append(bert_score_f1.item())
            total_bert_score.append(bert_score_f1.item())
            
            wf.write(f"{qid}, {ref_question}, {gen_question}, {bert_score_f1.item()}, {rouge_score}\n")
            wf.flush()

        for idx in range(len(rouge_l)):
            rouge_l[idx] = sum(rouge_l[idx]) / len(rouge_l[idx])
        for idx in range(len(bert_score)):
            bert_score[idx] = sum(bert_score[idx]) / len(bert_score[idx])

        avg_rouge_l = sum(total_rouge_l) / len(total_rouge_l)
        avg_bert_score = sum(total_bert_score) / len(total_bert_score)
        print(f"method: {args.method}, seed: {args.seed}, model: {args.model_name}")
        print("Rouge-L: ", rouge_l, "total: ", avg_rouge_l)
        print(f"bert_score: ", bert_score, "total: ", avg_bert_score)
        wf.write(f"Rouge-L: {rouge_l}, total: {avg_rouge_l}\n")
        wf.write(f"bert_score: {bert_score}, total: {avg_bert_score}\n")
        wf.close()

if __name__ == '__main__':
    evaluate_gen_methods(args)