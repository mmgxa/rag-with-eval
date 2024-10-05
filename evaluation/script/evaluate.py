from .prompts import (
    QA_GEN_PROMPT,
    GROUNDEDNESS_RUBRIC,
    RAG_SYS_PROMPT,
    RAG_PROMPT,
    EVALUATION_RUBRIC,
)
from tqdm import tqdm
import random
from .query import query
from rerankers import Reranker
from .score import calculate_rr_and_hit
import time


def create_qna_dataset(docs, N_GENERATIONS, qna_llm):
    outputs = []
    for row in tqdm(random.sample(docs, N_GENERATIONS)):
        # Generate QA couple
        output_QA_couple, tokens = qna_llm.generate(
            prompt=QA_GEN_PROMPT.format(context=row["text"])
        )
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split(
                "Answer: "
            )[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": row["text"],
                    "question": question,
                    "answer": answer,
                    "source": row["source"],
                    "id": row["id"],
                }
            )
        except:
            continue
    return outputs


def qna_critique(outputs, judge_llm):
    for output in tqdm(outputs):
        evaluations = {
            "groundedness": judge_llm.generate(
                output["question"], output["context"], GROUNDEDNESS_RUBRIC, "wo_ref"
            )
        }
        try:
            for criterion, evaluation in evaluations.items():
                feedback, score = evaluation

                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": feedback,
                    }
                )
        except Exception as e:
            continue


def filter_crit(
    qna_with_critique, ground_min_score=4, rel_min_score=4, self_min_score=4
):
    qna_with_critique_filtered = qna_with_critique.loc[
        (qna_with_critique["groundedness_score"] >= ground_min_score)
        # & (qna_with_critique["relevance_score"] >= rel_min_score)
        # & (qna_with_critique["standalone_score"] >= self_min_score)
    ]
    return qna_with_critique_filtered


def eval_ret(
    qna_with_critique_filtered,
    table,
    emb_model,
    reranker,
    reranker_name,
    method="vector",
):
    outputs = []
    total_rr = 0.0  # For MRR
    total_hits = 0  # For Hit Rate

    for index, row in tqdm(qna_with_critique_filtered.iterrows()):
        # perform sim search
        question = row["question"]
        source_id = row["id"]

        query_result = query(
            table,
            question,
            emb_model,
            reranker=reranker,
            method=method,
            metric="cosine",
            limit=5,
        )
        retrieved_docs_id = [doc["id"] for doc in query_result]
        rr, hit = calculate_rr_and_hit(retrieved_docs_id, source_id)

        total_rr += rr
        total_hits += hit
        result = {
            "question": question,
            "true_answer": row["answer"],
            "source_doc": row["source"],
            "source_id": source_id,
            "retrieved_docs": [doc["text"] for doc in query_result],
            "retrieved_docs_id": retrieved_docs_id,
            "reranker_type": reranker_name,
            "search_type": method,
            "rr": rr,
            "hit": hit,
        }
        outputs.append(result)

    mrr = total_rr / len(qna_with_critique_filtered)
    hit_rate = total_hits / len(qna_with_critique_filtered)
    return outputs, str(mrr), str(hit_rate)


def rag(
    qna_with_critique_filtered,
    table,
    emb_model,
    rag_llm,
    reranker,
    reranker_name,
    method="vector",
):
    outputs = []

    for index, row in tqdm(qna_with_critique_filtered.iterrows()):
        # perform sim search
        context = "\nExtracted documents:\n"
        question = row["question"]
        source_id = row["id"]

        query_result = query(
            table,
            question,
            emb_model,
            reranker=reranker,
            method=method,
            metric="cosine",
            limit=5,
        )
        retrieved_docs_id = [doc["id"] for doc in query_result]

        # build context from query_results
        context += "".join([doc["text"] + "\n" for doc in query_result])

        final_sys_prompt = RAG_SYS_PROMPT.format(context=context)
        final_prompt = RAG_PROMPT.format(question=question)
        answer, _ = rag_llm.generate(prompt=final_prompt, sys_prompt=final_sys_prompt)
        result = {
            "question": question,
            "true_answer": row["answer"],
            "source_doc": row["source"],
            "source_id": source_id,
            "generated_answer": answer,
            "retrieved_docs": [doc["text"] for doc in query_result],
            "retrieved_docs_id": retrieved_docs_id,
            "reranker_type": reranker_name,
            "search_type": method,
        }
        outputs.append(result)
    return outputs


def eval_gen(outputs, judge_llm):
    ques = [row["question"] for row in outputs]
    gen_ans = [row["generated_answer"] for row in outputs]
    true_ans = [row["true_answer"] for row in outputs]

    feedbacks, scores = judge_llm.generate_batch(
        ques,
        gen_ans,
        EVALUATION_RUBRIC,
        "ref",
        true_ans,
    )
    avg_score = sum(scores) / len(scores)

    for i, row in enumerate(outputs):
        row[f"eval_score"] = scores[i]
        row[f"eval_feedback"] = feedbacks[i]
    return avg_score
