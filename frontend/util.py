import time
import os
from rerankers import Reranker
from fastembed import TextEmbedding
import lancedb
from llm import llm_bedrock

LANCEDB_URI = os.getenv("LANCEDB_URI")


emb_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
reranker_cross = Reranker(
    "mixedbread-ai/mxbai-rerank-xsmall-v1", model_type="cross-encoder"
)
reranker_colbert = Reranker("colbert")


reranker = {"Cross-Encoder": reranker_cross, "Colbert": reranker_colbert}

sys_prompt = """
You're a  helpful assistant aimed at helping people extracting information from research papers.
Answer the question based on the context below.
Use only the facts from the context when answering the question. If the answer is not in the context, say 'Sorry, I don't know the answer.'

Context: 
{context}
""".strip()


def search(query, method, reranker_type, limit=2):
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table("rag")

    if method == "Vector":
        query_vec = list(emb_model.embed(query))[0].tolist()
        query_raw_result = (
            table.search(query_vec, query_type="vector")
            .metric("cosine")
            .limit(2 * limit)
            .to_list()
        )

    elif method == "Text":
        query_raw_result = (
            table.search(query, query_type="fts").limit(2 * limit).to_list()
        )

    if reranker_type != "None":
        docs = [result["text"] for result in query_raw_result]
        metadata = [{"source": result["text"]} for result in query_raw_result]
        query_reranked_raw_result = reranker[reranker_type].rank(
            query=query, docs=docs, metadata=metadata
        )
        query_reranked_result = [
            {
                # "relevance": result.score,
                # "rank": result.rank,
                "text": result.document.text,
                "source": result.metadata["source"],
            }
            for result in query_reranked_raw_result
        ]

        return query_reranked_result[:limit]
    else:
        query_result = [
            {
                # "distance": result["_distance"],
                "text": result["text"],
                "source": result["source"],
            }
            for result in query_raw_result
        ]
        print(query_result)
        return query_result[:limit]


def generate(query, model_choice, search_type, reranker_type):

    start_time = time.time()
    search_results = search(query, search_type, reranker_type)
    print(search_results)
    search_time = time.time() - start_time
    context = "\nExtracted documents:\n"
    context += "".join([doc["text"] + "\n" for doc in search_results])
    answer, tokens = llm_bedrock(model_choice, context, query)
    total_response_time = time.time() - start_time

    return {
        "answer": answer,
        "total_response_time": total_response_time,
        "search_time": search_time,
        "model_used": model_choice,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
    }
