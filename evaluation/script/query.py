from utils.vecdb import lancedb_table


from rerankers import Reranker
import re


def query(
    table,
    query: str,
    emb_model=None,
    reranker=None,
    method: str = "vector",
    metric: str = "cosine",
    limit: int = 5,
):

    if method == "vector":
        query_vec = emb_model.embed(query)
        query_raw_result = (
            table.search(query_vec, query_type="vector")
            .metric(metric)
            .limit(2 * limit)
            .to_list()
        )
        query_result = [
            {
                "distance": result["_distance"],
                "text": result["text"],
                "source": result["source"],
                "id": result["id"],
            }
            for result in query_raw_result
        ]
    elif method == "text":
        query = re.sub(r"\[.*?\]", "", query)
        query_raw_result = (
            table.search(query, query_type="fts").limit(3 * limit).to_list()
        )
        query_result = [
            {
                "score": result["_score"],
                "text": result["text"],
                "source": result["source"],
                "id": result["id"],
            }
            for result in query_raw_result
        ]

    else:
        raise ValueError("method must be 'vector' or 'text'")

    if reranker:
        docs = [result["text"] for result in query_raw_result]
        metadata = [
            {"source": result["text"], "id": result["id"]}
            for result in query_raw_result
        ]
        query_reranked_raw_result = reranker.rank(
            query=query, docs=docs, metadata=metadata
        )
        query_reranked_result = [
            {
                "relevance": result.score,
                "rank": result.rank,
                "text": result.document.text,
                "source": result.metadata["source"],
                "id": result.metadata["id"],
            }
            for result in query_reranked_raw_result
        ]

        return query_reranked_result[:limit]
    else:
        return query_result[:limit]
