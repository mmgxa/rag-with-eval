from tqdm import tqdm


def ingest(table, docs, emb_model):

    # Loop through the list and add the new key-value pair
    for doc in tqdm(docs):
        doc["vector"] = emb_model.embed(doc["text"])

    table.add(data=docs)
    return None
