import hashlib


def generate_id(doc):
    combined = f"{doc[:100]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:10]
    return document_id
