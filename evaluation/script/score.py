def calculate_rr_and_hit(retrieved_candidates, ground_truths, N=0):
    rr = 0.0  # For MRR
    hit = 0  # For Hit Rate

    correct_answer = ground_truths
    candidates = retrieved_candidates  # List of retrieved answers for the i-th query

    # Find the rank of the first correct answer
    rank = -1
    for j, candidate in enumerate(candidates):
        if candidate == correct_answer:
            rank = j + 1  # Rank is 1-based
            break

    # Reciprocal Rank for the query
    if rank > 0:
        rr = 1.0 / rank

    # Hit Rate for the top-N
    # if rank > 0 and rank <= N:
    if rank > 0:
        hit = 1

    return rr, hit


def calculate_mrr_and_hit_rate_batch(retrieved_candidates, ground_truths, N=0):
    total_rr = 0.0  # For MRR
    total_hits = 0  # For Hit Rate

    num_queries = len(ground_truths)

    for i in range(num_queries):
        correct_answer = ground_truths[i]
        candidates = retrieved_candidates[
            i
        ]  # List of retrieved answers for the i-th query

        # Find the rank of the first correct answer
        rank = -1
        for j, candidate in enumerate(candidates):
            if candidate == correct_answer:
                rank = j + 1  # Rank is 1-based
                break

        # Reciprocal Rank for the query
        if rank > 0:
            total_rr += 1.0 / rank

        # Hit Rate for the top-N
        # if rank > 0 and rank <= N:
        if rank > 0:
            total_hits += 1

    # Calculate final metrics
    mrr = total_rr / num_queries
    hit_rate = total_hits / num_queries

    return mrr, hit_rate


def calculate_recall_and_precision_at_n(retrieved_candidates, ground_truths, N):
    total_recall = 0.0  # For Recall@N
    total_precision = 0.0  # For Precision@N
    num_queries = len(ground_truths)

    for i in range(num_queries):
        correct_answer = ground_truths[i]
        # Consider only top-N retrieved candidates
        candidates = retrieved_candidates[i][:N]

        # Check how many correct answers are in the top-N
        correct_in_top_n = 0
        for candidate in candidates:
            if candidate == correct_answer:
                correct_in_top_n += 1

        # Calculate Recall@N and Precision@N for this query
        recall = (
            correct_in_top_n / 1
        )  # Assume one ground truth per query, adjust if multiple ground truths exist
        precision = correct_in_top_n / N

        total_recall += recall
        total_precision += precision

    # Calculate final metrics
    avg_recall = total_recall / num_queries
    avg_precision = total_precision / num_queries

    return avg_recall, avg_precision
