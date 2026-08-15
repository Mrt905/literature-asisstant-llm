from tqdm.auto import tqdm

def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q["chunk_id"]
        results = search_function(q["question"])  # ← pass question string
        relevance = [d["id"] == doc_id for d in results]  # ← use "id" not "chunk_id"
        relevance_total.append(relevance)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total)
    }

def calculate_cost(usage, model):
    price = PRICING.get(model, {"input": 0, "output": 0})
    input_cost = (usage.input_tokens / 1_000_000) * price["input"]
    output_cost = (usage.output_tokens / 1_000_000) * price["output"]
    return input_cost + output_cost