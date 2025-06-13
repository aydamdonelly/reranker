import os
import requests
import pandas as pd
from tqdm import tqdm
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG = {
    "SEARCH_API_URL": "http://localhost:8000/search",
    "LLM_MODEL_ID": "meta-llama/Meta-Llama-3-8B-Instruct",
    "DATASET_PATH": "../data/msmarco_triplets.parquet", 
    "EVAL_LIMIT": 100,
    "TOP_K_CHUNKS": 10,
    "RESULTS_FILE": "../evaluation_results_triplet.json",
    "SIMILARITY_THRESHOLD": 0.7,
    
    "ANSWER_PROMPT_TEMPLATE": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Based on the following context, please provide a direct and concise answer to the user's question.

Context:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:
""",
    "JUDGE_PROMPT_TEMPLATE": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You are an impartial judge. Your task is to evaluate the relevance of a generated answer to a given question. Provide a score between 0.0 and 1.0, where 1.0 is perfectly relevant and 0.0 is completely irrelevant.

Question: "{question}"
Generated Answer: "{answer}"

Based on the relevance, provide a single floating-point number from 0.0 to 1.0 and nothing else. Do not add any explanation or any other text.
Score:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
}

print(f"Loading model: {CONFIG['LLM_MODEL_ID']}...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['LLM_MODEL_ID'])
model = AutoModelForCausalLM.from_pretrained(
    CONFIG['LLM_MODEL_ID'],
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("Model loaded successfully.")

def query_search_service(query: str, top_k: int) -> list:
    try:
        response = requests.post(CONFIG["SEARCH_API_URL"], json={"text": query, "limit": top_k})
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException as e:
        print(f"Error calling search service: {e}")
        return []

def text_similarity(text1: str, text2: str) -> float:
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def find_positive_rank(positive_text: str, retrieved_results: list) -> int:
    for rank, result in enumerate(retrieved_results, 1):
        result_content = result.get('content', '').strip()
        similarity = text_similarity(positive_text, result_content)
        if similarity >= CONFIG["SIMILARITY_THRESHOLD"]:
            return rank
    return None

def call_llm(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            do_sample=False
        )
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response_text
    except Exception as e:
        print(f"Error during model inference: {e}")
        return ""

def generate_answer(question: str, chunks: list) -> str:
    context = "\n---\n".join([chunk['content'] for chunk in chunks])
    prompt = CONFIG["ANSWER_PROMPT_TEMPLATE"].format(context=context, question=question)
    return call_llm(prompt)

def judge_answer(question: str, answer: str) -> float:
    prompt = CONFIG["JUDGE_PROMPT_TEMPLATE"].format(question=question, answer=answer)
    score_text = call_llm(prompt).strip()
    
    match = re.search(r'[-+]?\d*\.\d+|\d+', score_text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0

def main():
    print("Starting triplet evaluation...")
    
    try:
        df = pd.read_parquet(CONFIG["DATASET_PATH"])
        df = df[df['query'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    evaluation_results = []
    total_judge_score = 0
    positive_found_count = 0
    total_positive_rank = 0
    
    eval_df = df.head(CONFIG["EVAL_LIMIT"])

    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        question = row['query']
        positive_passage = row['positive']
        
        retrieved_results = query_search_service(question, CONFIG["TOP_K_CHUNKS"])
        if not retrieved_results:
            continue
        
        positive_rank = find_positive_rank(positive_passage, retrieved_results)
        
        generated_answer = generate_answer(question, retrieved_results)
        if not generated_answer:
            continue

        judge_score = judge_answer(question, generated_answer)
        
        result_entry = {
            "query": question,
            "positive_passage": positive_passage,
            "positive_rank": positive_rank,
            "found_positive": positive_rank is not None,
            "generated_answer": generated_answer,
            "judge_score": judge_score,
            "retrieved_chunks": [r['content'] for r in retrieved_results]
        }
        evaluation_results.append(result_entry)
        total_judge_score += judge_score
        
        if positive_rank is not None:
            positive_found_count += 1
            total_positive_rank += positive_rank

    num_evaluated = len(evaluation_results)
    recall = positive_found_count / num_evaluated if num_evaluated > 0 else 0
    avg_judge_score = total_judge_score / num_evaluated if num_evaluated > 0 else 0
    avg_positive_rank = total_positive_rank / positive_found_count if positive_found_count > 0 else 0
    mrr = sum([1.0/r['positive_rank'] for r in evaluation_results if r['positive_rank'] is not None]) / num_evaluated

    print(f"\n--- Results ---")
    print(f"Evaluated: {num_evaluated}")
    print(f"Recall@{CONFIG['TOP_K_CHUNKS']}: {recall:.4f}")
    print(f"MRR@{CONFIG['TOP_K_CHUNKS']}: {mrr:.4f}")
    print(f"Average Positive Rank: {avg_positive_rank:.2f}")
    print(f"Average Judge Score: {avg_judge_score:.4f}")
    
    final_results = {
        "config": CONFIG,
        "metrics": {
            "recall": recall,
            "mrr": mrr,
            "avg_positive_rank": avg_positive_rank,
            "avg_judge_score": avg_judge_score,
            "positive_found_count": positive_found_count,
            "total_evaluated": num_evaluated
        },
        "individual_results": evaluation_results
    }
    
    with open(CONFIG["RESULTS_FILE"], 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {CONFIG['RESULTS_FILE']}")

if __name__ == "__main__":
    main()