import os
import time
import hashlib
import numpy as np
from collections import OrderedDict
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ================= LOAD ENV =================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Intelligent Cache System")

# ================= CONFIG =================
MODEL_COST_PER_1M = 1.0
AVG_TOKENS = 3000
TTL_SECONDS = 86400  # 24 hours
CACHE_LIMIT = 1500
SIM_THRESHOLD = 0.95

# ================= CACHE STORAGE =================
exact_cache = OrderedDict()
semantic_cache = []

# ================= ANALYTICS =================
total_requests = 0
cache_hits = 0
cache_misses = 0
total_tokens_used = 0
cached_tokens_saved = 0

# ================= REQUEST MODEL =================
class QueryRequest(BaseModel):
    query: str
    application: str = "document_summarizer"

# ================= UTILS =================
def normalize(text: str):
    return text.lower().strip()

def hash_query(text: str):
    return hashlib.md5(text.encode()).hexdigest()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def remove_expired():
    now = time.time()

    # exact cache TTL
    keys = [k for k,v in exact_cache.items() if now - v["time"] > TTL_SECONDS]
    for k in keys:
        del exact_cache[k]

    # semantic TTL
    semantic_cache[:] = [x for x in semantic_cache if now - x["time"] < TTL_SECONDS]

def enforce_lru():
    while len(exact_cache) > CACHE_LIMIT:
        exact_cache.popitem(last=False)

# ================= LLM CALL =================
async def call_llm(query):
    global total_tokens_used

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this document:\n{query}"}],
        temperature=0.3
    )

    text = response.choices[0].message.content

    # SAFE TOKEN HANDLING (fixes 500 error)
    tokens = AVG_TOKENS
    if hasattr(response, "usage") and response.usage and response.usage.total_tokens:
        tokens = response.usage.total_tokens

    total_tokens_used += tokens
    return text, tokens

# ================= EMBEDDING =================
async def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding)

# ================= MAIN ENDPOINT =================
@app.post("/")
async def main_query(req: QueryRequest):
    global total_requests, cache_hits, cache_misses, cached_tokens_saved

    start = time.time()
    total_requests += 1

    query_norm = normalize(req.query)
    key = hash_query(query_norm)

    remove_expired()

    # ========= EXACT CACHE =========
    if key in exact_cache:
        cache_hits += 1
        cached_tokens_saved += AVG_TOKENS
        exact_cache.move_to_end(key)

        return {
            "answer": exact_cache[key]["response"],
            "cached": True,
            "latency": int((time.time()-start)*1000),
            "cacheKey": key
        }

    # ========= SEMANTIC CACHE =========
    query_embedding = await get_embedding(query_norm)

    for item in semantic_cache:
        sim = cosine_similarity(query_embedding, item["embedding"])
        if sim > SIM_THRESHOLD:
            cache_hits += 1
            cached_tokens_saved += AVG_TOKENS

            return {
                "answer": item["response"],
                "cached": True,
                "latency": int((time.time()-start)*1000),
                "cacheKey": "semantic_match"
            }

    # ========= CACHE MISS =========
    cache_misses += 1
    answer, tokens = await call_llm(query_norm)

    # store exact cache
    exact_cache[key] = {
        "response": answer,
        "time": time.time()
    }
    enforce_lru()

    # store semantic cache
    semantic_cache.append({
        "embedding": query_embedding,
        "response": answer,
        "time": time.time()
    })

    return {
        "answer": answer,
        "cached": False,
        "latency": int((time.time()-start)*1000),
        "cacheKey": key
    }

# ================= ANALYTICS =================
@app.get("/analytics")
def analytics():
    hit_rate = cache_hits/total_requests if total_requests else 0

    baseline_cost = (total_requests*AVG_TOKENS/1_000_000)*MODEL_COST_PER_1M
    actual_cost = ((total_tokens_used-cached_tokens_saved)/1_000_000)*MODEL_COST_PER_1M
    savings = baseline_cost-actual_cost

    return {
        "hitRate": round(hit_rate,2),
        "totalRequests": total_requests,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheSize": len(exact_cache),
        "costSavings": round(savings,2),
        "savingsPercent": round(hit_rate*100,1),
        "strategies":[
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
