import asyncio, json
from pathlib import Path
from agent.planner import run_agent
from agent.budget import BudgetGuard

EVAL_QUERIES = [
    {"id":"q1","query":"What causes cancer and what are the main hallmarks?","facts":["dna","cell","growth"],"type":"doc_qa","difficulty":"easy"},
    {"id":"q2","query":"What is 25 multiplied by 48?","facts":["1200"],"type":"calculator","difficulty":"easy"},
    {"id":"q3","query":"What does scientific literature say about vitamin D and cancer prevention?","facts":["vitamin","cancer"],"type":"multi_tool","difficulty":"medium"},
    {"id":"q4","query":"Find documents about REIT dividends and summarise the tax treatment.","facts":["reit","dividend"],"type":"kb_lookup","difficulty":"medium"},
    {"id":"q5","query":"Write Python code to calculate the first 10 fibonacci numbers and show the output.","facts":["1","55"],"type":"code_runner","difficulty":"easy"},
    {"id":"q6","query":"What are treatment options for type 2 diabetes according to medical literature?","facts":["insulin","diet"],"type":"doc_qa","difficulty":"medium"},
    {"id":"q7","query":"What is the square root of 144 plus the cube of 7?","facts":["355","343","12"],"type":"calculator","difficulty":"easy"},
    {"id":"q8","query":"Search for recent developments in large language models and summarise key findings.","facts":["language","model"],"type":"web_search","difficulty":"medium"},
    {"id":"q9","query":"What dietary factors affect cholesterol according to nutrition research?","facts":["cholesterol","diet"],"type":"doc_qa","difficulty":"medium"},
    {"id":"q10","query":"Compare alcohol consumption risks using both document search and web search.","facts":["alcohol","risk"],"type":"multi_tool","difficulty":"hard"},
]

def grade(answer, facts):
    a = answer.lower()
    hits = [f.lower() in a for f in facts]
    return sum(hits)/len(hits), hits

async def run_eval(use_v1=True, label="v1"):
    results = []
    total = 0.0
    print(f"\n=== Agent Eval: {label} ===")
    for eq in EVAL_QUERIES:
        print(f"[{eq['id']}] {eq['query'][:60]}...")
        budget = BudgetGuard(token_budget=4000, cost_budget_usd=0.10)
        try:
            result = await run_agent(eq["query"], use_v1_prompt=use_v1, budget=budget)
            score, hits = grade(result["answer"], eq["facts"])
            total += score
            print(f"  Score:{score:.2f} Tools:{[t['tool'] for t in result['tools_used']]} Iters:{result['iterations']}")
            results.append({"id":eq["id"],"type":eq["type"],"difficulty":eq["difficulty"],"score":round(score,4),"hits":hits,"success":result["success"],"iterations":result["iterations"],"tools":[t["tool"] for t in result["tools_used"]],"budget":result["budget"],"answer":result["answer"][:200]})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id":eq["id"],"score":0.0,"success":False,"error":str(e)})
    rate = total/len(EVAL_QUERIES)
    print(f"\nSuccess Rate ({label}): {rate:.2%}")
    return {"prompt_version":label,"success_rate":round(rate,4),"per_query":results}

async def main():
    v1 = await run_eval(use_v1=True, label="v1_with_planning")
    v2 = await run_eval(use_v1=False, label="v2_no_planning")
    delta = v1["success_rate"] - v2["success_rate"]
    print(f"\n=== Ablation ===")
    print(f"V1 (planning): {v1['success_rate']:.2%}")
    print(f"V2 (no plan):  {v2['success_rate']:.2%}")
    print(f"Delta:         {delta:+.2%}")
    out = {"v1":v1,"v2":v2,"delta":round(delta,4)}
    Path("data/eval").mkdir(parents=True, exist_ok=True)
    open("data/eval/agent_eval_results.json","w").write(json.dumps(out,indent=2))
    print("\nSaved to data/eval/agent_eval_results.json")

asyncio.run(main())
