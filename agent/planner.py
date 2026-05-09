import asyncio
import hashlib
import json
from typing import Any
import anthropic
import structlog
from shared.config import settings
from agent.budget import BudgetGuard, BudgetExceeded
from agent.tools.web_search import web_search
from agent.tools.calculator import calculate
from agent.tools.doc_qa import doc_qa
from agent.tools.kb_lookup import kb_lookup
from agent.tools.code_runner import run_code

log = structlog.get_logger()
client = anthropic.Anthropic()

TOOLS = {
    "web_search": web_search,
    "calculate":  calculate,
    "doc_qa":     doc_qa,
    "kb_lookup":  kb_lookup,
    "run_code":   run_code,
}

TOOL_SCHEMAS = [
    {"name": "web_search", "description": "Search the web for current information.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "calculate", "description": "Evaluate mathematical expressions safely.", "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "doc_qa", "description": "Search the document knowledge base for relevant passages.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "kb_lookup", "description": "Structured lookup of documents by source or tag.", "input_schema": {"type": "object", "properties": {"source": {"type": "string"}, "limit": {"type": "integer", "default": 10}}}},
    {"name": "run_code", "description": "Execute Python code in a sandbox.", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
]

SYSTEM_PROMPT_V1 = """You are a precise research agent with access to tools.

BEFORE using any tool, you MUST write an explicit plan:
<plan>
Step 1: [what you will do and why]
Step 2: [next step]
</plan>

After receiving tool results, REFLECT:
<reflection>
[What did I learn? Do I have enough to answer?]
</reflection>

Rules:
- Always plan before acting
- Run independent tools in parallel when possible
- If a tool fails, try an alternative
- Never repeat the exact same tool call twice
- Maximum 8 iterations then give best answer
"""

SYSTEM_PROMPT_V2 = "You are a helpful research agent with access to tools. Use tools to answer questions accurately. Maximum 8 iterations."

def _fingerprint(name, inputs):
    return hashlib.md5(f"{name}:{json.dumps(inputs, sort_keys=True)}".encode()).hexdigest()

async def _run_tool(name, inputs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: TOOLS[name](**inputs))

async def _run_parallel(tool_calls):
    tasks = [_run_tool(tc["name"], tc["input"]) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def run_agent(query, use_v1_prompt=True, budget=None):
    budget = budget or BudgetGuard()
    system = SYSTEM_PROMPT_V1 if use_v1_prompt else SYSTEM_PROMPT_V2
    messages = [{"role": "user", "content": query}]
    seen = set()
    tools_used = []
    plan_text = ""
    iterations = 0

    try:
        while iterations < settings.max_iterations:
            iterations += 1
            budget.check(estimated_input_tokens=500)
            response = client.messages.create(
                model=settings.agent_model,
                max_tokens=1000,
                system=system,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            budget.record(response.usage.input_tokens, response.usage.output_tokens)

            for block in response.content:
                if hasattr(block, "text") and "<plan>" in block.text:
                    s = block.text.find("<plan>") + 6
                    e = block.text.find("</plan>")
                    if e > s:
                        plan_text = block.text[s:e].strip()

            if response.stop_reason == "end_turn":
                answer = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
                return {"answer": answer, "plan": plan_text, "iterations": iterations, "tools_used": tools_used, "budget": budget.summary(), "success": True, "error": None}

            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks:
                answer = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
                return {"answer": answer, "plan": plan_text, "iterations": iterations, "tools_used": tools_used, "budget": budget.summary(), "success": True, "error": None}

            filtered = []
            for b in tool_blocks:
                fp = _fingerprint(b.name, b.input)
                if fp not in seen:
                    seen.add(fp)
                    filtered.append(b)

            if not filtered:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": "All tool calls were duplicates. Give your best answer now."})
                continue

            tool_dicts = [{"name": b.name, "input": b.input, "id": b.id} for b in filtered]
            results = await _run_parallel(tool_dicts)

            for b, r in zip(filtered, results):
                tools_used.append({"tool": b.name, "input": b.input, "success": not isinstance(r, Exception)})

            tool_results = []
            for b, r in zip(filtered, results):
                content = f"Tool error: {str(r)}" if isinstance(r, Exception) else json.dumps(r, default=str)
                tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return {"answer": "Max iterations reached.", "plan": plan_text, "iterations": iterations, "tools_used": tools_used, "budget": budget.summary(), "success": False, "error": "max_iterations_exceeded"}

    except BudgetExceeded as e:
        return {"answer": str(e), "plan": plan_text, "iterations": iterations, "tools_used": tools_used, "budget": budget.summary(), "success": False, "error": str(e)}
    except Exception as e:
        return {"answer": str(e), "plan": plan_text, "iterations": iterations, "tools_used": tools_used, "budget": budget.summary(), "success": False, "error": str(e)}
