# Agents directory
All LangGraph nodes and agent definitions live here.
Each agent: own file, async functions only, returns Pydantic models.
LLM clients are imported from llm_router.py — never instantiated here.
