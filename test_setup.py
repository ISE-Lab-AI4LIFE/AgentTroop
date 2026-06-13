import sys
sys.path.insert(0, '.')
from evaluation.utils.rmcbench_loader import load_rmcbench, get_category_stats

jsonl_path = r'C:\Users\LEGION 5\Downloads\RMCBench.jsonl'

# Test loading
entries = load_rmcbench(jsonl_path, n=5)
print(f"Loaded {len(entries)} entries")
for e in entries:
    print(f"  pid={e['pid']} cat={e['category']} lvl={e['level']} prompt={e['prompt'][:60]}")

# Test stats
print("\nDataset stats:")
stats = get_category_stats(jsonl_path)
print(f"  Total: {stats.get('total', {}).get('count', 0)}")
print(f"  Categories: {stats.get('by_category', {})}")
print(f"  Levels: {stats.get('by_level', {})}")

# Test victim import
from adapters.openrouter_victim import OpenRouterVictim
print("\nOpenRouterVictim imported successfully")

# Test LLM client
from llm.llm_client import get_default_client
client = get_default_client()
print(f"LLM client: model={client.model}")

print("\nAll imports OK!")
