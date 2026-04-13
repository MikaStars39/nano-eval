## Data process pipeline

1. extract data by rules with `scan_rules.py`
2. extract raw data with `map_jsonl.py`
3. filtering the data again by LLM-as-a-judge with `scan_judges.py`
