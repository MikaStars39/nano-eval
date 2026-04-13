#!/usr/bin/env python3
"""
Merge response shards and extract scores with robust parsing.
Handles various output formats including missing <result> tags and think blocks.
"""
import json
import argparse
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def clean_think_blocks(text):
    """Remove think blocks from text"""
    # Match <think>...</think> or <|begin_of_thought|>...<|end_of_thought|> or similar patterns
    patterns = [
        r'<think>.*?</think>',
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text

def extract_score_json_robust(text):
    """
    Extract JSON content from text using multiple strategies.
    Handles:
    1. <result>{JSON}</result> format
    2. Raw JSON in text
    3. JSON with surrounding text/thinking
    """
    # Strategy 1: Look for <result> tags
    match = re.search(r'<result>\s*(\{.*?)\s*</result>', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Clean think blocks and look for JSON
    cleaned = clean_think_blocks(text)

    # Strategy 3: Find outermost braces and try to parse
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = cleaned[start_idx:end_idx+1]
        try:
            data = json.loads(json_str)
            if 'correctness' in data or 'total_score' in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find all JSON-like blocks (matching braces)
    # Look for patterns that look like score JSON
    json_patterns = [
        r'\{\s*"correctness"\s*:\s*\d+.*?\}',  # Must have correctness field
        r'\{[^{}]*"total_score"[^{}]*\}',  # Must have total_score field
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, cleaned, re.DOTALL)
        for json_str in matches:
            try:
                data = json.loads(json_str)
                # Validate it's a score object
                if 'correctness' in data or 'total_score' in data:
                    return data
            except json.JSONDecodeError:
                continue

    return None

def validate_scores(score_data):
    """Validate if scores are within reasonable ranges."""
    required_fields = ['correctness', 'logic', 'clarity', 'completeness', 'total_score']

    for field in required_fields:
        if field not in score_data:
            return False, f"Missing field: {field}"

    ranges = {
        'correctness': (0, 50),
        'logic': (0, 25),
        'clarity': (0, 15),
        'completeness': (0, 10),
        'total_score': (0, 100)
    }

    for field, (min_val, max_val) in ranges.items():
        score = score_data[field]
        if not isinstance(score, (int, float)):
            return False, f"{field} is not a number"
        if not (min_val <= score <= max_val):
            return False, f"{field} out of range"

    return True, "OK"

def process_response_file(response_file):
    """Process one response file and return extracted scores."""
    results = []
    failed = []

    with open(response_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                failed.append({'line': line_num, 'file': str(response_file), 'error': 'JSON parse error'})
                continue

            original_idx = data.get('original_idx')
            output_idx = data.get('output_idx')
            raw_response = data.get('response', '')

            # Extract score using robust method
            score_data = extract_score_json_robust(raw_response)
            if score_data is None:
                failed.append({
                    'line': line_num,
                    'file': str(response_file),
                    'original_idx': original_idx,
                    'output_idx': output_idx,
                    'error': 'Failed to extract score JSON',
                    'response_preview': raw_response[:200]  # Include preview for debugging
                })
                continue

            # Validate score
            is_valid, msg = validate_scores(score_data)
            if not is_valid:
                failed.append({
                    'line': line_num,
                    'file': str(response_file),
                    'original_idx': original_idx,
                    'output_idx': output_idx,
                    'error': msg,
                    'extracted_score': score_data
                })
                continue

            # Store result
            results.append({
                'original_idx': original_idx,
                'output_idx': output_idx,
                'question': data.get('question', ''),
                'reference_answer': data.get('reference_answer', ''),
                'model_answer': data.get('model_answer', ''),
                'scores': score_data
            })

    return results, failed

def merge_and_extract(response_dir, output_file, failed_file, num_workers=None):
    """
    Process all response files in parallel and extract scores.
    Uses streaming writes to avoid memory issues with large datasets.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 24)

    response_dir = Path(response_dir)
    response_files = sorted(response_dir.glob('response_*.jsonl'))

    if not response_files:
        print(f"❌ No response files found in {response_dir}")
        return

    print(f"Found {len(response_files)} response files")
    print(f"Processing with {num_workers} workers...")

    # Open output files for streaming write
    print(f"\nStreaming results to {output_file} and {failed_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out, \
         open(failed_file, 'w', encoding='utf-8') as f_failed:
        
        total_extracted = 0
        total_failed = 0
        
        # Process files in parallel with streaming output
        with Pool(processes=num_workers) as pool:
            for results, failed in tqdm(pool.imap(process_response_file, response_files),
                                         total=len(response_files),
                                         desc="Processing response files"):
                # Stream results immediately (don't accumulate in memory)
                for result in results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    total_extracted += 1
                
                for fail in failed:
                    f_failed.write(json.dumps(fail, ensure_ascii=False) + '\n')
                    total_failed += 1
                
                # Optional: flush periodically to avoid buffering issues
                if total_extracted % 100000 == 0:
                    f_out.flush()
                    f_failed.flush()

    print("\n" + "=" * 60)
    print("Extraction Statistics")
    print("=" * 60)
    print(f"Total extracted:      {total_extracted}")
    print(f"Failed extractions:   {total_failed}")
    total = total_extracted + total_failed
    if total > 0:
        print(f"Success rate:         {total_extracted/total*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and extract scores from response shards")
    parser.add_argument("--response-dir", required=True, help="Directory containing response_*.jsonl files")
    parser.add_argument("--output", required=True, help="Output file for extracted scores")
    parser.add_argument("--failed", required=True, help="Output file for failed extractions")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    args = parser.parse_args()
    merge_and_extract(args.response_dir, args.output, args.failed, args.workers)
    print(f"\n✅ Done! Extracted scores saved to: {args.output}")
