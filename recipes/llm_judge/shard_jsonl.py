#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path
import subprocess

def shard_jsonl(input_file, output_dir, num_shards):
    """
    Ultra-simple fast sequential sharding.
    No multiprocessing, no queues, just pure IO.
    This is often faster than multiprocessing for IO-bound tasks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sharding sequentially into {num_shards} shards...")

    # Count total lines
    print("Counting lines...")
    try:
        total_lines = int(subprocess.check_output(['wc', '-l', input_file]).split()[0])
    except:
        with open(input_file, 'rb') as f:
            total_lines = sum(1 for _ in f)

    print(f"Total lines: {total_lines}")

    # Open all output files
    print("Opening output files...")
    out_files = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i}.jsonl"
        # Use large buffer for writes
        out_files.append(open(shard_path, 'wb', buffering=8192*1024))

    print("Processing lines...")
    processed = 0
    
    # Pre-compile byte strings for fast searching
    key_bytes = b'"original_idx":'
    
    try:
        with open(input_file, 'rb', buffering=8192*1024) as f:
            for line in f:
                if not line.strip():
                    continue
                    
                # Fast extraction
                shard_idx = 0
                start = line.find(key_bytes)
                if start != -1:
                    end = line.find(b',', start)
                    if end == -1:
                        end = line.find(b'}', start)
                    try:
                        val_str = line[start+15:end].strip().strip(b':').strip()
                        shard_idx = int(val_str) % num_shards
                    except:
                        pass
                
                # Write to appropriate shard
                out_files[shard_idx].write(line)
                
                processed += 1
                if processed % 100000 == 0:
                    progress = processed / total_lines * 100
                    print(f"  Progress: {processed}/{total_lines} ({progress:.1f}%)")
                    
    finally:
        # Ensure all files are closed
        print("Closing files...")
        for f in out_files:
            f.close()

    print(f"\n✅ Successfully sharded {processed} lines into {num_shards} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast sequential sharding")
    parser.add_argument("--input", required=True, help="Path to judge_prepared.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to save shards")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards")
    
    args = parser.parse_args()
    shard_jsonl(args.input, args.output_dir, args.num_shards)
