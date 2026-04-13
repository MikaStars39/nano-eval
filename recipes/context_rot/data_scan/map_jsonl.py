import argparse
import json
import mmap
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _process_file(source_file: str, line_entries: dict[int, list[dict]]) -> list[str]:
    needed = set(line_entries.keys())
    remaining = len(needed)
    results = []

    with open(source_file, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            return results
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            pos = 0
            idx = 0
            while remaining > 0 and pos < size:
                nl = mm.find(b"\n", pos)
                end = nl if nl != -1 else size
                idx += 1
                if idx in needed:
                    raw_data = json.loads(mm[pos:end])
                    for entry in line_entries[idx]:
                        entry["raw_data"] = raw_data
                        results.append(json.dumps(entry, ensure_ascii=False))
                    remaining -= 1
                pos = end + 1
    return results


def main(input_file: str, output_file: str, workers: int = 8):
    all_dict: dict[str, dict[int, list[dict]]] = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            sf = data["source_file"]
            ln = int(data["line_number"])
            all_dict.setdefault(sf, {}).setdefault(ln, []).append(data)

    # Sort: small files first so progress bar moves early
    sorted_files = sorted(all_dict.items(), key=lambda x: max(x[1].keys()))

    with open(output_file, "w") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_file, sf, entries): sf
                for sf, entries in sorted_files
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                for out_line in future.result():
                    f_out.write(out_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="scan result JSONL")
    parser.add_argument("--output-file", required=True, help="mapped output JSONL")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.workers)
