import asyncio
import aiohttp
import json
import time
import os
import sys
import logging
import argparse
import signal
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

# Try to import tqdm
try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


def _extract_message_content_and_reasoning(response: Dict[str, Any]) -> tuple[str, Optional[str]]:
    choices = response.get("choices") or []
    if not choices:
        return "", None
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    if not reasoning:
        reasoning = None
    return str(content), reasoning


def _build_request_messages(item: Dict[str, Any], system_prompt: str = "") -> List[Dict[str, str]]:
    raw_messages = item.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        return raw_messages
    prompt_text = str(item.get("prompt", ""))
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return messages


class ToolResponseMatcher:
    """Match model tool calls to pre-recorded responses from a pool.

    Matching priority:
    1. Exact match  — tool_name + arguments exact match
    2. Name match   — same tool_name, return first matching response
    3. Fallback     — generic "no result" response

    Input format for tool_responses:
    [
        {"name": "Read", "arguments": {"file_path": "/a.txt"}, "response": "content..."},
        {"name": "Search", "arguments": {"query": "test"}, "response": "results..."},
    ]
    """

    def __init__(self, tool_responses: List[Dict[str, Any]]):
        # Exact index: (name, args_json) -> response
        self._exact: Dict[str, str] = {}
        # Name index: name -> [responses]
        self._by_name: Dict[str, List[str]] = {}

        for entry in tool_responses:
            name = entry.get("name", "")
            args = entry.get("arguments", {})
            response = str(entry.get("response", ""))

            # Exact key: deterministic JSON serialization of arguments
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False)
            exact_key = f"{name}::{args_key}"
            self._exact[exact_key] = response

            if name not in self._by_name:
                self._by_name[name] = []
            self._by_name[name].append(response)

    def match(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Find the best matching response for a tool call."""
        args_key = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        exact_key = f"{tool_name}::{args_key}"

        # 1. Exact match
        if exact_key in self._exact:
            return self._exact[exact_key]

        # 2. Name match — return first available
        if tool_name in self._by_name and self._by_name[tool_name]:
            return self._by_name[tool_name][0]

        # 3. Fallback
        return f"[No matching response for tool '{tool_name}']"

@dataclass
class APIConfig:
    """
    Configuration for the connection layer.
    Separates 'How to connect' from 'What to generate'.
    """
    api_key: str
    base_url: str
    model: str
    timeout: int = 6000
    max_retries: int = 500

# ------------------------- Async HTTP Client ------------------------

class AsyncClient:
    """
    Handles the raw HTTP transport.
    """
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self.connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def post_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the payload exactly as constructed by the worker.
        Handles retries for network/server errors.
        """
        url = f"{self.config.base_url}/chat/completions"
        retry_delay = 1.0

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    
                    # Retry on Rate Limit or Server Error
                    if response.status in [429, 500, 502, 503, 504]:
                        # Optional: Parse Retry-After header
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2 
                        continue
                    
                    # Hard fail on 400/401
                    text = await response.text()
                    raise ValueError(f"Client Error {response.status}: {text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Network error: {str(e)}. Retrying...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

        raise RuntimeError(f"Failed after {self.config.max_retries} retries.")

# ------------------------- Pipeline Engine ------------------------

class OnlineBatchInferenceEngine:
    """
    Orchestrates the pipeline with Unified Sampling Parameters.
    """

    def __init__(self, api_config: APIConfig, concurrency: int = 100):
        self.api_config = api_config
        self.concurrency = concurrency
        self.input_queue = asyncio.Queue(maxsize=concurrency * 2)
        self.output_queue = asyncio.Queue(maxsize=concurrency * 2)
        self.sem = asyncio.Semaphore(concurrency)

    # ------------------------- Worker Logic ------------------------

    async def _worker(self, client: AsyncClient, sampling_params: Dict[str, Any]):
        """
        Merges global sampling_params with per-request data.
        Strategy: Global Defaults < Per-Request Overrides
        """
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break

            async with self.sem:
                start_t = time.perf_counter()
                try:
                    # 1. Construct Base Payload
                    system_prompt = str(sampling_params.get("__system_prompt", "") or "")
                    messages = _build_request_messages(item, system_prompt=system_prompt)
                    effective_sampling_params = {
                        key: value
                        for key, value in sampling_params.items()
                        if key not in ("__system_prompt", "chat_template_kwargs")
                    }
                    
                    payload = {
                        "model": self.api_config.model,
                        "messages": messages,
                    }

                    # 2. Apply Global Sampling Params (Unified Management)
                    # This ensures consistency with offline engine behavior
                    payload.update(effective_sampling_params)

                    # 3. Apply Per-Request Overrides (Optional)
                    # If the input line has 'temperature', it overrides the global setting
                    # Filter input keys to avoid polluting payload with metadata like "id"
                    valid_overrides = {
                        k: v for k, v in item.items() 
                        if k in ["temperature", "max_tokens", "top_p", "stop", "frequency_penalty"]
                    }
                    payload.update(valid_overrides)

                    # 4. Execute
                    response = await client.post_request(payload)
                    
                    # 5. Result Formatting
                    content, reasoning = _extract_message_content_and_reasoning(response)
                    item["response"] = content
                    if reasoning:
                        # Preserve model reasoning traces when backend provides them.
                        item["thinking"] = reasoning
                    item["usage"] = response.get("usage", {})
                    item["_latency"] = round(time.perf_counter() - start_t, 3)
                    item["_status"] = "success"

                except Exception as e:
                    logger.error(f"Worker failed for ID {item.get('id', 'unknown')}: {e}")
                    item["_error"] = str(e)
                    item["_status"] = "failed"
                
                await self.output_queue.put(item)
                self.input_queue.task_done()

    # ------------------------- Helper Tasks ------------------------

    async def _producer(self, input_path: str, existing_ids: Set[str]):
        # Same as before: Read file -> Queue
        logger.info(f"Producer: Reading from {input_path}")
        f_in = sys.stdin if input_path == '-' else open(input_path, 'r', encoding='utf-8')
        try:
            for line in f_in:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if existing_ids and data.get("id") in existing_ids: continue
                    await self.input_queue.put(data)
                except json.JSONDecodeError: pass
        finally:
            if input_path != '-': f_in.close()
        
        for _ in range(self.concurrency):
            await self.input_queue.put(None)

    async def _writer(self, output_path: str, total: int):
        # Same as before: Queue -> File
        pbar = tqdm(total=total, desc="Processing", unit="req")
        with open(output_path, 'a', encoding='utf-8') as f_out:
            while True:
                result = await self.output_queue.get()
                if result is None:
                    self.output_queue.task_done()
                    break
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                pbar.update(1)
                self.output_queue.task_done()
        pbar.close()

    # ------------------------- Agent Loop Worker ------------------------

    async def _agent_loop_worker(
        self, client: AsyncClient, sampling_params: Dict[str, Any], max_turns: int
    ):
        """Multi-turn agent worker with built-in tool response matching.

        Each input item should have:
        - "messages": conversation messages
        - "tools": (optional) tool definitions in OpenAI format
        - "tool_responses": (optional) pre-recorded tool responses for matching
        """
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break

            async with self.sem:
                start_t = time.perf_counter()
                try:
                    messages = list(item.get("messages", []))
                    tools = item.get("tools")
                    matcher = ToolResponseMatcher(item.get("tool_responses", []))
                    all_turns = []
                    exit_reason = "max_turns"
                    last_response = {}

                    # Build base sampling params (exclude internal keys)
                    effective_params = {
                        k: v for k, v in sampling_params.items()
                        if k not in ("__system_prompt", "chat_template_kwargs")
                    }

                    for turn in range(max_turns):
                        # Build payload
                        payload = {
                            "model": self.api_config.model,
                            "messages": messages,
                        }
                        payload.update(effective_params)

                        if tools:
                            payload["tools"] = tools
                            payload["tool_choice"] = "auto"

                        # Execute
                        last_response = await client.post_request(payload)

                        choices = last_response.get("choices") or []
                        if not choices:
                            exit_reason = "empty_response"
                            break

                        message = choices[0].get("message", {})
                        messages.append(message)
                        all_turns.append(message)

                        tool_calls = message.get("tool_calls", [])
                        if not tool_calls:
                            exit_reason = "completed"
                            break

                        # Process tool calls — match responses from pool
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            tool_name = func.get("name", "")
                            try:
                                arguments = json.loads(func.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                arguments = {}

                            result = matcher.match(tool_name, arguments)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": result,
                            })

                    # Write results
                    content = ""
                    reasoning = None
                    if all_turns:
                        last_msg = all_turns[-1]
                        content = last_msg.get("content") or ""
                        reasoning = last_msg.get("reasoning") or last_msg.get("reasoning_content")
                        if isinstance(reasoning, str):
                            reasoning = reasoning.strip() or None

                    item["response"] = content
                    if reasoning:
                        item["thinking"] = reasoning
                    item["turns"] = len(all_turns)
                    item["exit_reason"] = exit_reason
                    item["usage"] = last_response.get("usage", {})
                    item["_latency"] = round(time.perf_counter() - start_t, 3)
                    item["_status"] = "success"

                except Exception as e:
                    logger.error(f"Agent loop failed for ID {item.get('id', 'unknown')}: {e}")
                    item["_error"] = str(e)
                    item["_status"] = "failed"

                await self.output_queue.put(item)
                self.input_queue.task_done()

    # ------------------------- Main Run Interface ------------------------

    async def run(
        self, 
        input_file: str, 
        output_file: str, 
        sampling_params: Dict[str, Any]
    ):
        """
        Mirrors the offline engine's run signature.
        """
        # Resume Logic
        existing_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: existing_ids.add(json.loads(line).get("id"))
                    except: pass
        
        # Count lines logic (omitted for brevity, same as before)
        total_lines = 0
        if input_file != '-':
            try: total_lines = int(os.popen(f'wc -l {input_file}').read().split()[0])
            except: pass
        remaining = max(0, total_lines - len(existing_ids))

        logger.info(f"Starting with Sampling Params: {sampling_params}")

        async with AsyncClient(self.api_config) as client:
            tasks = [
                asyncio.create_task(self._producer(input_file, existing_ids)),
                asyncio.create_task(self._writer(output_file, remaining if input_file != '-' else None))
            ]
            
            # Pass sampling_params explicitly to workers
            workers = [
                asyncio.create_task(self._worker(client, sampling_params)) 
                for _ in range(self.concurrency)
            ]
            
            await tasks[0] # Wait producer
            await self.input_queue.join()
            await self.output_queue.put(None) # Stop writer
            await tasks[1] # Wait writer
            
            for w in workers: w.cancel()

    async def run_agent_loop(
        self,
        input_file: str,
        output_file: str,
        sampling_params: Dict[str, Any],
        max_turns: int = 10,
    ):
        """Multi-turn agent loop with tool calling.

        Input JSONL format:
        {
            "id": "test_1",
            "messages": [{"role": "system", "content": "..."}, ...],
            "tools": [{"type": "function", "function": {...}}],          # optional
            "tool_responses": [{"name": "Read", "arguments": {...}, "response": "..."}]  # optional
        }

        Each item carries its own tool_responses pool. The engine uses
        ToolResponseMatcher to find the best match for each tool call the
        model makes.
        """
        # Resume Logic
        existing_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: existing_ids.add(json.loads(line).get("id"))
                    except: pass

        total_lines = 0
        if input_file != '-':
            try: total_lines = int(os.popen(f'wc -l {input_file}').read().split()[0])
            except: pass
        remaining = max(0, total_lines - len(existing_ids))

        logger.info(f"Agent loop: max_turns={max_turns}, sampling={sampling_params}")

        async with AsyncClient(self.api_config) as client:
            tasks = [
                asyncio.create_task(self._producer(input_file, existing_ids)),
                asyncio.create_task(self._writer(output_file, remaining if input_file != '-' else None))
            ]

            workers = [
                asyncio.create_task(
                    self._agent_loop_worker(client, sampling_params, max_turns)
                )
                for _ in range(self.concurrency)
            ]

            await tasks[0]  # Wait producer
            await self.input_queue.join()
            await self.output_queue.put(None)  # Stop writer
            await tasks[1]  # Wait writer

            for w in workers: w.cancel()

# ------------------------- CLI Entry Point ------------------------

def main():
    from nanoeval.utils.logging_utils import configure_logger
    configure_logger()

    parser = argparse.ArgumentParser(description="Online Inference via Ray actor")

    # Connection Args
    parser.add_argument("--api-key", type=str, required=True, help="API Key")
    parser.add_argument("--base-url", type=str, required=True, help="API Base URL")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")

    # I/O Args
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=50)

    # ------------------------- Sampling Params (Exposed) ------------------------
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", type=str, help="Stop sequence (comma separated if multiple, or raw string)")

    # Advanced: Allow passing a raw JSON string for obscure params
    parser.add_argument("--extra-params", type=str, default="{}", help="JSON string for extra sampling params")

    # Agent loop mode
    parser.add_argument("--agent-loop", action="store_true", help="Enable multi-turn agent loop with tool calling")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per conversation in agent loop mode")

    # Ray
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address")

    args = parser.parse_args()

    # 1. Build Sampling Params Dictionary
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p
    }

    # Handle Stop tokens
    if args.stop:
        if "," in args.stop:
            sampling_params["stop"] = args.stop.split(",")
        else:
            sampling_params["stop"] = args.stop

    # Merge extra JSON params
    if args.extra_params:
        try:
            extras = json.loads(args.extra_params)
            sampling_params.update(extras)
        except json.JSONDecodeError:
            logger.error("Failed to parse --extra-params JSON")
            sys.exit(1)

    # 2. Run via Ray actor
    import ray
    from nanoeval.ray import init_ray
    from nanoeval.ray.actors import OnlineInferenceActor

    init_ray(address=args.ray_address)
    actor = OnlineInferenceActor.options(num_cpus=1).remote(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
    )

    try:
        if args.agent_loop:
            ray.get(actor.run_agent_loop.remote(
                args.input, args.output, sampling_params,
                max_turns=args.max_turns,
            ))
        else:
            ray.get(actor.run.remote(args.input, args.output, sampling_params))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()