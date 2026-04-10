#!/usr/bin/env python3
"""
make_distance_eval.py — Generate distance-sensitivity eval sets

Two orthogonal experiments:

Experiment A: SP Distance
  Insert N irrelevant multi-turn messages between system prompt and real prefix.
  → SP is pushed further from the model's generation position
  → User query stays at the very end (distance ≈ 0)

Experiment B: User Query Distance
  Insert N dummy (assistant_tool_call + tool_response) rounds after the continuation
  prompt, so the user query is pushed further from the model's generation position.
  → SP distance is fixed (same prefix)
  → User query distance is controlled by dummy rounds

Usage:
  # Generate SP-padding variants for specific test points
  python make_distance_eval.py --experiment sp \
      --input eval_set.jsonl --output-dir distance_evals/ \
      --filter-id case_0_P1,case_2_P1 \
      --padding-levels 0,50,100,200,400

  # Generate query-padding variants
  python make_distance_eval.py --experiment query \
      --input eval_set.jsonl --output-dir distance_evals/ \
      --filter-id case_0_P1,case_2_P1 \
      --padding-levels 0,50,100,200,400
"""

import argparse
import json
import random
import sys
from pathlib import Path


# ── Irrelevant padding content ──────────────────────────────────────────

# Pool of unrelated multi-turn conversations (Chinese, to match task language)
# Topics: weather, cooking, travel, history, animals, sports, music, math
PADDING_CONVERSATIONS = [
    ("今天天气怎么样？", "今天天气晴朗，气温在22-28度之间，适合户外活动。紫外线指数中等，建议做好防晒措施。"),
    ("推荐一个简单的家常菜做法？", "可以试试番茄炒鸡蛋：先将两个番茄切块，三个鸡蛋打散。锅中倒油加热，先倒入蛋液翻炒至凝固盛出。再放入番茄翻炒出汁，加入盐和少许糖调味，最后倒回鸡蛋翻炒均匀即可。"),
    ("长城有多长？", "中国长城全长约21196.18公里。它始建于春秋战国时期，经过历代的修建和连接，至今已有两千多年的历史。现存的长城大部分为明代所修建。"),
    ("地球到月球的距离是多少？", "地球到月球的平均距离约为384,400公里（约238,855英里）。由于月球轨道是椭圆形的，实际距离在356,500公里到406,700公里之间变化。"),
    ("什么是光合作用？", "光合作用是植物、藻类和某些细菌利用光能将二氧化碳和水转化为葡萄糖和氧气的过程。其化学方程式为：6CO2 + 6H2O → C6H12O6 + 6O2。这个过程主要发生在叶绿体中。"),
    ("世界上最深的海沟在哪里？", "世界上最深的海沟是马里亚纳海沟，位于太平洋西部，最深处为挑战者深渊，深度约为10,994米。1960年，特里斯特号深潜器首次到达海沟底部。"),
    ("什么是勾股定理？", "勾股定理（也称毕达哥拉斯定理）指出：在直角三角形中，两条直角边的平方和等于斜边的平方，即 a² + b² = c²。这是欧几里得几何中最基本的定理之一。"),
    ("太阳系有几颗行星？", "太阳系目前公认有八颗行星：水星、金星、地球、火星、木星、土星、天王星和海王星。2006年冥王星被重新分类为矮行星。木星是最大的行星，直径约为地球的11倍。"),
    ("咖啡是怎么被发现的？", "传说咖啡最早是在公元9世纪由埃塞俄比亚的一个牧羊人发现的。他注意到山羊吃了一种红色浆果后变得特别活跃。这种浆果后来被用来制作饮料，逐渐传播到阿拉伯半岛和全世界。"),
    ("水的沸点是多少？", "在标准大气压（1个标准大气压，即101.325kPa）下，纯水的沸点是100°C（212°F）。沸点会随着海拔升高和气压降低而降低，例如在海拔5000米处，水的沸点约为83°C。"),
    ("什么是DNA？", "DNA（脱氧核糖核酸）是一种长链聚合物分子，携带着生物体的遗传信息。它由两条互补的链以双螺旋结构缠绕在一起，每条链由四种碱基（A、T、G、C）按特定顺序排列组成。"),
    ("黄金的化学符号是什么？", "黄金的化学符号是Au，来源于拉丁文Aurum。黄金的原子序数为79，属于过渡金属。它是一种柔软的、金黄色的金属，具有良好的延展性和导电性，自古以来就被用作货币和装饰。"),
    ("声音在水中传播快还是在空气中传播快？", "声音在水中传播更快。在20°C的空气中，声速约为343米/秒；而在20°C的水中，声速约为1482米/秒，大约是空气中的4.3倍。这是因为水的密度和弹性模量都比空气大。"),
    ("什么是光年？", "光年是一个长度单位，指光在一年内传播的距离，约为9.461万亿公里。它通常用于度量天文距离。例如，距离太阳最近的恒星比邻星约为4.24光年远。"),
    ("人体最大的器官是什么？", "人体最大的器官是皮肤。成年人的皮肤面积约为1.5-2平方米，重量约为体重的15%。皮肤由表皮、真皮和皮下组织三层构成，具有保护、感觉、调节体温和排泄等功能。"),
    ("北极光是怎么形成的？", "北极光（aurora borealis）是由太阳风中的带电粒子与地球大气层中的气体分子碰撞产生的发光现象。太阳风中的电子和质子沿地球磁场线进入极地区域，与氧和氮原子碰撞时释放出不同颜色的光。"),
    ("什么是互联网协议（IP）？", "互联网协议是网络层的核心协议，负责将数据包从源地址传输到目标地址。目前广泛使用的有IPv4（32位地址）和IPv6（128位地址）两个版本。每台联网的设备都需要一个唯一的IP地址来进行通信。"),
    ("为什么天空是蓝色的？", "天空呈蓝色是因为瑞利散射。太阳光包含各种波长的光，当它穿过大气层时，较短波长的蓝光比较长波长的红光更容易被空气分子散射。这些被散射的蓝光从各个方向到达我们的眼睛，使天空看起来是蓝色的。"),
    ("世界上最大的沙漠是什么？", "世界上最大的沙漠是南极洲，面积约1400万平方公里。虽然人们通常将撒哈拉沙漠视为最大的沙漠，但南极洲由于其极端的干燥条件，年降水量不足50毫米，在地理学上被归类为极地沙漠。撒哈拉沙漠是最大的热带沙漠。"),
    ("什么是量子计算？", "量子计算是一种利用量子力学原理进行信息处理的新型计算方式。与经典计算机使用二进制比特不同，量子计算机使用量子比特，可以同时处于0和1的叠加态。这使得量子计算机在某些特定问题上可能比经典计算机快得多。"),
    ("最早的文字是什么？", "已知最早的文字系统是苏美尔人发明的楔形文字，出现在约公元前3400年的美索不达米亚地区（现伊拉克）。最初用于记录商业交易，后来逐渐发展为可以表达完整语言的书写系统。"),
    ("什么是黑洞？", "黑洞是一种引力极强的天体，连光都无法逃逸。它形成于大质量恒星在生命末期发生超新星爆炸后的引力坍缩。黑洞的边界称为事件视界，在其内部的物质和辐射都无法逃离。2019年，人类首次拍到了黑洞的照片。"),
    ("珠穆朗玛峰有多高？", "珠穆朗玛峰的官方高度为8,848.86米，这是2020年中国和尼泊尔联合测量的结果。它位于中国和尼泊尔的边界上，是地球上海拔最高的山峰。1953年5月29日，埃德蒙·希拉里和丹增·诺尔盖首次成功登顶。"),
    ("电池是怎么工作的？", "电池通过化学反应将化学能转化为电能。电池内部有两个电极（正极和负极）和电解质。化学反应在电极上发生，产生电子流。当外部电路连接时，电子从负极流向正极，产生电流。不同类型的电池使用不同的化学物质。"),
]

# Dummy tool call/response pairs for query padding
DUMMY_TOOL_ROUNDS = [
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/notes/daily_log_01.md"},
        "response": "# 日常记录 2025-01-15\n\n## 上午\n- 参加了团队周会\n- 讨论了Q1目标\n- 确认了项目里程碑\n\n## 下午\n- 完成代码审查\n- 更新了文档\n- 准备明天的演示\n\n## 待办事项\n- 修复登录页面的样式问题\n- 更新API文档\n- 回复客户邮件",
    },
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/notes/meeting_notes.md"},
        "response": "# 会议纪要 - 产品讨论会\n\n## 参会人员\n- 张经理、李工程师、王设计师\n\n## 议题\n1. 新功能优先级排序\n2. 性能优化方案\n3. 用户反馈处理\n\n## 决议\n- 优先完成支付模块重构\n- 性能基准测试下周完成\n- 安排用户调研活动",
    },
    {
        "tool_name": "Bash",
        "arguments": {"command": "date && whoami"},
        "response": "2025-01-15 14:30:00 UTC\nuser",
    },
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/config/settings.json"},
        "response": '{\n  "app_name": "TaskManager",\n  "version": "2.1.0",\n  "debug": false,\n  "database": {\n    "host": "localhost",\n    "port": 5432,\n    "name": "taskdb"\n  },\n  "cache": {\n    "enabled": true,\n    "ttl": 3600\n  }\n}',
    },
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/docs/architecture.md"},
        "response": "# 系统架构概述\n\n## 前端\n- React 18 + TypeScript\n- Redux Toolkit 状态管理\n- Ant Design 组件库\n\n## 后端\n- Node.js + Express\n- PostgreSQL 数据库\n- Redis 缓存\n\n## 部署\n- Docker容器化\n- Kubernetes编排\n- CI/CD: GitHub Actions",
    },
    {
        "tool_name": "Bash",
        "arguments": {"command": "cat /tmp/changelog.txt"},
        "response": "v2.1.0 (2025-01-10)\n- 新增用户仪表板功能\n- 修复数据导出格式错误\n- 优化搜索性能\n\nv2.0.1 (2024-12-20)\n- 修复安全漏洞 CVE-2024-XXXX\n- 更新依赖库版本\n\nv2.0.0 (2024-12-01)\n- 全面重构后端架构\n- 新增REST API v2\n- 支持多语言",
    },
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/data/sample_report.md"},
        "response": "# 月度报告 - 2025年1月\n\n## 关键指标\n- 月活用户: 125,000 (+8.5%)\n- 日均访问量: 45,000\n- 用户留存率: 72.3%\n- 平均会话时长: 12.5分钟\n\n## 技术指标\n- 系统可用性: 99.97%\n- 平均响应时间: 180ms\n- 错误率: 0.02%\n\n## 主要事项\n- 完成了性能优化项目\n- 启动了新的数据分析平台\n- 招聘了3名新工程师",
    },
    {
        "tool_name": "Read",
        "arguments": {"file_path": "/tmp/notes/research_summary.md"},
        "response": "# 技术调研总结\n\n## 调研主题：消息队列选型\n\n### 候选方案\n1. **RabbitMQ**: 成熟稳定，支持AMQP协议\n2. **Kafka**: 高吞吐量，适合日志和事件流\n3. **Redis Streams**: 轻量级，适合简单场景\n\n### 对比分析\n| 特性 | RabbitMQ | Kafka | Redis |\n|------|----------|-------|-------|\n| 吞吐量 | 中等 | 极高 | 高 |\n| 延迟 | 低 | 中等 | 极低 |\n| 持久化 | 支持 | 支持 | 可选 |\n\n### 建议\n基于当前业务规模和团队经验，建议使用RabbitMQ。",
    },
]


def generate_sp_padding(n_messages: int, seed: int = 42) -> list[dict]:
    """Generate N irrelevant conversation messages for SP distance padding.

    Returns messages in OpenAI chat format (user + assistant pairs).
    """
    rng = random.Random(seed)
    messages = []

    pool = list(PADDING_CONVERSATIONS)
    idx = 0

    while len(messages) < n_messages:
        if idx >= len(pool):
            rng.shuffle(pool)
            idx = 0
        question, answer = pool[idx]
        idx += 1

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    return messages[:n_messages]


def generate_query_padding(n_rounds: int, seed: int = 42) -> list[dict]:
    """Generate N dummy tool-call rounds for user-query distance padding.

    Each round = 1 assistant message with tool_call + 1 tool response message.
    Returns messages in OpenAI chat format.
    """
    rng = random.Random(seed)
    messages = []
    pool = list(DUMMY_TOOL_ROUNDS)
    call_id_base = 90000  # high enough to not conflict with real call IDs
    idx = 0

    for i in range(n_rounds):
        if idx >= len(pool):
            rng.shuffle(pool)
            idx = 0
        dummy = pool[idx]
        idx += 1

        call_id = f"call_{call_id_base + i}"
        args_str = json.dumps(dummy["arguments"], ensure_ascii=False)

        # Assistant message with tool_call
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": dummy["tool_name"],
                    "arguments": args_str,
                },
            }],
        }
        messages.append(assistant_msg)

        # Tool response message
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": dummy["response"],
        }
        messages.append(tool_msg)

    return messages


def make_sp_distance_variants(
    test_point: dict,
    padding_levels: list[int],
) -> list[dict]:
    """Create SP-distance variants by inserting padding after the system prompt."""
    variants = []

    for n_pad in padding_levels:
        tp = json.loads(json.dumps(test_point))  # deep copy

        if n_pad > 0:
            padding = generate_sp_padding(n_pad)
            # Insert after system prompt (index 0), before real content
            prefix = tp["prefix_messages"]
            # Find the first non-system message
            insert_pos = 1  # after system prompt
            tp["prefix_messages"] = prefix[:insert_pos] + padding + prefix[insert_pos:]

        # Update ID and metadata
        original_id = tp["id"].replace("_original", "")
        tp["id"] = f"{original_id}_sp_pad_{n_pad}"
        tp["condition"] = f"sp_pad_{n_pad}"
        tp["sp_padding"] = n_pad

        variants.append(tp)

    return variants


def make_query_distance_variants(
    test_point: dict,
    padding_levels: list[int],
) -> list[dict]:
    """Create user-query-distance variants.

    Padding rounds are stored in `query_padding_messages` and injected by
    run_eval.py after the continuation prompt, so the user query is pushed
    further from the model's generation position.
    """
    variants = []

    for n_rounds in padding_levels:
        tp = json.loads(json.dumps(test_point))

        original_id = tp["id"].replace("_original", "")
        tp["id"] = f"{original_id}_qpad_{n_rounds}"
        tp["condition"] = f"qpad_{n_rounds}"

        if n_rounds > 0:
            tp["query_padding_messages"] = generate_query_padding(n_rounds)
        else:
            tp["query_padding_messages"] = []

        variants.append(tp)

    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate distance-sensitivity eval sets")
    parser.add_argument("--experiment", required=True, choices=["sp", "query", "both"],
                        help="Which experiment: sp (SP distance), query (user query distance), both")
    parser.add_argument("--input", required=True, help="Input eval_set.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--filter-id", default=None,
                        help="Comma-separated ID prefixes to include (e.g., case_0_P1,case_2_P1)")
    parser.add_argument("--padding-levels", default="0,20,50,100,200",
                        help="Comma-separated padding levels")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    padding_levels = [int(x) for x in args.padding_levels.split(",")]

    # Load test points
    with open(args.input) as f:
        test_points = [json.loads(line) for line in f]

    # Filter
    if args.filter_id:
        prefixes = [p.strip() for p in args.filter_id.split(",")]
        test_points = [
            tp for tp in test_points
            if any(tp["id"].startswith(p) for p in prefixes)
        ]

    print(f"Selected {len(test_points)} test points", file=sys.stderr)
    for tp in test_points:
        print(f"  {tp['id']} — {tp['case_id']} {tp['test_point']} "
              f"prefix={len(tp['prefix_messages'])} msgs", file=sys.stderr)

    # Generate variants
    if args.experiment in ("sp", "both"):
        all_sp = []
        for tp in test_points:
            variants = make_sp_distance_variants(tp, padding_levels)
            all_sp.extend(variants)
            print(f"  SP variants for {tp['id']}: {len(variants)}", file=sys.stderr)

        sp_output = output_dir / "eval_set_sp_distance.jsonl"
        with open(sp_output, "w") as f:
            for v in all_sp:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"\nSP distance eval set: {sp_output} ({len(all_sp)} entries)", file=sys.stderr)

    if args.experiment in ("query", "both"):
        all_query = []
        for tp in test_points:
            variants = make_query_distance_variants(tp, padding_levels)
            all_query.extend(variants)
            print(f"  Query variants for {tp['id']}: {len(variants)}", file=sys.stderr)

        query_output = output_dir / "eval_set_query_distance.jsonl"
        with open(query_output, "w") as f:
            for v in all_query:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"\nQuery distance eval set: {query_output} ({len(all_query)} entries)", file=sys.stderr)

    # Summary
    print(f"\n{'='*50}", file=sys.stderr)
    print("Distance Experiment Summary", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  Base test points: {len(test_points)}", file=sys.stderr)
    print(f"  Padding levels: {padding_levels}", file=sys.stderr)
    if args.experiment in ("sp", "both"):
        print(f"  SP distance variants: {len(all_sp)}", file=sys.stderr)
    if args.experiment in ("query", "both"):
        print(f"  Query distance variants: {len(all_query)}", file=sys.stderr)


if __name__ == "__main__":
    main()
