"""
GRPO 训练脚本 — 基于 TRL 的 NL2SQL DSL 强化学习训练

Reward 设计：
  1. format_reward: JSON 格式是否合法 (0/1)
  2. field_reward: 经 ValidatorV2 修正后，字段/op/intent 是否合规 (0~1)
  3. dsl_match_reward: 修正后 DSL 与 ground truth 完全匹配 (0/1)
  4. sql_reward: 编译后 SQL 与 ground truth SQL 一致 (0/1)

用法:
  python scripts/train_grpo.py \
      --model_name_or_path Qwen/Qwen3-4B \
      --dataset data/train_v2.jsonl \
      --output_dir output/grpo_nl2sql \
      --num_train_epochs 3
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compiler import CompilerV2
from validator import ValidatorV2
from prompt_builder import PromptBuilderV2, FewShotRetriever

import yaml


# ============== Rollout 日志 Callback ==============

class RolloutLogger(TrainerCallback):
    """记录每次 rollout 的 prompt、completion、reward 到 JSONL 文件。"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._f = open(log_path, "a", encoding="utf-8")
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step = state.global_step

    def log_rollout(self, prompts, completions, rewards_per_func):
        """由 reward wrapper 调用，写入 rollout 记录并打印到屏幕。"""
        for i, msgs in enumerate(completions):
            text = msgs[-1]["content"] if isinstance(msgs, list) else str(msgs)
            # 从 prompt 末尾提取用户 query（prompt 最后一行是 "# 用户输入\nXXX"）
            prompt_text = ""
            if i < len(prompts):
                p = prompts[i]
                p_content = p[-1]["content"] if isinstance(p, list) else str(p)
                # 取最后一行作为 query
                prompt_text = p_content.strip().split("\n")[-1]
            record = {
                "step": self._step,
                "idx": i,
                "query": prompt_text,
                "completion": text,
                "rewards": rewards_per_func[i],
            }
            self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 打印到屏幕
            short = text[:100].replace("\n", "\\n")
            r_str = " | ".join(f"{k}={v:.2f}" for k, v in rewards_per_func[i].items())
            print(f"[step={self._step}] Q: {prompt_text[:40]} | {r_str} | {short}")
        self._f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._f.close()


def wrap_reward_with_logging(reward_fns: list, logger: "RolloutLogger", names: list):
    """包装 reward 函数，在最后一个 reward 执行完后统一记录。"""
    reward_cache = {}

    def make_wrapper(fn, name, is_last):
        def wrapper(completions, **kwargs):
            rewards = fn(completions, **kwargs)
            reward_cache[name] = rewards
            if is_last:
                # 所有 reward 都算完了，合并记录
                prompts = kwargs.get("prompts", [])
                per_sample = []
                for i in range(len(completions)):
                    per_sample.append({n: reward_cache[n][i] for n in names})
                logger.log_rollout(prompts, completions, per_sample)
                reward_cache.clear()
            return rewards
        return wrapper

    return [make_wrapper(fn, names[i], i == len(reward_fns) - 1) for i, fn in enumerate(reward_fns)]


# ============== 加载配置 ==============

def load_valid_fields(config_dir: str) -> Set[str]:
    fm = yaml.safe_load(open(f"{config_dir}/field_mapping.yaml", encoding="utf-8")) or {}
    return set(fm.keys()) | {"salesPriceYuan", "salesModelName", "productPositionName"}


# ============== Reward 函数 ==============

def build_reward_functions(config_dir: str):
    """构建 4 个 reward 函数，共享 validator 和 compiler 实例。"""
    valid_fields = load_valid_fields(config_dir)
    validator = ValidatorV2(valid_fields)
    compiler = CompilerV2(config_dir)

    def _extract_json_safe(text: str) -> Optional[Dict]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def format_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
        """JSON 可解析则得 1 分。"""
        rewards = []
        for msgs in completions:
            text = msgs[-1]["content"] if isinstance(msgs, list) else msgs
            rewards.append(1.0 if _extract_json_safe(text) is not None else 0.0)
        return rewards

    def field_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
        """经 validator 修正后，保留的 filter 数占原始比例。intent 合法 +0.2。"""
        rewards = []
        for msgs in completions:
            text = msgs[-1]["content"] if isinstance(msgs, list) else msgs
            raw_dsl = _extract_json_safe(text)
            if raw_dsl is None:
                rewards.append(0.0)
                continue
            score = 0.0
            # intent 合法
            if raw_dsl.get("intent") in {"list", "price", "spec", "field"}:
                score += 0.2
            # filter 保留率
            raw_filters = raw_dsl.get("filters") or []
            if raw_filters:
                valid_count = sum(
                    1 for f in raw_filters
                    if isinstance(f, dict) and f.get("field") in valid_fields
                )
                score += 0.8 * (valid_count / len(raw_filters))
            else:
                score += 0.8  # 无 filter 不扣分
            rewards.append(score)
        return rewards

    def dsl_match_reward(completions: List[List[Dict]], prompts: List[str] = None, **kwargs) -> List[float]:
        """修正后 DSL 与 ground truth 完全匹配。"""
        gt_dsls = kwargs.get("gt_dsl", [])
        rewards = []
        for i, msgs in enumerate(completions):
            text = msgs[-1]["content"] if isinstance(msgs, list) else msgs
            try:
                pred = validator.parse_and_validate(text, "")
            except (ValueError, Exception):
                rewards.append(0.0)
                continue
            if i < len(gt_dsls):
                gt = gt_dsls[i] if isinstance(gt_dsls[i], dict) else json.loads(gt_dsls[i])
                score = _dsl_match_score(pred, gt)
            else:
                score = 0.0
            rewards.append(score)
        return rewards

    def sql_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
        """修正后 DSL 编译为 SQL，与 ground truth SQL 一致则得 1 分。"""
        gt_dsls = kwargs.get("gt_dsl", [])
        rewards = []
        for i, msgs in enumerate(completions):
            text = msgs[-1]["content"] if isinstance(msgs, list) else msgs
            try:
                pred = validator.parse_and_validate(text, "")
                pred_sql = compiler.compile(pred)
            except (ValueError, Exception):
                rewards.append(0.0)
                continue
            if i < len(gt_dsls):
                gt = gt_dsls[i] if isinstance(gt_dsls[i], dict) else json.loads(gt_dsls[i])
                try:
                    gt_sql = compiler.compile(gt)
                except Exception:
                    rewards.append(0.0)
                    continue
                rewards.append(1.0 if _normalize_sql(pred_sql) == _normalize_sql(gt_sql) else 0.0)
            else:
                rewards.append(0.0)
        return rewards

    return [format_reward, field_reward, dsl_match_reward, sql_reward]


def _dsl_match_score(pred: Dict, gt: Dict) -> float:
    """细粒度 DSL 匹配打分 (0~1)。"""
    score = 0.0
    total = 4.0

    # category + brand
    if (pred.get("category") or None) == (gt.get("category") or None):
        score += 1.0
    if (pred.get("brand") or None) == (gt.get("brand") or None):
        score += 0.5
    total += 0.5

    # filters
    def _fk(f):
        return (str(f.get("field")), str(f.get("op")), str(f.get("value")).strip("%"))
    pred_f = sorted([_fk(f) for f in pred.get("filters") or []])
    gt_f = sorted([_fk(f) for f in gt.get("filters") or []])
    if pred_f == gt_f:
        score += 1.0

    # sort
    ps = pred.get("sort") or {}
    gs = gt.get("sort") or {}
    if (ps.get("field"), ps.get("dir")) == (gs.get("field"), gs.get("dir")):
        score += 1.0

    # intent
    if (pred.get("intent") or "list") == (gt.get("intent") or "list"):
        score += 1.0

    return score / total


_WS_RE = re.compile(r"\s+")

def _normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    s = _WS_RE.sub(" ", sql.strip().upper())
    return s


# ============== 数据加载 ==============

def load_train_dataset(dataset_path: str, prompt_builder: PromptBuilderV2) -> Dataset:
    """加载 JSONL 训练数据，构建 prompt 列。"""
    from prompt_builder import extract_model, MODEL_PLACEHOLDER
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item["query"]
            dsl = item["dsl"]

            # 与推理流程一致：型号提取 + 类目检测
            query_for_prompt, extracted_model = extract_model(query)
            category = _detect_category(query)
            prompt = prompt_builder.build(query_for_prompt, category)

            # gt_dsl 中的型号值也替换为占位符（让 reward 对齐）
            gt_dsl = json.loads(json.dumps(dsl))
            if extracted_model:
                for f in gt_dsl.get("filters") or []:
                    if f.get("value") == extracted_model:
                        f["value"] = MODEL_PLACEHOLDER

            records.append({
                "prompt": [{"role": "user", "content": prompt}],
                "gt_dsl": json.dumps(gt_dsl, ensure_ascii=False),
            })

    return Dataset.from_list(records)


def _detect_category(query: str) -> Optional[str]:
    """关键词匹配类目（与 PipelineV2._detect_category 保持一致）"""
    kw_map = {
        "电视": ["电视", "TV"],
        "空调": ["空调"],
        "冰箱": ["冰箱"],
        "冷柜": ["冷柜", "冰柜"],
        "洗衣机": ["洗衣机"],
        "烘干机": ["烘干机", "干衣机"],
        "投影": ["投影", "投影仪"],
        "显示器": ["显示器"],
    }
    for cat, kws in kw_map.items():
        for kw in kws:
            if kw in query:
                return cat
    return None


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser(description="GRPO training for NL2SQL")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/train_v2.jsonl")
    parser.add_argument("--config_dir", type=str, default=str(_PROJECT_ROOT / "configs"))
    parser.add_argument("--output_dir", type=str, default="output/grpo_nl2sql")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()

    # 构建 prompt builder
    retriever = FewShotRetriever(f"{args.config_dir}/few_shots.yaml")
    prompt_builder = PromptBuilderV2(retriever)

    # 加载数据
    dataset = load_train_dataset(args.dataset, prompt_builder)
    print(f"Loaded {len(dataset)} training samples")

    # 构建 reward 函数
    reward_fns = build_reward_functions(args.config_dir)
    reward_names = ["format", "field", "dsl_match", "sql"]

    # Rollout 日志
    rollout_log_path = str(Path(args.output_dir) / "rollout_log.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rollout_logger = RolloutLogger(rollout_log_path)
    reward_fns = wrap_reward_with_logging(reward_fns, rollout_logger, reward_names)

    # GRPO 配置
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        remove_unused_columns=False,
    )

    # 加载模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建 Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[rollout_logger],
    )

    # 开始训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
