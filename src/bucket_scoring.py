"""
========================================
bucket_scoring.py — 检索多维评分子函数
========================================

从 bucket_manager.py 拆出。BucketManager.search() 用加权多维评分给候选桶
排序：文本相关性(topic) + 情感共鸣(emotion) + 时间亲近(time) + 触碰频率
(touch)，这四维是纯函数（只读传入参数 + 本模块常量），不碰文件系统/网络，
所以独立成模块，方便单测和复用。

importance / semantic(embedding) / bm25 三个维度的计算逻辑较短，仍留在
bucket_manager.search() 内联（importance 是一行归一化，semantic/bm25 依赖
self.embedding_engine / self._bm25 等实例状态，硬抽出去反而增加耦合）。

不做什么：
- 不做加权求和/归一化（那是 search() 的事，这里只给出单维度 0~1 分）
- 不读 bucket 文件、不碰 self.config（topic 分需要的 content_weight 由
  调用方显式传入，不在这里读配置）

另有 merge_relevance：hold/grow 判断「要不要并进一个老桶」专用的分数，
只看两段正文像不像（见该函数说明）。

对外暴露：calc_topic_score / calc_emotion_score / calc_time_score /
         calc_touch_score / content_overlap / merge_relevance
========================================
"""

import math
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz
from utils import parse_iso_datetime

# --- topic 文本维度权重 ---
TOPIC_NAME_W = 3.0
TOPIC_DOMAIN_W = 2.5
TOPIC_TAG_W = 2.0
TOPIC_BODY_SLICE = 1000   # body 文本参与 fuzzy 的首部截断长度

# --- emotion 维度 ---
_DEFAULT_VALENCE = 0.5
_DEFAULT_AROUSAL = 0.3
EMOTION_MAX_DIST = math.sqrt(2)  # Russell 理论最大欧氏距离

# --- time 维度 ---
TIME_DECAY_LAMBDA = 0.02  # e^(-λ*days)，越小 → 起冷起慢
TIME_FALLBACK_DAYS = 30   # 无可解析 last_active 时的默认天数

# --- touch 维度 ---
TOUCH_NORMALIZE_CAP = 10.0   # activation_count / 该值，裁到 1.0


# ---------------------------------------------------------
# Topic relevance sub-score:
# name(×3) + domain(×2.5) + tags(×2) + body(×1)
# 文本相关性子分：桶名(×3) + 主题域(×2.5) + 标签(×2) + 正文(×1)
# ---------------------------------------------------------
def calc_topic_score(query: str, bucket: dict, content_weight: float = 1.0) -> float:
    """
    Calculate text dimension relevance score (0~1).
    计算文本维度的相关性得分。
    """
    meta = bucket.get("metadata", {})

    name_score = fuzz.partial_ratio(query, meta.get("name", "")) * TOPIC_NAME_W
    domain_score = (
        max(
            (fuzz.partial_ratio(query, d) for d in meta.get("domain", [])),
            default=0,
        )
        * TOPIC_DOMAIN_W
    )
    tag_score = (
        max(
            (fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])),
            default=0,
        )
        * TOPIC_TAG_W
    )
    content_score = fuzz.partial_ratio(query, bucket.get("content", "")[:TOPIC_BODY_SLICE]) * content_weight

    return (name_score + domain_score + tag_score + content_score) / (
        100 * (TOPIC_NAME_W + TOPIC_DOMAIN_W + TOPIC_TAG_W + content_weight)
    )


# ---------------------------------------------------------
# Emotion resonance sub-score:
# Based on Russell circumplex Euclidean distance
# 情感共鸣子分：基于环形情感模型的欧氏距离
# No emotion in query → neutral 0.5 (doesn't affect ranking)
# ---------------------------------------------------------
def calc_emotion_score(
    q_valence: Optional[float], q_arousal: Optional[float], meta: dict
) -> float:
    """
    Calculate emotion resonance score (0~1, closer = higher).
    计算情感共鸣度（0~1，越近越高）。
    """
    if q_valence is None or q_arousal is None:
        return 0.5  # No emotion coordinates → neutral / 无情感坐标时给中性分

    try:
        b_valence = float(meta.get("valence", _DEFAULT_VALENCE))
        b_arousal = float(meta.get("arousal", _DEFAULT_AROUSAL))
    except (ValueError, TypeError):
        return 0.5

    # Euclidean distance, max sqrt(2) ≈ 1.414
    dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
    return max(0.0, 1.0 - dist / EMOTION_MAX_DIST)


# ---------------------------------------------------------
# Time proximity sub-score:
# More recent activation → higher score
# 时间亲近子分：距上次激活越近分越高
# ---------------------------------------------------------
def calc_time_score(meta: dict) -> float:
    """
    Calculate time proximity score (0~1, more recent = higher).
    计算时间亲近度。
    """
    last_active_str = meta.get("last_active", meta.get("created", ""))
    try:
        last_active = parse_iso_datetime(last_active_str)
        days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
    except (ValueError, TypeError):
        days = TIME_FALLBACK_DAYS
    return math.exp(-TIME_DECAY_LAMBDA * days)


# ---------------------------------------------------------
# Touch frequency sub-score (iter 2.1)
# 触碰频率子分：被主动召回次数越多分越高
# ---------------------------------------------------------
def calc_touch_score(meta: dict) -> float:
    """
    Calculate touch frequency score (0~1).
    Normalizes activation_count over 10; capped at 1.0.
    计算触碰频率得分（0~1），以 10 次为上限归一化。
    """
    count = float(meta.get("activation_count") or 0)
    return min(count / TOUCH_NORMALIZE_CAP, 1.0)


# ---------------------------------------------------------
# Merge relevance (hold / grow 自动合并的门)
# ---------------------------------------------------------
# 以前 merge_or_create 直接拿 search() 的七维排序分去和 merge_threshold 比。
# 那个分数回答的是「这条记忆该不该被想起来、排第几」，里面有：
#   - 桶名 / 主题域 / 标签的 partial_ratio：查询一长（日记、信），短短的
#     桶名、标签几乎必然整段出现在查询里，直接拿满分；
#   - BM25 按本次最高分归一，排第一的桶永远是 1.0；
#   - 重要度、新旧、被想起次数、情绪 —— 和「两段话是不是同一件事」无关。
# 结果是长内容会被吸进「重要、常被想起」的桶（2026-11-15 一整篇周记被并进
# 锚点桶「Ink的诞生与命名」，score 78.07），并且越并越像，滚雪球。
#
# 合并的问题只有一个：「新的这段，是不是老桶里那件事的又一份」。所以这里只看
# 正文：语义相似度 + 正文重合度，各占一半。2026-11-22 在真实记忆副本上量过：
#   不同的日记之间语义 0.78~0.86，但正文重合只有 0.13~0.16 → 约 50 分；
#   同一条记忆的两个版本 → 75~100 分。门槛仍用 merge_threshold（默认 75）。
MERGE_SEMANTIC_W = 0.5
MERGE_OVERLAP_W = 0.5


def content_overlap(new: str, old: str) -> float:
    """
    How much of ``new`` is already in ``old`` (0~1).

    新内容比老桶短：用 partial_ratio（新的这段能不能在老桶里找到对应的一段）；
    新内容比老桶长：用整体 ratio —— 一段长文不能因为「包含了老桶那几句话」
    就被当成老桶的又一份，否则长文会把短桶吞进来。
    """
    new = (new or "").strip()
    old = (old or "").strip()
    if not new or not old:
        return 0.0
    if len(new) <= len(old):
        return fuzz.partial_ratio(new, old) / 100.0
    return fuzz.ratio(new, old) / 100.0


def merge_relevance(new: str, old: str, semantic: Optional[float] = None) -> float:
    """
    Merge gate score (0~100): is ``new`` another copy of what ``old`` holds?

    只用正文：semantic（余弦相似度，没有就不算）和 content_overlap。
    不看桶名、标签、主题域、重要度、新旧、被想起次数、情绪。
    """
    overlap = content_overlap(new, old)
    if semantic is None:
        return overlap * 100.0
    sem = min(1.0, max(0.0, float(semantic)))
    return (sem * MERGE_SEMANTIC_W + overlap * MERGE_OVERLAP_W) / (MERGE_SEMANTIC_W + MERGE_OVERLAP_W) * 100.0
