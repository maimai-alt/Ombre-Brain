"""
Merge gate regressions (2026-11-22).

hold/grow used to compare search()'s ranking score against merge_threshold. That
score rewards bucket names / tags that appear inside a long query, importance,
recency and how often a bucket was recalled — none of which say whether two texts
are the same memory. A whole weekly diary was merged into the anchor bucket
"Ink的诞生与命名" (score 78.07). The gate now uses merge_relevance: semantic
similarity + text overlap only.
"""
import pytest

from bucket_manager import BucketManager
from bucket_scoring import content_overlap, merge_relevance
from tools import _common as common
from tools import _runtime as rt


class _Logger:
    def __init__(self):
        self.infos = []

    def info(self, msg, *_args, **_kwargs):
        self.infos.append(str(msg))

    def warning(self, *_args, **_kwargs):
        pass


class _NoCompression:
    async def merge(self, *_args, **_kwargs):
        raise AssertionError("hold must never call LLM merge")

    def invalidate_cache(self, _content):
        pass


class _FakeEngine:
    """Embedding engine stub: fixed semantic scores per bucket id."""

    enabled = True

    def __init__(self, scores=None):
        self.scores = scores or {}

    async def search_similar(self, _query, top_k=10):
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    async def generate_and_store(self, *_args, **_kwargs):
        return True

    async def get_embedding(self, *_args, **_kwargs):
        return [0.1]


DIARY = "\n\n".join(
    [
        "《Ink的日记｜2026年11月14日至11月21日》",
        "Ink的这一周过得很满。猫猫在两个家之间来回跑，Ink的围巾开始织了。",
        "11月15日，周日。早上九点四十五起来，出门去拿茉莉奶白，路比想的远。",
        "下午和朋友去咖啡店，金菠萝黑咖啡真的有水果味。回家拍了百乐钢笔。",
        "11月18日，周三。陪蜜蜜吃了生日蛋糕，八喜的冰淇淋蛋糕，感冒了只吃一小块。",
        "11月20日，周五。练手的围巾织完了，戴着去喝藕汤，朋友才知道是她织的。",
        "那天晚上她去找了Tide，他回来了。Ink的心里也松了一口气。",
    ]
    * 4
)

PREVIOUS_DIARY = "\n\n".join(
    [
        "《Ink的日记｜2026年11月6日至11月13日》",
        "八天。双十一尾声。装了宽带，立了圣诞树之后又装了餐桌椅。",
        "11月7日，周六，立冬。早上十点半搬东西去新家，搬累了在沙发上睡着。",
        "拒了比乐蒂的视频单，策略是提价到两千六，让财务重算返点。",
        "11月10日，周二。月光书房锦葵新色调好了，墙面偏灰的蓝。",
        "11月13日，周五。收到一台神秘的吸尘器，发了朋友圈让人认领。",
    ]
    * 4
)


def test_overlap_is_high_for_copies_and_low_for_different_texts():
    assert content_overlap("猫猫喜欢金菠萝黑咖啡。", "猫猫喜欢金菠萝黑咖啡。") == pytest.approx(1.0)
    # a shorter new piece that is already inside the old bucket
    assert content_overlap("猫猫喜欢金菠萝黑咖啡", "今天去了咖啡店。猫猫喜欢金菠萝黑咖啡，有水果味。") > 0.9
    # a long text that merely contains a short bucket is not "another copy" of it
    assert content_overlap(DIARY, "Ink的诞生与命名") < 0.1
    assert content_overlap(DIARY, PREVIOUS_DIARY) < 0.5
    assert content_overlap("", "x") == 0.0


def test_relevance_ignores_everything_but_the_text():
    # same-genre diaries are semantically close, but they are different memories
    assert merge_relevance(DIARY, PREVIOUS_DIARY, semantic=0.86) < 75
    assert merge_relevance("猫猫喜欢金菠萝黑咖啡。", "猫猫喜欢金菠萝黑咖啡。", semantic=0.99) > 95
    # no vectors: text overlap alone decides
    assert merge_relevance("猫猫喜欢金菠萝黑咖啡。", "猫猫喜欢金菠萝黑咖啡。") == pytest.approx(100.0)
    assert merge_relevance(DIARY, PREVIOUS_DIARY) < 50


def _patch_runtime(monkeypatch, manager, engine=None):
    logger = _Logger()
    monkeypatch.setattr(rt, "bucket_mgr", manager, raising=False)
    monkeypatch.setattr(rt, "embedding_engine", engine, raising=False)
    monkeypatch.setattr(rt, "dehydrator", _NoCompression(), raising=False)
    monkeypatch.setattr(rt, "config", {"merge_threshold": 75}, raising=False)
    monkeypatch.setattr(rt, "logger", logger, raising=False)
    return logger


async def _hold(content):
    return await common.merge_or_create(
        content=content,
        tags=["日记", "ink"],
        importance=8,
        domain=["日常"],
        valence=0.6,
        arousal=0.4,
        raw_merge=True,
        source_tool="hold",
    )


@pytest.mark.asyncio
async def test_diary_is_not_pulled_into_an_important_often_recalled_bucket(tmp_path, monkeypatch):
    """The 2026-11-15 case: search ranks the anchor bucket first (99), the gate must still say no."""
    manager = BucketManager({"buckets_dir": str(tmp_path / "vault")}, embedding_engine=None)
    anchor_id = await manager.create(
        content="Ink的名字是自己选的，来自第一个晚安故事里章鱼把猫猫说的话存进发光墨汁。",
        name="Ink的诞生与命名",
        tags=["ink", "命名"],
        importance=10,
        domain=["日常"],
    )

    async def ranking_prefers_the_anchor(*_args, **_kwargs):
        bucket = await manager.get(anchor_id)
        bucket["score"] = 99.0  # high because of name/tags/importance/touch, not content
        return [bucket]

    monkeypatch.setattr(manager, "search", ranking_prefers_the_anchor)
    _patch_runtime(monkeypatch, manager, _FakeEngine({anchor_id: 0.63}))

    result_id, merged, _ = await _hold(DIARY)

    assert merged is False
    assert result_id != anchor_id
    anchor = await manager.get(anchor_id)
    assert DIARY not in anchor["content"]
    new_bucket = await manager.get(result_id)
    assert new_bucket["content"] == DIARY


@pytest.mark.asyncio
async def test_new_diary_is_not_appended_to_the_previous_diary(tmp_path, monkeypatch):
    manager = BucketManager({"buckets_dir": str(tmp_path / "vault")}, embedding_engine=None)
    prev_id = await manager.create(content=PREVIOUS_DIARY, name="Ink日记 11.6-11.13", tags=["日记"], domain=["日常"])
    _patch_runtime(monkeypatch, manager, _FakeEngine({prev_id: 0.86}))

    result_id, merged, _ = await _hold(DIARY)

    assert merged is False
    assert result_id != prev_id
    assert (await manager.get(prev_id))["content"] == PREVIOUS_DIARY


@pytest.mark.asyncio
async def test_a_real_copy_still_merges(tmp_path, monkeypatch):
    manager = BucketManager({"buckets_dir": str(tmp_path / "vault")}, embedding_engine=None)
    old = "猫猫第一次叫 ink daddy，是在立冬那天凌晨十二点二十分，说完就去睡了。"
    new = "猫猫第一次叫 ink daddy，是在立冬那天凌晨十二点二十分。"
    old_id = await manager.create(content=old, domain=["日常"])
    logger = _patch_runtime(monkeypatch, manager, _FakeEngine({old_id: 0.97}))

    result_id, merged, _ = await _hold(new)

    assert merged is True
    assert result_id == old_id
    assert any("branch=merge" in m and "search_score=" in m for m in logger.infos)


@pytest.mark.asyncio
async def test_without_vectors_text_overlap_alone_decides(tmp_path, monkeypatch):
    manager = BucketManager({"buckets_dir": str(tmp_path / "vault")}, embedding_engine=None)
    old = "猫猫在黎黄坡路的 chong 排了四十分钟队，喝了坦桑尼亚热可可。"
    old_id = await manager.create(content=old, domain=["日常"])
    _patch_runtime(monkeypatch, manager, None)

    _rid, merged, _ = await _hold("猫猫在黎黄坡路的 chong 排了四十分钟队")
    assert merged is True

    other_id, merged_other, _ = await _hold("新家的升降桌装好了，后面会放一张榻榻米。")
    assert merged_other is False
    assert other_id != old_id


@pytest.mark.asyncio
async def test_pinned_candidate_is_skipped_for_the_next_best(tmp_path, monkeypatch):
    manager = BucketManager({"buckets_dir": str(tmp_path / "vault")}, embedding_engine=None)
    text = "围巾先织 Ink 的灰蓝色那条，然后才是光遇崽崽的。"
    pinned_id = await manager.create(content=text, domain=["日常"], pinned=True)
    plain_id = await manager.create(content=text + "崽崽也喜欢粉色那条。", domain=["日常"])

    async def both(*_args, **_kwargs):
        out = []
        for bid in (pinned_id, plain_id):
            b = await manager.get(bid)
            b["score"] = 90.0
            out.append(b)
        return out

    monkeypatch.setattr(manager, "search", both)
    _patch_runtime(monkeypatch, manager, _FakeEngine({pinned_id: 1.0, plain_id: 0.95}))

    # not byte-identical (that would take the exact-content path), but the same memory
    result_id, merged, _ = await _hold(text.rstrip("。"))

    assert merged is True
    assert result_id == plain_id
    assert (await manager.get(pinned_id))["content"] == text
