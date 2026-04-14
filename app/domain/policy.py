from __future__ import annotations

HIGH_RISK_PAIRS = {
    frozenset({"ask_howto", "trigger_workflow"}),
    frozenset({"create_ticket", "trigger_workflow"}),
    frozenset({"update_ticket", "trigger_workflow"}),
    frozenset({"update_ticket", "summarize_content"}),
    frozenset({"ask_knowledge", "query_status"}),
    frozenset({"create_ticket", "update_ticket"}),
}

QUESTION_TOKENS = {
    "怎么",
    "如何",
    "怎样",
    "哪一步",
    "为什么",
    "是否",
    "怎么标记",
    "怎么处理",
}

AMBIGUOUS_ACTION_TOKENS = {
    "处理一下",
    "操作一下",
    "弄一下",
    "搞一下",
    "处理一下单子",
    "处理一下工单",
    "操作一下单子",
    "操作一下工单",
    "弄一下单子",
    "弄一下工单",
    "搞一下单子",
    "搞一下工单",
    "看一下这个单子",
    "看一下这个工单",
}


def contains_ambiguous_action_signal(text: str) -> bool:
    return any(token in text for token in AMBIGUOUS_ACTION_TOKENS)


def contains_question_signal(text: str) -> bool:
    return any(token in text for token in QUESTION_TOKENS)


def build_clarification_question(emb_top: str, cls_top: str) -> str | None:
    pair = frozenset({emb_top, cls_top})

    if pair == frozenset({"ask_howto", "trigger_workflow"}):
        return "你是想查看怎么操作，还是想直接执行这个流程？"

    if pair == frozenset({"create_ticket", "trigger_workflow"}):
        return "你是想新建一个申请单，还是想推进已有流程？"

    if pair == frozenset({"update_ticket", "trigger_workflow"}):
        return "你是想修改当前对象，还是想直接执行流程动作？"

    if pair == frozenset({"update_ticket", "summarize_content"}):
        return "你是想修改当前内容，还是想让我先帮你总结一下？"

    if pair == frozenset({"ask_knowledge", "query_status"}):
        return "你是想了解规则说明，还是想查询当前状态？"

    if pair == frozenset({"create_ticket", "update_ticket"}):
        return "你是想新建一个单子，还是修改已有单子？"

    return "我理解到了两个可能的意图，你希望我执行哪一种？"


def hybrid_decide(
    *,
    query: str,
    embedding_top: str,
    embedding_score: float,
    embedding_ranked_intents: list[tuple[str, float]],
    classifier_top: str,
    classifier_score: float,
    ambiguity_margin: float,
    classifier_override_threshold: float,
    risky_pair_override_threshold: float,
) -> dict:
    emb_top = embedding_top
    cls_top = classifier_top
    cls_score = classifier_score

    emb_gap = (
        embedding_ranked_intents[0][1] - embedding_ranked_intents[1][1]
        if len(embedding_ranked_intents) > 1
        else 999.0
    )

    if emb_top == cls_top:
        if emb_top in {"create_ticket", "update_ticket"} and contains_ambiguous_action_signal(query):
            return {
                "top_intent": emb_top,
                "source": "agree_but_clarify_on_ambiguous_action",
                "need_clarification": True,
                "clarification_question": "你是想新建一个单子，还是修改已有单子？",
                "risky_pair": True,
            }

        return {
            "top_intent": emb_top,
            "source": "agree",
            "need_clarification": False,
            "clarification_question": None,
            "risky_pair": False,
        }

    pair = frozenset({emb_top, cls_top})
    risky_pair = pair in HIGH_RISK_PAIRS

    if contains_question_signal(query) and cls_top == "ask_howto" and cls_score >= 0.55:
        return {
            "top_intent": cls_top,
            "source": "classifier_override_by_question_signal",
            "need_clarification": risky_pair,
            "clarification_question": build_clarification_question(emb_top, cls_top) if risky_pair else None,
            "risky_pair": risky_pair,
        }

    if risky_pair:
        if cls_score >= risky_pair_override_threshold:
            return {
                "top_intent": cls_top,
                "source": "classifier_override_on_risky_pair",
                "need_clarification": True,
                "clarification_question": build_clarification_question(emb_top, cls_top),
                "risky_pair": True,
            }

        return {
            "top_intent": emb_top,
            "source": "embedding_keep_on_risky_pair",
            "need_clarification": True,
            "clarification_question": build_clarification_question(emb_top, cls_top),
            "risky_pair": True,
        }

    if emb_gap < ambiguity_margin and cls_score >= 0.45:
        return {
            "top_intent": cls_top,
            "source": "classifier_on_ambiguous_embedding",
            "need_clarification": True,
            "clarification_question": build_clarification_question(emb_top, cls_top),
            "risky_pair": False,
        }

    if cls_score >= classifier_override_threshold:
        return {
            "top_intent": cls_top,
            "source": "classifier_override",
            "need_clarification": False,
            "clarification_question": None,
            "risky_pair": False,
        }

    return {
        "top_intent": emb_top,
        "source": "embedding_keep",
        "need_clarification": False,
        "clarification_question": None,
        "risky_pair": False,
    }