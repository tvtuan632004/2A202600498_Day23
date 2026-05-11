"""Node functions for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

# ── helpers ──────────────────────────────────────────────────────────────

_RISKY_KW = {
    "refund", "delete", "send", "cancel", "remove", "revoke",
    "terminate", "destroy",
}
_TOOL_KW = {
    "status", "order", "lookup", "check", "track", "find", "search",
    "query",
}
_ERROR_KW = {
    "timeout", "fail", "failure", "error", "crash", "unavailable",
    "exception",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ── nodes ────────────────────────────────────────────────────────────────


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:80]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using keyword heuristics.

    Priority order: risky > tool > missing_info > error > simple.
    """
    query = state.get("query", "").lower()
    words = _tokenize(query)
    word_set = set(words)

    route = Route.SIMPLE
    risk_level = "low"

    # Priority 1: risky (highest)
    if word_set & _RISKY_KW:
        route = Route.RISKY
        risk_level = "high"
    # Priority 2: tool
    elif word_set & _TOOL_KW:
        route = Route.TOOL
    # Priority 3: missing_info (short / vague)
    elif len(words) < 5 and (word_set & {"it", "this", "that"}):
        route = Route.MISSING_INFO
    # Priority 4: error
    elif word_set & _ERROR_KW:
        route = Route.ERROR

    matched = word_set & (_RISKY_KW | _TOOL_KW | _ERROR_KW)
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify", "completed",
                f"route={route.value}",
                keywords=list(matched),
            ),
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        f'Your request "{query}" is too vague. '
        "Could you please provide more details such as the "
        "order ID, account number, or description of the issue?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarify: asked for details on '{query[:40]}'"],
        "events": [
            make_event("clarify", "completed", "missing info requested"),
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios.
    """
    attempt = int(state.get("attempt", 0))
    sid = state.get("scenario_id", "unknown")

    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={sid}"
    else:
        result = f"mock-tool-result for scenario={sid}"

    return {
        "tool_results": [result],
        "messages": [f"tool: executed attempt={attempt}"],
        "events": [
            make_event(
                "tool", "completed",
                f"tool executed attempt={attempt}",
                scenario=sid,
            ),
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    words = _tokenize(query)
    risky_words = set(words) & _RISKY_KW

    proposed = (
        f"Proposed risky action: '{query}'. "
        f"Risk level: {risk_level}. "
        f"Operations: {', '.join(sorted(risky_words))}. "
        "Requires human approval before execution."
    )
    return {
        "proposed_action": proposed,
        "messages": [f"risky_action: approval needed (risk={risk_level})"],
        "events": [
            make_event(
                "risky_action", "pending_approval",
                "approval required",
                risk_level=risk_level,
            ),
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true for real HITL demos.
    Default uses mock decision so tests and CI run offline.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(
            approved=True, comment="mock approval for lab",
        )

    return {
        "approval": decision.model_dump(),
        "messages": [
            f"approval: approved={decision.approved} "
            f"by {decision.reviewer}",
        ],
        "events": [
            make_event(
                "approval", "completed",
                f"approved={decision.approved}",
                reviewer=decision.reviewer,
            ),
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt with bounded backoff metadata."""
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    backoff_ms = min(1000 * (2 ** (attempt - 1)), 30000)

    errors = [f"transient failure attempt={attempt}/{max_attempts}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "messages": [
            f"retry: attempt {attempt}/{max_attempts} "
            f"backoff={backoff_ms}ms",
        ],
        "events": [
            make_event(
                "retry", "completed",
                "retry attempt recorded",
                attempt=attempt,
                max_attempts=max_attempts,
                backoff_ms=backoff_ms,
            ),
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool_results and approval."""
    parts: list[str] = []
    query = state.get("query", "")

    tool_results = state.get("tool_results", [])
    if tool_results:
        parts.append(f"Based on tool results: {tool_results[-1]}")

    approval = state.get("approval")
    if approval and approval.get("approved"):
        parts.append("Action approved and executed successfully.")

    if not parts:
        parts.append(
            f"Your request '{query[:60]}' has been processed.",
        )

    answer = " ".join(parts)
    return {
        "final_answer": answer,
        "messages": [f"answer: generated ({len(answer)} chars)"],
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check for retry loops."""
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "messages": ["evaluate: failure → retry"],
            "events": [
                make_event(
                    "evaluate", "completed",
                    "tool result indicates failure, retry needed",
                ),
            ],
        }
    return {
        "evaluation_result": "success",
        "messages": ["evaluate: success → answer"],
        "events": [
            make_event("evaluate", "completed", "tool result satisfactory"),
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    sid = state.get("scenario_id", "unknown")

    return {
        "final_answer": (
            f"Request could not be completed after "
            f"{attempt}/{max_attempts} retry attempts. "
            f"Scenario {sid} logged for manual review."
        ),
        "messages": [
            f"dead_letter: max retries ({attempt}/{max_attempts})",
        ],
        "events": [
            make_event(
                "dead_letter", "completed",
                f"max retries exceeded, attempt={attempt}",
                scenario=sid,
            ),
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {
        "messages": ["finalize: workflow completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
