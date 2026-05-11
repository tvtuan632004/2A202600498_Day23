"""Report generation helper.

Generates a comprehensive lab report based on scenario metrics.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Return a complete lab report using scenario metrics."""

    # Build scenario results table
    scenario_rows = []
    for m in metrics.scenario_metrics:
        success_icon = "✅" if m.success else "❌"
        scenario_rows.append(
            f"| {m.scenario_id} | {m.expected_route} | {m.actual_route or 'N/A'} "
            f"| {success_icon} | {m.retry_count} | {m.interrupt_count} |"
        )
    scenario_table = "\n".join(scenario_rows)

    return f"""# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / Student

- Name: Trần Văn Tuấn
- Date: 2026-05-11
- Lab: Day 08 — LangGraph Agentic Orchestration

## 2. Architecture

The agent is built as a **LangGraph StateGraph** with 10 nodes and conditional edges that implement
a support-ticket workflow. The graph follows a linear pipeline (`intake → classify`) then fans out
into 5 conditional routes based on keyword classification.

### Graph Flow

```
START → intake → classify → [conditional routing]
  simple       → answer → finalize → END
  tool         → tool → evaluate → answer → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → tool → evaluate → answer → finalize → END
  error        → retry → tool → evaluate → [retry loop or dead_letter]
  max retry    → dead_letter → finalize → END
```

### Key Design Decisions

1. **Keyword-based classification** with strict priority order
   (risky > tool > missing_info > error > simple)
   prevents ambiguous routing when queries contain overlapping keywords.

2. **Bounded retry loop** via `route_after_evaluate` → `retry` → `route_after_retry` ensures error
   scenarios don't loop forever. The `max_attempts` field controls the bound.

3. **HITL approval** is implemented as a mock by default but supports real `interrupt()` when
   `LANGGRAPH_INTERRUPT=true` is set.

4. **Append-only audit trail** via `events`, `messages`, `errors`, and `tool_results` fields with
   the `add` reducer ensures full observability without state mutation.

## 3. State Schema

| Field | Reducer | Why |
|---|---|---|
| `messages` | append (`add`) | Audit trail of all node activities |
| `tool_results` | append (`add`) | Accumulate tool outputs across retries |
| `errors` | append (`add`) | Track all errors for debugging |
| `events` | append (`add`) | Structured audit events for metrics |
| `route` | overwrite | Current classification route |
| `attempt` | overwrite | Current retry attempt counter |
| `evaluation_result` | overwrite | Latest evaluate decision (success/needs_retry) |
| `approval` | overwrite | Latest approval decision |
| `final_answer` | overwrite | Final response to user |

## 4. Scenario Results

### Metrics Summary

- **Total scenarios**: {metrics.total_scenarios}
- **Success rate**: {metrics.success_rate:.2%}
- **Average nodes visited**: {metrics.avg_nodes_visited:.2f}
- **Total retries**: {metrics.total_retries}
- **Total interrupts (HITL)**: {metrics.total_interrupts}
- **Resume success**: {metrics.resume_success}

### Per-Scenario Results

| Scenario | Expected Route | Actual Route | Success | Retries | Interrupts |
|---|---|---|---|---:|---:|
{scenario_table}

## 5. Failure Analysis

### 1. Retry / Tool Failure (S05_error)

The error route simulates transient tool failures. The `tool_node` returns an `ERROR:` prefix on
early attempts (attempt < 2), causing `evaluate_node` to set `evaluation_result = "needs_retry"`.
The `route_after_evaluate` function sends the flow back to `retry`, which increments the attempt
counter. After the transient failures clear (attempt >= 2), the tool succeeds and the flow
proceeds to `answer`.

**Mitigation**: Bounded retry with `max_attempts` prevents infinite loops. Exponential backoff
metadata is recorded for production observability.

### 2. Dead Letter (S07_dead_letter)

When `max_attempts = 1`, the retry loop immediately exhausts. `route_after_retry` detects
`attempt >= max_attempts` and routes to `dead_letter_node`, which logs the failure and produces
a "logged for manual review" response. This ensures the graph always terminates.

**Mitigation**: Dead letter queue with full error context enables manual investigation.

### 3. Risky Action Without Approval

If the approval node rejects an action (`approved=False`), the flow routes to `clarify` instead
of `tool`, preventing unauthorized risky operations. In the current lab, mock approval always
approves, but the architecture supports reject/edit flows.

## 6. Persistence / Recovery Evidence

- **Checkpointer**: `MemorySaver` is used by default for development. `SqliteSaver` with WAL mode
  is available via `checkpointer: sqlite` in `configs/lab.yaml`.
- **Thread ID**: Each scenario run uses a unique `thread_id` (format: `thread-{{scenario_id}}`),
  enabling independent state tracking per conversation.
- **State history**: The checkpointer stores state at each node transition, enabling time-travel
  debugging via `get_state_history()`.
- **SQLite WAL mode**: Enables concurrent reads during state inspection without blocking writes.

## 7. Extension Work

### Graph Diagram (Mermaid)

The graph architecture can be exported as a Mermaid diagram using:
```python
graph = build_graph()
print(graph.get_graph().draw_mermaid())
```

### Additional Test Scenarios

Custom scenarios (S08, S09) were added to test edge cases:
- S08: "Cancel my subscription immediately" → risky route with approval
- S09: "Find my recent transactions" → tool route

### SQLite Persistence

SQLite checkpointer implemented with:
- `sqlite3.connect()` with `check_same_thread=False`
- WAL journal mode for concurrent read performance
- State survives process restarts with the same thread_id

## 8. Improvement Plan

If given one more day, the following improvements would be prioritized:

1. **LLM-based classification**: Replace keyword heuristics with a small LLM classifier
   (e.g., Gemma 2B) for more robust routing that handles paraphrased queries.

2. **Real tool integration**: Connect to actual APIs (order management, refund system) instead of
   mock tool results.

3. **Structured evaluation**: Use LLM-as-judge for `evaluate_node` to assess tool result quality
   beyond simple ERROR string matching.

4. **Production monitoring**: Add OpenTelemetry tracing, Prometheus metrics, and alerting for
   dead letter queue depth.

5. **Multi-turn conversation**: Support follow-up queries within the same thread using state
   history and context windowing.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the lab report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
