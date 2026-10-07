"""BTVN#3 — Mẫu 1: ReAct Agent (Reasoning + Acting)

Kiến trúc:
    Model quyết định MỘT bước tại một thời điểm.
    Sau mỗi observation, model suy luận và chọn bước tiếp theo.
    Vòng lặp: Suy luận → Hành động → Quan sát → lặp lại.

Harness 4 lớp:
    1. Ràng buộc là dữ liệu    → Constraints
    2. Kiểm quyền               → check_permission(), chạy TRƯỚC tool
    3. Tiêu chí hoàn thành      → is_done(), kiểm bằng code
    4. Bàn giao                  → handoff(), khi agent thất bại

Run:  python react_agent.py
      Chọn case: 1 (success) hoặc 2 (fail)
"""
import json

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

from shared import (
    FLIGHTS, SYSTEM_PROMPT,
    AgentState, Constraints, LoopDetector,
    check_permission, fake_model_react, handoff, is_done, make_tools,
    ungrounded_facts,
)


# =====================================================================
# CẤU HÌNH
# =====================================================================
CONSTRAINTS = Constraints()
STATE = AgentState()
LOOP_DETECTOR = LoopDetector(window=6, repeat_k=2, stall_n=5)


# =====================================================================
# HARNESS — mỗi lần agent gọi tool đều đi qua đây
# =====================================================================
class StopAgent(Exception):
    """Harness dùng exception này để dừng agent ngay lập tức."""


@wrap_tool_call
def harness(request, handler):
    call = request.tool_call
    name, args = call["name"], call["args"]

    # --- Phát hiện lặp
    loop_signal = LOOP_DETECTOR.check(name, args, progress=len(STATE.bookings))
    if loop_signal:
        raise StopAgent(f"{loop_signal}: {name}({args}) đã lặp lại quá nhiều lần")

    # --- Kiểm quyền TRƯỚC khi tool chạy
    reason = check_permission(name, args, CONSTRAINTS, FLIGHTS[STATE.case])
    if reason:
        result = {"status": "denied", "reason": reason}
    else:
        # Cho phép → chạy tool thật
        tools_by_name = {t.__name__: t for t in make_tools(STATE)}
        result = tools_by_name[name](**args)

    STATE.log.append((name, args, result))
    return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])


# =====================================================================
# CHẠY AGENT
# =====================================================================
def run_react(case: str) -> dict:
    """Chạy ReAct agent, trả về dict kết quả."""
    STATE.reset(case)
    LOOP_DETECTOR.reset()

    model = fake_model_react(case)
    tools = make_tools(STATE)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            harness,                                      # harness của chúng ta
            ModelCallLimitMiddleware(run_limit=10),        # ngân sách: tối đa 10 lượt model
        ],
    )

    print(f"\n{'='*60}")
    print(f"  MẪU 1: ReAct  |  Case: {case}")
    print(f"{'='*60}")
    print(f"Ràng buộc (dữ liệu): {CONSTRAINTS}")
    print(f"Yêu cầu người dùng: {CONSTRAINTS.to_prompt()}")

    stop_reason = None
    try:
        out = agent.invoke(
            {"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]}
        )
        answer = out["messages"][-1].content
    except StopAgent as e:
        stop_reason = str(e)
        answer = ""

    # Kiểm tra hallucination
    fake_facts = ungrounded_facts(answer, STATE.log)
    if fake_facts:
        stop_reason = (
            stop_reason or ""
        ) + f" | HALLUCINATION: {fake_facts} không có nguồn từ tool"

    # --- Trace
    print("\n--- trace ---")
    for i, (tool, args, result) in enumerate(STATE.log, 1):
        status = result.get("status", "?")
        reason_str = result.get("reason", "")
        print(f"  [{i}] {tool}({args}) -> {status} {reason_str}")

    # --- Kết quả: harness quyết định, không phải model
    if is_done(STATE, CONSTRAINTS) and not stop_reason:
        output = {
            "pattern": "ReAct",
            "result": "DONE",
            "model_calls": len(STATE.log),
            "booking": list(STATE.bookings.values()),
        }
    else:
        output = {
            "pattern": "ReAct",
            "result": "FAILED",
            "model_calls": len(STATE.log),
            "stop_reason": stop_reason or "GOAL NOT REACHED",
            "model_said": answer,
            "handoff": handoff(STATE),
        }

    print("\n--- kết quả (harness quyết định) ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


# =====================================================================
if __name__ == "__main__":
    print("BTVN#3 — Mẫu 1: ReAct Agent")
    print("Chọn case:")
    print("  1. success  - có chuyến bay hợp lệ")
    print("  2. fail     - không có chuyến bay thỏa, harness phải bàn giao")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Lựa chọn (1/2): ").strip()
    case = "success" if choice == "1" else "fail"
    run_react(case)
