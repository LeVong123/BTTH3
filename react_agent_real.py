"""BTVN#3 — Mẫu 1: ReAct Agent với LLM thật (Google Gemini 2.5 Flash)

Kiến trúc ReAct sử dụng mô hình gemini-2.5-flash từ Google AI Studio.
Tích hợp cơ chế tự động Fallback khi Google AI Studio Free Tier chạm giới hạn quota 20 req/ngày.

Harness 4 lớp:
    1. Ràng buộc là dữ liệu    → Constraints
    2. Kiểm quyền               → check_permission(), chạy TRƯỚC tool
    3. Tiêu chí hoàn thành      → is_done(), kiểm bằng code
    4. Bàn giao                  → handoff(), khi agent thất bại

Run:  python react_agent_real.py
      Chọn case: 1 (success) hoặc 2 (fail)
"""
import json
import sys

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

from shared import (
    FLIGHTS,
    AgentState, Constraints, LoopDetector,
    check_permission, handoff, is_done, make_tools,
    ungrounded_facts, load_real_model, fake_model_react,
)


# =====================================================================
# CẤU HÌNH
# =====================================================================
CONSTRAINTS = Constraints()
STATE = AgentState()
LOOP_DETECTOR = LoopDetector(window=6, repeat_k=2, stall_n=5)

SYSTEM_PROMPT_REAL = (
    "You are an automated flight booking assistant. Follow these steps strictly:\n"
    "1. First call `search_flights(origin, destination, date)` to inspect available flights.\n"
    "2. Check if there is any flight satisfying ALL user constraints (route, date, departure before time, and price limit).\n"
    "3. If a valid flight is found: call `book_seat(flight)` to hold the seat, then call `pay(code)` to finalize payment, and call `get_booking(code)`.\n"
    "4. If no flight meets all constraints, do NOT book or pay. Stop immediately and explain clearly.\n"
)


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
# CHẠY AGENT VỚI LLM THẬT
# =====================================================================
def run_react_real(case: str) -> dict:
    """Chạy ReAct agent với Gemini 2.5 Flash thật, trả về dict kết quả."""
    STATE.reset(case)
    LOOP_DETECTOR.reset()

    # Sử dụng model LLM thật từ Google AI Studio
    tools = make_tools(STATE)

    print(f"\n{'='*60}")
    print(f"  MẪU 1: ReAct (LLM THẬT - Gemini 2.5 Flash)  |  Case: {case}")
    print(f"{'='*60}")
    print(f"Ràng buộc (dữ liệu): {CONSTRAINTS}")
    print(f"Yêu cầu người dùng: {CONSTRAINTS.to_prompt()}")

    stop_reason = None
    answer = ""
    engine_used = "Gemini 2.5 Flash (Google AI Studio)"

    try:
        model = load_real_model("gemini-2.5-flash")
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT_REAL,
            middleware=[
                harness,
                ModelCallLimitMiddleware(run_limit=10),
            ],
        )
        out = agent.invoke(
            {"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]}
        )
        raw_answer = out["messages"][-1].content
        if isinstance(raw_answer, list):
            answer = " ".join(
                (b.get("text", "") if isinstance(b, dict) else str(b))
                for b in raw_answer
            )
        elif not isinstance(raw_answer, str):
            answer = str(raw_answer)
        else:
            answer = raw_answer
    except StopAgent as e:
        stop_reason = str(e)
        answer = ""
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print("\n[THÔNG BÁO QUOTA] Project Google AI Studio đã chạm hạn mức Free Tier 20 lượt/ngày.")
            print("-> Tự động kích hoạt Fallback Engine để hoàn tất hành trình và bảo toàn kiểm soát Harness!")
            engine_used = "Fallback Engine (do chạm giới hạn 20 req/ngày của Google AI Studio)"
            
            # Reset state để chạy lại một lượt chuẩn xác
            STATE.reset(case)
            LOOP_DETECTOR.reset()
            fallback_tools = make_tools(STATE)
            fallback_model = fake_model_react(case)
            fallback_agent = create_agent(
                model=fallback_model,
                tools=fallback_tools,
                system_prompt=SYSTEM_PROMPT_REAL,
                middleware=[
                    harness,
                    ModelCallLimitMiddleware(run_limit=10),
                ],
            )
            out = fallback_agent.invoke(
                {"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]}
            )
            raw_answer = out["messages"][-1].content
            answer = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
        else:
            stop_reason = f"LLM Error: {e}"
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
            "pattern": f"ReAct ({engine_used})",
            "result": "DONE",
            "model_calls": len(STATE.log),
            "booking": list(STATE.bookings.values()),
        }
    else:
        output = {
            "pattern": f"ReAct ({engine_used})",
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
    print("BTVN#3 — Mẫu 1: ReAct Agent (LLM THẬT - Google Gemini 2.5 Flash)")
    print("Chọn case:")
    print("  1. success  - có chuyến bay hợp lệ")
    print("  2. fail     - không có chuyến bay thỏa, harness phải bàn giao")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Lựa chọn (1/2): ").strip()
    case = "success" if choice == "1" else "fail"
    run_react_real(case)
