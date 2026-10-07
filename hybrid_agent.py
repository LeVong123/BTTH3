"""BTVN#3 — Mẫu 3: Hybrid Agent (ReAct + Plan)

Kiến trúc LAI:
    Kết hợp Plan-then-Execute và ReAct:
    1. PLAN: Model sinh kế hoạch tổng thể (giống Plan-then-Execute)
    2. EXECUTE k bước theo kế hoạch (giống Plan-then-Execute)
    3. REPLAN: Nếu một bước thất bại hoặc observation thay đổi đáng kể,
       model được gọi lại để VIẾT LẠI kế hoạch (giống ReAct — thích nghi)

    Lập kế hoạch → Thực thi k bước → Observation đổi đáng kể?
        có  → lập lại kế hoạch
        không → tiếp tục → Xong

So với 2 mẫu kia:
    ReAct:              linh hoạt nhất, nhưng tốn model call nhất
    Plan-then-Execute:  duyệt trước được, nhưng không thích nghi
    Hybrid:             duyệt trước + thích nghi được, nhưng khó debug hơn

Harness 4 lớp:
    1. Ràng buộc là dữ liệu    → Constraints
    2. Kiểm quyền               → check_permission()
    3. Tiêu chí hoàn thành      → is_done()
    4. Bàn giao                  → handoff()

Run:  python hybrid_agent.py
"""
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel

from shared import (
    FLIGHTS,
    AgentState, Constraints,
    check_permission, handoff, is_done, make_tools,
)


# =====================================================================
# CẤU HÌNH
# =====================================================================
CONSTRAINTS = Constraints()
STATE = AgentState()


# =====================================================================
# TOOL — wrapped bằng @tool cho LangChain
# =====================================================================
@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    return {"status": "ok", "flights": FLIGHTS[STATE.case]}


@tool
def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code."""
    info = next((f for f in FLIGHTS[STATE.case] if f["flight"] == flight), None)
    if info is None:
        return {"status": "not_found", "flight": flight}
    code = f"{flight}-12A"
    STATE.bookings[code] = {"code": code, "paid": False, **info}
    return {"status": "ok", **STATE.bookings[code]}


@tool
def pay(code: str) -> dict:
    """Pay for a held booking. This spends money and cannot be undone."""
    if code not in STATE.bookings:
        return {"status": "not_found", "code": code}
    STATE.bookings[code]["paid"] = True
    return {"status": "ok", **STATE.bookings[code]}


@tool
def get_booking(code: str) -> dict:
    """Read a booking back from the system."""
    if code in STATE.bookings:
        return {"status": "ok", **STATE.bookings[code]}
    return {"status": "not_found"}


TOOLS = {t.name: t for t in [search_flights, book_seat, pay, get_booking]}


# =====================================================================
# KẾ HOẠCH — structured output
# =====================================================================
class Step(BaseModel):
    tool: str          # book_seat | pay | get_booking
    args: dict


class Plan(BaseModel):
    steps: list[Step]  # rỗng = "không có chuyến bay thỏa"


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You plan a flight booking. Write the WHOLE plan at once, as a list of steps. "
     "Tools: book_seat(flight), pay(code), get_booking(code). "
     "The booking code is not known yet: write \"$booking_code\" and it will be filled in. "
     "If no flight meets ALL constraints, return an empty plan. "
     "If given failed_steps, avoid those actions and find an alternative."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}\n"
              "Previous failures: {failed_steps}"),
])

MAX_REPLANS = 3   # ngân sách: tối đa 3 lần lập/lập lại kế hoạch


# =====================================================================
# FAKE MODEL — kịch bản cho hybrid
# =====================================================================
def fake_model_hybrid(case: str):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.runnables import RunnableLambda

    book_pay_check = lambda flight: Plan(steps=[
        Step(tool="book_seat", args={"flight": flight}),
        Step(tool="pay", args={"code": "$booking_code"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])

    plans = {
        # Success: kế hoạch đầu tiên chọn đúng VN122
        "success": [book_pay_check("VN122")],
        # Fail: kế hoạch 1 chọn VJ604 (quá giá) → bị chặn →
        #       kế hoạch 2 chọn QH118 (chiều) → bị chặn → empty
        "fail": [
            book_pay_check("VJ604"),     # replan 1: chọn sáng nhưng quá giá
            book_pay_check("QH118"),     # replan 2: chọn rẻ nhưng chiều
            Plan(steps=[]),              # replan 3: bỏ cuộc
        ],
    }[case]
    plans_iter = iter(plans)

    class ScriptedModel(GenericFakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: next(plans_iter))

    return ScriptedModel(messages=iter([]))


# =====================================================================
# HELPER
# =====================================================================
def fill_placeholders(args: dict) -> dict:
    """Thay '$booking_code' bằng code thật từ book_seat."""
    code = next(
        (r["code"] for t, _, r in reversed(STATE.log)
         if t == "book_seat" and r.get("status") == "ok"),
        ""
    )
    return {k: (code if v == "$booking_code" else v) for k, v in args.items()}


# =====================================================================
# CHẠY AGENT
# =====================================================================
def run_hybrid(case: str, auto_approve: bool = False) -> dict:
    """Chạy Hybrid agent, trả về dict kết quả."""
    STATE.reset(case)
    model = fake_model_hybrid(case)
    model_calls = 0

    print(f"\n{'='*60}")
    print(f"  MẪU 3: Hybrid (ReAct + Plan)  |  Case: {case}")
    print(f"{'='*60}")
    print(f"Ràng buộc (dữ liệu): {CONSTRAINTS}")
    print(f"Yêu cầu người dùng: {CONSTRAINTS.to_prompt()}")

    # --- SEARCH: gọi bằng DỮ LIỆU ràng buộc
    c = CONSTRAINTS
    found = TOOLS["search_flights"].invoke(
        {"origin": c.origin, "destination": c.destination, "date": c.date}
    )
    STATE.log.append((
        "search_flights",
        {"origin": c.origin, "destination": c.destination, "date": c.date},
        found,
    ))
    model_calls += 0  # search không cần model
    print(f"\nChuyến bay tìm được: {json.dumps(found['flights'], ensure_ascii=False)}")

    # --- VÒNG LẶP PLAN → EXECUTE → REPLAN
    planner = PLANNER_PROMPT | model.with_structured_output(Plan)
    failed_steps: list[str] = []

    for replan in range(1, MAX_REPLANS + 1):
        # 1. LẬP KẾ HOẠCH
        draft = planner.invoke({
            "goal": c.to_prompt(),
            "flights": json.dumps(found["flights"]),
            "failed_steps": json.dumps(failed_steps) if failed_steps else "None",
        })
        model_calls += 1

        print(f"\n--- kế hoạch #{replan} ---")
        if not draft.steps:
            print("  (kế hoạch rỗng: model cho rằng không có chuyến bay thỏa)")
            break

        for i, step in enumerate(draft.steps, 1):
            print(f"  Bước {i}: {step.tool}({step.args})")

        # 2. THỰC THI TỪNG BƯỚC
        need_replan = False
        print(f"\n--- thực thi kế hoạch #{replan} ---")
        for i, step in enumerate(draft.steps, 1):
            args = fill_placeholders(step.args)

            # Harness: kiểm quyền
            reason = check_permission(
                step.tool, args, CONSTRAINTS, FLIGHTS[STATE.case]
            )
            if reason:
                result = {"status": "denied", "reason": reason}
            else:
                result = TOOLS[step.tool].invoke(args)

            STATE.log.append((step.tool, args, result))
            status = result.get("status", "?")
            reason_str = result.get("reason", "")
            print(f"  Bước {i}: {step.tool}({args}) -> {status} {reason_str}")

            # 3. KIỂM TRA: observation thay đổi đáng kể? → REPLAN
            if status != "ok":
                failed_steps.append(f"{step.tool}({args}) -> {status}: {reason_str}")
                print(f"  ⚠ Bước thất bại → kích hoạt REPLAN (khác Plan-then-Execute: sẽ dừng)")
                need_replan = True
                break

        # Đã done? → thoát
        if is_done(STATE, CONSTRAINTS):
            break

        # Không cần replan (tất cả bước ok nhưng chưa done)
        if not need_replan:
            break

        print(f"\n  → Replan: model sẽ viết kế hoạch mới dựa trên thất bại")

    # --- Trace
    print("\n--- trace ---")
    for i, (tool_name, args, result) in enumerate(STATE.log, 1):
        status = result.get("status", "?")
        print(f"  [{i}] {tool_name}({args}) -> {status}")

    # --- KẾT QUẢ
    if is_done(STATE, CONSTRAINTS):
        output = {
            "pattern": "Hybrid (ReAct + Plan)",
            "result": "DONE",
            "model_calls": model_calls,
            "replans": replan if 'replan' in dir() else 0,
            "tool_calls": len(STATE.log),
            "booking": list(STATE.bookings.values()),
        }
    else:
        output = {
            "pattern": "Hybrid (ReAct + Plan)",
            "result": "FAILED",
            "model_calls": model_calls,
            "replans": replan if 'replan' in dir() else 0,
            "tool_calls": len(STATE.log),
            "handoff": handoff(STATE),
        }

    print("\n--- kết quả (harness quyết định) ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


# =====================================================================
if __name__ == "__main__":
    print("BTVN#3 — Mẫu 3: Hybrid Agent (ReAct + Plan)")
    print("Chọn case:")
    print("  1. success  - có chuyến bay hợp lệ")
    print("  2. fail     - không có chuyến bay thỏa, agent replan rồi bàn giao")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Lựa chọn (1/2): ").strip()
    case = "success" if choice == "1" else "fail"
    run_hybrid(case)
