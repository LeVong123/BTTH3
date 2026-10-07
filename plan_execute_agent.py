"""BTVN#3 — Mẫu 2: Plan-then-Execute Agent

Kiến trúc:
    Model được gọi MỘT LẦN để sinh TRỌN kế hoạch.
    Người duyệt kế hoạch trước khi chạy.
    Code thực thi từng bước KHÔNG gọi lại model.

    Goal → Model sinh kế hoạch (1 call) → Người duyệt
         → approved: thực thi Step 1, 2, 3 → Kết quả
         → rejected: model viết kế hoạch mới

So với ReAct:
    ReAct:              model quyết định MỘT bước tại một thời điểm
    Plan-then-Execute:  model gọi MỘT LẦN cho toàn bộ kế hoạch
    + Kế hoạch nhìn thấy TRƯỚC khi chạy → duyệt được, ước lượng chi phí được
    - Kế hoạch cố định, không thích nghi khi bước sai → dừng, bàn giao

Harness 4 lớp:
    1. Ràng buộc là dữ liệu    → Constraints
    2. Kiểm quyền               → check_permission(), chạy TRƯỚC mỗi step
    3. Tiêu chí hoàn thành      → is_done(), kiểm bằng code
    4. Bàn giao                  → handoff(), khi agent thất bại

Run:  python plan_execute_agent.py
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
# TOOL — wrapped bằng @tool cho LangChain structured output
# =====================================================================
@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    return {"status": "ok", "flights": FLIGHTS[STATE.case]}


@tool
def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
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
    args: dict         # {"flight": "VN122"} hoặc {"code": "$booking_code"}


class Plan(BaseModel):
    steps: list[Step]  # danh sách rỗng = "không có chuyến bay thỏa"


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You plan a flight booking. Write the WHOLE plan at once, as a list of steps. "
     "Tools: book_seat(flight), pay(code), get_booking(code). "
     "The booking code is not known yet: write \"$booking_code\" and it will be filled in. "
     "If no flight meets ALL constraints, return an empty plan."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}"),
])

MAX_PLANS = 2   # ngân sách: model viết tối đa 2 kế hoạch


# =====================================================================
# FAKE MODEL — chạy không cần API key
# =====================================================================
def fake_model_plan(case: str):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.runnables import RunnableLambda

    book_pay_check = lambda flight: Plan(steps=[
        Step(tool="book_seat", args={"flight": flight}),
        Step(tool="pay", args={"code": "$booking_code"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])
    plans = {
        "success": [book_pay_check("VN122"), book_pay_check("VN122")],
        "fail": [book_pay_check("VJ604"), Plan(steps=[])],
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
    """Thay thế '$booking_code' bằng code thật từ book_seat."""
    code = next(
        (r["code"] for t, _, r in reversed(STATE.log)
         if t == "book_seat" and r.get("status") == "ok"),
        ""
    )
    return {k: (code if v == "$booking_code" else v) for k, v in args.items()}


# =====================================================================
# CHẠY AGENT
# =====================================================================
def run_plan_execute(case: str, auto_approve: bool = False) -> dict:
    """Chạy Plan-then-Execute agent, trả về dict kết quả."""
    STATE.reset(case)
    model = fake_model_plan(case)

    print(f"\n{'='*60}")
    print(f"  MẪU 2: Plan-then-Execute  |  Case: {case}")
    print(f"{'='*60}")
    print(f"Ràng buộc (dữ liệu): {CONSTRAINTS}")
    print(f"Yêu cầu người dùng: {CONSTRAINTS.to_prompt()}")

    # --- SEARCH: gọi bằng DỮ LIỆU ràng buộc, không cần model
    c = CONSTRAINTS
    found = TOOLS["search_flights"].invoke(
        {"origin": c.origin, "destination": c.destination, "date": c.date}
    )
    STATE.log.append((
        "search_flights",
        {"origin": c.origin, "destination": c.destination, "date": c.date},
        found,
    ))
    print(f"\nChuyến bay tìm được: {json.dumps(found['flights'], ensure_ascii=False)}")

    # --- LẬP KẾ HOẠCH + DUYỆT
    planner = PLANNER_PROMPT | model.with_structured_output(Plan)
    plan = None
    model_calls = 1  # đã gọi search_flights

    for attempt in range(1, MAX_PLANS + 1):
        draft = planner.invoke({
            "goal": c.to_prompt(),
            "flights": json.dumps(found["flights"]),
        })
        model_calls += 1

        print(f"\n--- kế hoạch #{attempt} (model sinh trong MỘT lần gọi) ---")
        if not draft.steps:
            print("  (kế hoạch rỗng: model cho rằng không có chuyến bay thỏa)")
            break
        for i, step in enumerate(draft.steps, 1):
            print(f"  Bước {i}: {step.tool}({step.args})")

        if auto_approve:
            answer = "y"
            print("  [Tự động duyệt: y]")
        else:
            answer = ""
            while answer not in ("y", "n"):
                answer = input("Người duyệt — duyệt kế hoạch này? (y/n): ").strip().lower()

        if answer == "y":
            plan = draft
            break
        print("  Từ chối → yêu cầu model viết kế hoạch mới.")

    # --- THỰC THI: code chạy từng bước, KHÔNG gọi lại model
    if plan:
        print("\n--- thực thi ---")
        for i, step in enumerate(plan.steps, 1):
            args = fill_placeholders(step.args)

            # Harness: kiểm quyền TRƯỚC khi step chạy
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

            if status != "ok":
                print("  Bước thất bại. Plan-then-Execute không thể sửa kế hoạch → dừng.")
                break

    # --- Trace
    print("\n--- trace ---")
    for i, (tool_name, args, result) in enumerate(STATE.log, 1):
        status = result.get("status", "?")
        print(f"  [{i}] {tool_name}({args}) -> {status}")

    # --- KẾT QUẢ: harness quyết định, không phải model
    if is_done(STATE, CONSTRAINTS):
        output = {
            "pattern": "Plan-then-Execute",
            "result": "DONE",
            "model_calls": model_calls,
            "tool_calls": len(STATE.log),
            "booking": list(STATE.bookings.values()),
        }
    else:
        output = {
            "pattern": "Plan-then-Execute",
            "result": "FAILED",
            "model_calls": model_calls,
            "tool_calls": len(STATE.log),
            "handoff": handoff(STATE),
        }

    print("\n--- kết quả (harness quyết định) ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


# =====================================================================
if __name__ == "__main__":
    print("BTVN#3 — Mẫu 2: Plan-then-Execute Agent")
    print("Chọn case:")
    print("  1. success  - có chuyến bay hợp lệ")
    print("  2. fail     - không có chuyến bay thỏa, harness bàn giao")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Lựa chọn (1/2): ").strip()
    case = "success" if choice == "1" else "fail"
    run_plan_execute(case)
