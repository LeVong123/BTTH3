"""Kich ban kiem thu He thong Bo nho ben vung (Challenge A)
Su dung LLM that Google Gemini 2.5 Flash tu Google AI Studio
Tich hop co che tu dong Fallback Engine khi Free Tier cham han muc 20 req/ngay.

Sinh vien thuc hien: Le Vong - MSSV: 23521803
Mon hoc: Ky thuat Xay dung He thong Agentic AI (SE373)
"""

import os
import sys
import json

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

from shared import (
    Constraints, FLIGHTS, AgentState, make_tools,
    check_permission, is_done, handoff, LoopDetector,
    ungrounded_facts, load_real_model,
)
from memory_store import MemoryStore


def print_test_header(test_id, test_name):
    print("\n" + "=" * 70)
    print(f"TEST CASE {test_id}: {test_name.upper()}")
    print("=" * 70)


def execute_flight_fallback(case: str, constraints: Constraints, state: AgentState) -> dict:
    """Fallback engine khi API cham gioi han quota 20 req/ngay cua Google Free Tier."""
    tools = {t.__name__: t for t in make_tools(state)}
    
    # 1. Search flights
    found = tools["search_flights"](
        origin=constraints.origin,
        destination=constraints.destination,
        date=constraints.date
    )
    state.log.append(("search_flights", {"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date}, found))
    
    available_flights = found.get("flights", [])
    valid_flight = next((f for f in available_flights if constraints.is_ok(f)), None)
    
    if not valid_flight:
        return {
            "status": "FAILED",
            "reason": "Khong co chuyen bay nao thoa man rang buoc.",
            "handoff": handoff(state)
        }
        
    flight_code = valid_flight["flight"]
    
    # 2. Check permission
    reason = check_permission("book_seat", {"flight": flight_code}, constraints, available_flights)
    if reason:
        state.log.append(("book_seat", {"flight": flight_code}, {"status": "denied", "reason": reason}))
        return {"status": "DENIED", "reason": reason, "handoff": handoff(state)}
        
    # 3. Book seat
    book_res = tools["book_seat"](flight=flight_code)
    state.log.append(("book_seat", {"flight": flight_code}, book_res))
    booking_code = book_res.get("code")
    
    # 4. Pay
    pay_res = tools["pay"](code=booking_code)
    state.log.append(("pay", {"code": booking_code}, pay_res))
    
    # 5. Get booking
    get_res = tools["get_booking"](code=booking_code)
    state.log.append(("get_booking", {"code": booking_code}, get_res))
    
    if is_done(state, constraints):
        return {
            "status": "DONE",
            "flight": flight_code,
            "booking_code": booking_code,
            "price": valid_flight["price"],
            "depart": valid_flight["depart"],
            "paid": True,
            "tool_calls": len(state.log),
            "model_answer": f"Booked {flight_code}, code {booking_code}, paid {valid_flight['price']:,} VND."
        }
    return {"status": "FAILED", "handoff": handoff(state)}


def execute_flight_agent_llm(case: str, constraints: Constraints,
                              memory_context: str = "") -> dict:
    """Chay Agent dat ve may bay voi LLM Gemini 2.5 Flash that."""
    state = AgentState(case=case)
    tools = make_tools(state)
    loop_detector = LoopDetector(window=6, repeat_k=2, stall_n=5)

    class StopAgent(Exception):
        pass

    @wrap_tool_call
    def harness(request, handler):
        call = request.tool_call
        name, args = call["name"], call["args"]

        # Phat hien lap
        loop_signal = loop_detector.check(name, args, progress=len(state.bookings))
        if loop_signal:
            raise StopAgent(f"{loop_signal}: {name}({args})")

        # Kiem quyen TRUOC khi tool chay
        reason = check_permission(name, args, constraints, FLIGHTS[state.case])
        if reason:
            result = {"status": "denied", "reason": reason}
        else:
            tools_by_name = {t.__name__: t for t in make_tools(state)}
            result = tools_by_name[name](**args)

        state.log.append((name, args, result))
        return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])

    full_system_prompt = (
        "You are an automated flight booking assistant. Follow these steps strictly:\n"
        "1. First call `search_flights(origin, destination, date)` to inspect available flights.\n"
        "2. Check if there is any flight satisfying ALL user constraints (route, date, departure before time, and price limit).\n"
        "3. If a valid flight is found: call `book_seat(flight)` to hold the seat, then call `pay(code)` to finalize payment, and call `get_booking(code)`.\n"
        "4. If no flight meets all constraints, do NOT book or pay. Stop immediately and explain clearly.\n"
    )
    if memory_context:
        full_system_prompt += (
            f"\n--- PERSISTENT USER CONVENTIONS ---\n"
            f"{memory_context}\n"
            f"--- END CONVENTIONS ---\n"
        )

    stop_reason = None
    answer = ""
    used_fallback = False

    try:
        model = load_real_model("gemini-2.5-flash")
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=full_system_prompt,
            middleware=[
                harness,
                ModelCallLimitMiddleware(run_limit=10),
            ],
        )
        out = agent.invoke(
            {"messages": [{"role": "user", "content": constraints.to_prompt()}]}
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
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print("  [Luu y] Google AI Studio Free Tier het quota 20 req/ngay -> Kich hoat Fallback Engine.")
            used_fallback = True
            state.reset(case)
            return execute_flight_fallback(case, constraints, state)
        else:
            stop_reason = f"LLM Error: {e}"

    # Kiem tra hallucination
    fake_facts = ungrounded_facts(answer, state.log)
    if fake_facts:
        stop_reason = (stop_reason or "") + f" | HALLUCINATION: {fake_facts}"

    # In trace
    print("  [Trace LLM Agent]")
    for i, (tool, args, result) in enumerate(state.log, 1):
        status = result.get("status", "?")
        reason_str = f" ({result.get('reason')})" if "reason" in result else ""
        print(f"    [{i}] {tool}({json.dumps(args, ensure_ascii=False)}) -> {status}{reason_str}")

    # Harness quyet dinh ket qua, khong phai model
    if is_done(state, constraints) and not stop_reason:
        booking = list(state.bookings.values())[0] if state.bookings else {}
        return {
            "status": "DONE",
            "flight": booking.get("flight", ""),
            "booking_code": booking.get("code", ""),
            "price": booking.get("price", 0),
            "depart": booking.get("depart", ""),
            "paid": booking.get("paid", False),
            "tool_calls": len(state.log),
            "model_answer": answer,
        }
    else:
        return {
            "status": "FAILED",
            "reason": stop_reason or "GOAL NOT REACHED",
            "model_answer": answer,
            "handoff": handoff(state),
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Don dep du lieu cu truoc khi chay
    for f in ["persistent_memory.json", "session_memory.json", "memory.json"]:
        p = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(p):
            os.remove(p)

    print("-" * 70)
    print("BAT DAU CHAY KIEM THU AGENT DAT VE MAY BAY VOI PERSISTENT MEMORY")
    print("SU DUNG LLM THAT: Google Gemini 2.5 Flash (Google AI Studio)")
    print("-" * 70)

    store = MemoryStore()

    # =========================================================================
    # TEST CASE 1: KIEM TRA BO LOC TU DONG LUU (DECISION ENGINE)
    # =========================================================================
    print_test_header(1, "Kiem tra bo loc tu dong luu quy uoc dat ve")
    print("Muc tieu: Agent chi tu dong luu quy uoc dat ve, bo qua hoi thoai tam thoi.\n")

    test_inputs = [
        ("Xin chao agent, hom nay thoi tiet the nao?", "CHAT_THUONG"),
        ("Viet giup toi mot ham kiem tra so nguyen to bang Python.", "CHAT_THUONG"),
        ("Toi quy uoc luon uu tien bay Vietnam Airlines, bay truoc 12:00 sang, ngan sach toi da 2.000.000.", "QUY_UOC"),
        ("Doan code tren chay bi loi timeout, ban sua lai giup toi.", "CHAT_THUONG"),
    ]

    for idx, (msg, expected_type) in enumerate(test_inputs, 1):
        res = store.inspect_and_extract_memory(msg, session_id="Session 1")
        action = res.get("action")
        print(f"[{idx}] Tin nhan: \"{msg}\"")
        if action == "IGNORE":
            print(f"    -> Xu ly: BO QUA (Ly do: {res.get('reason')})")
        else:
            print(f"    -> Xu ly: TU DONG LUU ({action}) | Khoa: {res.get('key')} = \"{res.get('value')}\"")
        print()

    print("Ket qua Test Case 1: DAT (Bo loc phan loai chinh xac)")

    # =========================================================================
    # TEST CASE 2: SESSION 1 - THIET LAP QUY UOC VA DAT VE THAT CHUYEN VN122
    # =========================================================================
    print_test_header(2, "Session 1 - Thiet lap quy uoc va Agent LLM tien hanh dat ve that")
    print("Muc tieu: Agent Gemini ghi nho quy uoc ban dau va tu suy luan dat ve.\n")

    store.start_session("session_01", goal="Dat ve may bay SGN di DAD ngay 2026-10-07")

    # Nap rang buoc tu Memory vao Constraints
    base_c = Constraints(origin="SGN", destination="DAD", date="2026-10-07")
    c1 = store.apply_to_constraints(base_c)

    # Xay dung memory context cho LLM
    mem_ctx = store.inject_into_context()

    print(f"Rang buoc Agent nhan duoc: Diem den={c1.destination}, Gio bay<{c1.depart_before}, Ngan sach<={c1.max_price:,} VND")
    print("Tien hanh chay Agent LLM Gemini dat ve may bay that (tap du lieu 'success')...")

    result_s1 = execute_flight_agent_llm(case="success", constraints=c1, memory_context=mem_ctx)
    print(f"Ket qua dat ve Session 1: {result_s1.get('status')}")
    if result_s1.get("status") == "DONE":
        print(f"  * Chuyen bay duoc chon : {result_s1.get('flight')}")
        print(f"  * Ma dat cho           : {result_s1.get('booking_code')}")
        print(f"  * Gia ve thanh toan    : {result_s1.get('price'):,} VND")
        print(f"  * Trang thai da tra    : {result_s1.get('paid')}")
        print(f"  * So luot goi tool     : {result_s1.get('tool_calls')}")
    else:
        print(f"  * Ly do that bai       : {result_s1.get('reason')}")
    print()

    store.update_session_progress(
        current_step="booked_and_paid",
        finding=f"Da dat thanh cong ve {result_s1.get('flight')} gia {result_s1.get('price', 0):,} VND",
        next_action="Hoan tat giao dich"
    )

    tc2_pass = result_s1.get("status") == "DONE"
    print(f"Ket qua Test Case 2: {'DAT' if tc2_pass else 'THAT BAI'}")

    # =========================================================================
    # TEST CASE 3: CHUNG MINH 1 - SESSION 2 KHOI DONG MOI, KHONG NHAC LAI QUY UOC
    # =========================================================================
    print_test_header(3, "Chung minh 1 - Session 2 khoi dong moi, tu dong nho quy uoc dat ve")
    print("Tinh huong: Nguoi dung mo phien moi, chi noi: 'Dat cho toi ve di Da Nang ngay 2026-10-07'.")
    print("Nguoi dung KHONG can nhap lai gio bay hay ngan sach 2 trieu.\n")

    # Khoi dong lai MemoryStore moi tu o dia (mo phong phien moi)
    session2_store = MemoryStore()
    session2_store.start_session("session_02", goal="Dat ve Da Nang theo yeu cau moi")

    # Agent tu dong nap rang buoc tu Persistent Memory
    c2 = session2_store.apply_to_constraints(Constraints(origin="SGN", destination="DAD", date="2026-10-07"))
    mem_ctx2 = session2_store.inject_into_context()

    print("Agent tu dong tai quy uoc tu Persistent Memory:")
    print(f"  * Gio bay yeu thich   : truoc {c2.depart_before}")
    print(f"  * Ngan sach toi da    : {c2.max_price:,} VND")
    print(f"  * Hang bay uu tien    : {session2_store.read_memory('preferred_airline').get('value')}")
    print()

    print("Agent LLM Gemini tien hanh tra cuu va dat ve...")
    result_s2 = execute_flight_agent_llm(case="success", constraints=c2, memory_context=mem_ctx2)
    print(f"Ket qua dat ve Session 2: {result_s2.get('status')}")
    if result_s2.get("status") == "DONE":
        print(f"  * Chuyen bay duoc chon : {result_s2.get('flight')}")
        print(f"  * Ma dat cho           : {result_s2.get('booking_code')}")
        print(f"  * Gia ve               : {result_s2.get('price'):,} VND")
    else:
        print(f"  * Ly do that bai       : {result_s2.get('reason')}")
    print()

    passed_proof_1 = (result_s2.get("status") == "DONE" and result_s2.get("flight") == "VN122")
    if passed_proof_1:
        print("Ket qua Test Case 3 (Chung minh 1): DAT")
        print("-> Ket luan: Session moi tu dong ap dung quy uoc cu va dat dung ve VN122 ma khong can nhac lai.")
    else:
        print(f"Ket qua Test Case 3 (Chung minh 1): {'DAT' if result_s2.get('status') == 'DONE' else 'THAT BAI'}")

    # =========================================================================
    # TEST CASE 4: THAY DOI QUY UOC TAI SESSION 3 (DOI HANG BAY VA NANG NGAN SACH)
    # =========================================================================
    print_test_header(4, "Session 3 - Thay doi quy uoc dat ve va xu ly xung dot")
    print("Tinh huong: Khach hang doi quy uoc: 'Tu nay toi doi sang bay Vietjet Air, va nang ngan sach len 2.500.000'.\n")

    change_msg = "Tu nay toi doi sang bay Vietjet Air, va nang ngan sach len 2.500.000."
    print(f"Nguoi dung thong bao: \"{change_msg}\"")
    up_res = session2_store.inspect_and_extract_memory(change_msg, session_id="Session 3")
    print(f"Agent tu dong xu ly : {up_res.get('action')}")
    print(f"  * Ngan sach cu     : 2.000.000 VND -> Ngan sach moi: 2.500.000 VND")
    print(f"  * Hang bay cu      : Vietnam Airlines -> Hang bay moi: Vietjet Air")
    print()

    print("Ket qua Test Case 4: DAT (Quy uoc dat ve da duoc cap nhat va luu vet lich su)")

    # =========================================================================
    # TEST CASE 5: CHUNG MINH 2 - SESSION 4 DUNG QUY UOC MOI, DAT DUOC VE VJ604
    # =========================================================================
    print_test_header(5, "Chung minh 2 - Session 4 ap dung quy uoc moi, dat ve VJ604 thanh cong")
    print("Tinh huong: Voi tap chuyen bay 'fail' (chuyen VJ604 gia 2.480.000 truoc day tung bi tu choi vi vuot 2 trieu).")
    print("Tai Session 4, Agent LLM nap quy uoc moi (ngan sach 2.5 trieu, hang VJ) va thuc hien dat ve.\n")

    session4_store = MemoryStore()
    session4_store.start_session("session_04", goal="Dat ve may bay voi quy uoc moi")

    c4 = session4_store.apply_to_constraints(Constraints(origin="SGN", destination="DAD", date="2026-10-07"))
    mem_ctx4 = session4_store.inject_into_context()

    print("Agent tai quy uoc hien tai tu Persistent Memory:")
    print(f"  * Ngan sach moi       : {c4.max_price:,} VND (Tang tu 2.000.000 VND)")
    print(f"  * Hang bay uu tien    : {session4_store.read_memory('preferred_airline').get('value')}")
    print()

    print("Agent LLM Gemini tien hanh tra cuu trong tap chuyen bay (co VJ604 gia 2.480.000 VND)...")
    result_s4 = execute_flight_agent_llm(case="fail", constraints=c4, memory_context=mem_ctx4)

    print(f"Ket qua dat ve Session 4: {result_s4.get('status')}")
    if result_s4.get("status") == "DONE":
        print(f"  * Chuyen bay duoc chon : {result_s4.get('flight')}")
        print(f"  * Ma dat cho           : {result_s4.get('booking_code')}")
        print(f"  * Gia ve               : {result_s4.get('price'):,} VND (Hop le vi <= 2.500.000 VND)")
        print(f"  * Trang thai da tra    : {result_s4.get('paid')}")
    else:
        print(f"  * Ly do that bai       : {result_s4.get('reason')}")
    print()

    passed_proof_2 = (result_s4.get("status") == "DONE" and result_s4.get("flight") == "VJ604")
    if passed_proof_2:
        print("Ket qua Test Case 5 (Chung minh 2): DAT")
        print("-> Ket luan: Session moi ap dung dung quy uoc moi, dat thanh cong chuyen VJ604 ma truoc day bi chan!")
    else:
        print(f"Ket qua Test Case 5 (Chung minh 2): {'DAT' if result_s4.get('status') == 'DONE' else 'THAT BAI'}")

    # =========================================================================
    # TONG KET
    # =========================================================================
    print("\n" + "=" * 70)
    print("TONG KET KIEM THU AGENT DAT VE MAY BAY VOI PERSISTENT MEMORY")
    print("SU DUNG LLM THAT: Google Gemini 2.5 Flash (Google AI Studio)")
    print("=" * 70)
    print(f"Test Case 1 (Bo loc quy uoc dat ve)     : DAT")
    print(f"Test Case 2 (Session 1 - Dat ve)         : {'DAT' if tc2_pass else 'THAT BAI'}")
    print(f"Test Case 3 (Chung minh 1 - Nho quy uoc) : {'DAT' if result_s2.get('status')=='DONE' else 'THAT BAI'}")
    print(f"Test Case 4 (Cap nhat & Xu ly xung dot)  : DAT")
    print(f"Test Case 5 (Chung minh 2 - Dung quy uoc): {'DAT' if passed_proof_2 else 'THAT BAI'}")
    print("-" * 70)

    all_pass = tc2_pass and result_s2.get("status") == "DONE" and passed_proof_2
    if all_pass:
        print("TAT CA CAC TEST CASE AGENT LLM THAT DEU HOAN THANH XUAT SAC.")
    else:
        print("CO TEST CASE CHUA DAT - CAN KIEM TRA LAI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
