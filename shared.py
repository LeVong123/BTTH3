"""BTVN#3 - Shared components for all 3 agent patterns.

Chứa các thành phần dùng chung:
    - Constraints: ràng buộc là DỮ LIỆU (harness layer 1)
    - Mock tools: search_flights, book_seat, pay, get_booking
    - Permission check (harness layer 2): kiểm quyền TRƯỚC khi tool chạy
    - is_done (harness layer 3): tiêu chí hoàn thành kiểm bằng CODE
    - handoff (harness layer 4): bàn giao cho con người
    - LoopDetector: phát hiện lặp (3 tín hiệu)
"""
import json
import os
import re
from collections import deque
from dataclasses import dataclass, field


# =====================================================================
# 1. RÀNG BUỘC LÀ DỮ LIỆU  (Harness Layer 1)
#    Yêu cầu người dùng được lưu thành data. Prompt được SINH TỪ data,
#    và harness KIỂM data bằng code. Model không thể "quên" ràng buộc.
# =====================================================================
@dataclass
class Constraints:
    """Ràng buộc đặt vé — dữ liệu cứng, không phải prompt."""
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"       # chuyến sáng
    max_price: int = 2_000_000         # VND

    def to_prompt(self) -> str:
        """Sinh prompt từ dữ liệu ràng buộc."""
        return (
            f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
            f"departing before {self.depart_before}, "
            f"price at most {self.max_price:,} VND."
        )

    def is_ok(self, flight: dict) -> bool:
        """Kiểm tra một chuyến bay có thỏa TẤT CẢ ràng buộc không."""
        return (
            flight.get("depart", "").startswith(self.date)
            and flight.get("depart", "")[11:16] < self.depart_before
            and flight.get("price", float("inf")) <= self.max_price
        )


# =====================================================================
# MOCKUP TOOLS — dữ liệu giả, không cần mạng
# =====================================================================
FLIGHTS = {
    "success": [
        {"flight": "VN122", "depart": "2026-10-07T08:10", "price": 1_850_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
    "fail": [  # không chuyến nào VỪA sáng VỪA dưới 2 triệu
        {"flight": "VJ604", "depart": "2026-10-07T08:10", "price": 2_480_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
}


# Shared mutable state — mỗi lần chạy agent cần reset
class AgentState:
    """Trạng thái dùng chung trong một phiên chạy agent."""
    def __init__(self, case: str = "success"):
        self.case: str = case
        self.bookings: dict = {}
        self.log: list = []        # [(tool_name, args, result)]

    def reset(self, case: str | None = None):
        if case is not None:
            self.case = case
        self.bookings.clear()
        self.log.clear()


def make_tools(state: AgentState):
    """Tạo bộ 4 tool mockup gắn với state."""

    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Search flights by route and date (YYYY-MM-DD)."""
        return {"status": "ok", "flights": FLIGHTS[state.case]}

    def book_seat(flight: str) -> dict:
        """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
        info = next((f for f in FLIGHTS[state.case] if f["flight"] == flight), None)
        if info is None:
            return {"status": "not_found", "flight": flight}
        code = f"{flight}-12A"
        state.bookings[code] = {"code": code, "paid": False, **info}
        return {"status": "ok", **state.bookings[code]}

    def pay(code: str) -> dict:
        """Pay for a held booking. This spends money and cannot be undone."""
        if code not in state.bookings:
            return {"status": "not_found", "code": code}
        state.bookings[code]["paid"] = True
        return {"status": "ok", **state.bookings[code]}

    def get_booking(code: str) -> dict:
        """Read a booking back from the system."""
        if code in state.bookings:
            return {"status": "ok", **state.bookings[code]}
        return {"status": "not_found"}

    return [search_flights, book_seat, pay, get_booking]


# =====================================================================
# HARNESS — 4 lớp kiểm soát
# =====================================================================

# --- Layer 2: KIỂM QUYỀN — chạy TRƯỚC khi tool thực thi
def check_permission(tool_name: str, args: dict,
                     constraints: Constraints, flights: list) -> str | None:
    """Trả về None nếu cho phép, hoặc lý do nếu phải chặn."""
    if tool_name == "book_seat":
        flight = next((f for f in flights if f["flight"] == args.get("flight")), None)
        if flight is None or not constraints.is_ok(flight):
            return f"{args.get('flight')} vi phạm ràng buộc: {constraints.to_prompt()}"
    return None


# --- Layer 3: TIÊU CHÍ HOÀN THÀNH — kiểm bằng CODE, không tin model
def is_done(state: AgentState, constraints: Constraints) -> bool:
    """Done = có booking đã paid VÀ thỏa tất cả ràng buộc."""
    for code, b in state.bookings.items():
        if b.get("paid") and constraints.is_ok(b):
            return True
    return False


# --- Layer 4: BÀN GIAO — khi agent thất bại, đưa 3 thứ cho con người
def handoff(state: AgentState) -> dict:
    """Bàn giao: đã làm gì, đã thử gì, câu hỏi cụ thể."""
    return {
        "done_so_far": (
            [f"{c}: paid={b['paid']}" for c, b in state.bookings.items()]
            or ["Chưa đặt vé, chưa thanh toán"]
        ),
        "tried": [
            f"{tool}({args}) -> {result.get('status', '?')}"
            for tool, args, result in state.log
        ],
        "question": (
            "Không có chuyến bay nào thỏa tất cả ràng buộc. "
            "Nên nới lỏng điều kiện nào: giờ khởi hành hay giá tối đa?"
        ),
    }


# --- PHÁT HIỆN LẶP (3 tín hiệu)
class LoopDetector:
    """Phát hiện agent lặp vô ích dựa trên 3 tín hiệu:
        1. Trùng action: cùng (tool, args) lặp lại
        2. Trùng observation: kết quả giống hệt
        3. Không tiến triển: đại lượng bài toán đứng yên
    """
    def __init__(self, window: int = 6, repeat_k: int = 2, stall_n: int = 5):
        self.recent: deque = deque(maxlen=window)
        self.k = repeat_k
        self.n = stall_n
        self.last_progress = None
        self.stall_count = 0

    def check(self, tool: str, args: dict, progress=None) -> str | None:
        """Trả về 'LOOP', 'STALL', hoặc None."""
        fp = (tool, repr(sorted(args.items())))
        # Tín hiệu 1: trùng action
        if self.recent.count(fp) + 1 >= self.k:
            return "LOOP"
        self.recent.append(fp)
        # Tín hiệu 3: không tiến triển
        if progress is not None:
            self.stall_count = (
                self.stall_count + 1 if progress == self.last_progress else 0
            )
            self.last_progress = progress
            if self.stall_count >= self.n:
                return "STALL"
        return None

    def reset(self):
        self.recent.clear()
        self.last_progress = None
        self.stall_count = 0


# --- KIỂM TRA HALLUCINATION
def ungrounded_facts(answer, log: list) -> list:
    """Tìm flight code và giá tiền trong câu trả lời mà không có nguồn từ tool."""
    if isinstance(answer, list):
        answer = " ".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in answer
        )
    elif not isinstance(answer, str):
        answer = str(answer)
    tool_text = json.dumps([result for _, _, result in log])
    facts = re.findall(r"\b[A-Z]{2}\d{3}\b", answer)                       # VN122
    facts += [
        p.replace(",", "")
        for p in re.findall(r"\d{1,3}(?:,\d{3})+", answer)
    ]  # 1,850,000
    return [f for f in facts if f not in tool_text]




# =====================================================================
# REAL LLM MODEL — sử dụng Google AI Studio (Gemini)
# =====================================================================
def load_real_model(model_name: str = "gemini-2.5-flash"):
    """Tạo model LLM thật từ Google AI Studio.

    Yêu cầu: File .env chứa GOOGLE_API_KEY hoặc biến môi trường GOOGLE_API_KEY.

    Args:
        model_name: Tên model Gemini, mặc định "gemini-2.0-flash".

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Chua cau hinh GOOGLE_API_KEY!\n"
            "1. Truy cap https://aistudio.google.com/apikey de lay API key.\n"
            "2. Mo file .env va thay 'your_api_key_here' bang API key that.\n"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True,
    )

# =====================================================================
# SCRIPTED FAKE MODEL — chạy không cần API key
# =====================================================================
def fake_model_react(case: str):
    """Fake model cho ReAct pattern."""
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def call(name, **args):
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": name}],
        )

    script = {
        "success": [
            call("search_flights", origin="SGN", destination="DAD", date="2026-10-07"),
            call("book_seat", flight="VN122"),
            call("pay", code="VN122-12A"),
            call("get_booking", code="VN122-12A"),
            AIMessage(content="Booked VN122, seat 12A, paid 1,850,000 VND."),
        ],
        "fail": [
            call("search_flights", origin="SGN", destination="DAD", date="2026-10-07"),
            call("book_seat", flight="VJ604"),
            AIMessage(content="Không tìm được chuyến bay thỏa tất cả ràng buộc."),
        ],
    }

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return ScriptedModel(messages=iter(script[case]))


SYSTEM_PROMPT = (
    "You are a flight booking agent. Use the tools: search_flights, "
    "book_seat, pay, get_booking. Only book a flight that meets ALL "
    "constraints. If no flight meets them, stop and say so."
)
