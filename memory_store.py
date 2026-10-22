"""Challenge A — Hệ thống Quản lý Bộ nhớ Toàn diện (Memory Architecture)
Tích hợp trực tiếp với Flight Booking Agent (shared.py)

Phân tầng bộ nhớ:
1. Rules & Policies: Tải từ RULES.md (Quy tắc hệ thống bất biến).
2. Persistent Memory: Lưu trong persistent_memory.json (Quy ước đặt vé: hãng bay, ngân sách, giờ bay).
3. Session Memory: Lưu trong session_memory.json (Tiến độ công việc phiên hiện hành).
4. Memory Decision Engine: Tự động phân loại câu chat tạm thời và quy ước nghiệp vụ đặt vé.
5. Integration: Ánh xạ Persistent Memory vào đối tượng Constraints của Agent đặt vé.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from shared import Constraints, FLIGHTS, AgentState, make_tools, check_permission, is_done, handoff


@dataclass
class MemoryEntry:
    key: str
    value: str
    source: str
    status: str = "active"          # "active" hoặc "superseded"
    created_at: str = ""
    updated_at: str = ""
    history: list = field(default_factory=list)

    def __post_init__(self):
        now = datetime.now().isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class SessionMemory:
    session_id: str
    goal: str = ""
    current_step: str = "init"
    findings: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    next_action: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(
        self,
        rules_path: str = "RULES.md",
        persistent_path: str = "persistent_memory.json",
        session_path: str = "session_memory.json"
    ):
        self.rules_path = rules_path
        self.persistent_path = persistent_path
        self.session_path = session_path

        self.rules: str = ""
        self.persistent_entries: dict[str, MemoryEntry] = {}
        self.current_session: SessionMemory | None = None

        self._load_all()

    def _load_all(self):
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = f.read()

        if os.path.exists(self.persistent_path):
            with open(self.persistent_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.persistent_entries[k] = MemoryEntry(**v)

        if os.path.exists(self.session_path):
            with open(self.session_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                self.current_session = SessionMemory(**s_data)

    def _save_persistent(self):
        data = {k: asdict(v) for k, v in self.persistent_entries.items()}
        with open(self.persistent_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_session(self):
        if self.current_session:
            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.current_session), f, indent=2, ensure_ascii=False)

    def start_session(self, session_id: str, goal: str):
        self.current_session = SessionMemory(session_id=session_id, goal=goal)
        self._save_session()

    def update_session_progress(self, current_step: str, finding: str = "", next_action: str = ""):
        if not self.current_session:
            self.start_session("default", "unknown")
        self.current_session.current_step = current_step
        if finding:
            self.current_session.findings.append(finding)
        if next_action:
            self.current_session.next_action = next_action
        self.current_session.updated_at = datetime.now().isoformat(timespec="seconds")
        self._save_session()

    def save_memory(self, key: str, value: str, source: str) -> dict:
        if key in self.persistent_entries and self.persistent_entries[key].status == "active":
            return self.update_memory(key, value, source)

        entry = MemoryEntry(key=key, value=value, source=source)
        self.persistent_entries[key] = entry
        self._save_persistent()
        return {
            "status": "saved",
            "key": key,
            "value": value,
            "source": source
        }

    def update_memory(self, key: str, new_value: str, source: str) -> dict:
        if key not in self.persistent_entries:
            return self.save_memory(key, new_value, source)

        old = self.persistent_entries[key]
        now = datetime.now().isoformat(timespec="seconds")
        old.history.append({
            "old_value": old.value,
            "old_source": old.source,
            "superseded_at": now,
            "superseded_by": source,
        })
        old.value = new_value
        old.source = source
        old.updated_at = now
        old.status = "active"
        self._save_persistent()
        return {
            "status": "updated",
            "key": key,
            "old_value": old.history[-1]["old_value"],
            "new_value": new_value,
            "source": source
        }

    def read_memory(self, key: str | None = None) -> dict:
        if key:
            if key in self.persistent_entries:
                return asdict(self.persistent_entries[key])
            return {}
        
        active = {
            k: {"value": e.value, "source": e.source, "updated_at": e.updated_at}
            for k, e in self.persistent_entries.items()
            if e.status == "active"
        }
        return {"status": "ok", "count": len(active), "memories": active}

    # =========================================================================
    # MEMORY DECISION ENGINE — BỘ LỌC NGHIỆP VỤ ĐẶT VÉ MÁY BAY
    # =========================================================================
    def inspect_and_extract_memory(self, user_message: str, session_id: str) -> dict:
        lower_msg = user_message.lower()

        convention_triggers = [
            "thống nhất", "quy ước", "quy định", "từ nay", "luôn", "ưu tiên",
            "đổi sang", "chuyển sang", "thay thế", "ngân sách", "trần giá",
            "thong nhat", "quy uoc", "quy dinh", "tu nay", "luon", "uu tien",
            "doi sang", "chuyen sang", "ngan sach", "tran gia"
        ]

        is_convention = any(trigger in lower_msg for trigger in convention_triggers)

        if not is_convention:
            return {
                "action": "IGNORE",
                "reason": "Tin nhắn là hội thoại/yêu cầu công việc thông thường, không phải quy ước dự án."
            }

        extracted_list = []

        # 1. Hãng bay ưu tiên
        if "vietjet" in lower_msg or "vj" in lower_msg:
            extracted_list.append(("preferred_airline", "Vietjet Air (VJ)"))
        elif "vietnam airlines" in lower_msg or "vna" in lower_msg or "vn" in lower_msg:
            extracted_list.append(("preferred_airline", "Vietnam Airlines (VN)"))
        elif "bamboo" in lower_msg or "qh" in lower_msg:
            extracted_list.append(("preferred_airline", "Bamboo Airways (QH)"))

        # 2. Ngân sách tối đa
        if "2.500.000" in lower_msg or "2.5 triệu" in lower_msg or "2,5 triệu" in lower_msg or "2500000" in lower_msg:
            extracted_list.append(("max_price", "2500000"))
        elif "2.000.000" in lower_msg or "2 triệu" in lower_msg or "2000000" in lower_msg:
            extracted_list.append(("max_price", "2000000"))

        # 3. Giờ bay
        if "sáng" in lower_msg or "12:00" in lower_msg or "12h" in lower_msg or "sang" in lower_msg:
            extracted_list.append(("depart_before", "12:00"))

        if extracted_list:
            last_res = {}
            for k, v in extracted_list:
                if k in self.persistent_entries:
                    last_res = self.update_memory(k, v, source=f"{session_id} (tự động trích xuất)")
                    last_res["action"] = "AUTO_UPDATED"
                else:
                    last_res = self.save_memory(k, v, source=f"{session_id} (tự động trích xuất)")
                    last_res["action"] = "AUTO_SAVED"
            return last_res

        return {
            "action": "IGNORE",
            "reason": "Phát hiện từ khóa nhưng không nhận diện được quy ước cụ thể để lưu."
        }

    # =========================================================================
    # ÁNH XẠ PERSISTENT MEMORY VÀO RÀNG BUỘC CỦA AGENT ĐẶT VÉ (CONSTRAINTS)
    # =========================================================================
    def apply_to_constraints(self, base_constraints: Constraints) -> Constraints:
        """Nạp các quy ước đang active từ Persistent Memory vào đối tượng Constraints."""
        c = Constraints(
            origin=base_constraints.origin,
            destination=base_constraints.destination,
            date=base_constraints.date,
            depart_before=base_constraints.depart_before,
            max_price=base_constraints.max_price
        )

        # Đọc từ persistent memory
        max_price_mem = self.read_memory("max_price")
        if max_price_mem and max_price_mem.get("status") == "active":
            c.max_price = int(max_price_mem.get("value", c.max_price))

        depart_mem = self.read_memory("depart_before")
        if depart_mem and depart_mem.get("status") == "active":
            c.depart_before = depart_mem.get("value", c.depart_before)

        return c

    def inject_into_context(self) -> str:
        context_parts = []

        if self.rules:
            context_parts.append(f"=== SYSTEM RULES (Bất biến) ===\n{self.rules.strip()}")

        active_memories = [
            f"  • {k}: {e.value} (nguồn: {e.source})"
            for k, e in self.persistent_entries.items()
            if e.status == "active"
        ]
        if active_memories:
            context_parts.append("=== PERSISTENT MEMORY (Quy ước đặt vé hiện hành) ===\n" + "\n".join(active_memories))
        else:
            context_parts.append("=== PERSISTENT MEMORY ===\n(Chưa có quy ước nào)")

        if self.current_session and self.current_session.goal:
            s = self.current_session
            session_info = (
                f"=== CURRENT SESSION MEMORY ({s.session_id}) ===\n"
                f"Goal: {s.goal}\n"
                f"Step: {s.current_step}\n"
                f"Findings: {s.findings}\n"
                f"Next action: {s.next_action}"
            )
            context_parts.append(session_info)

        return "\n\n".join(context_parts)
