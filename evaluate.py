"""BTVN#3 — So sánh & Đánh giá hiệu quả 3 mẫu thiết kế Agent

Chạy tất cả 3 pattern (ReAct, Plan-then-Execute, Hybrid) trên cả 2 case
(success, fail) và in bảng so sánh.

Run:  python evaluate.py
"""
import json
import sys
import io
from contextlib import redirect_stdout

from react_agent import run_react
from plan_execute_agent import run_plan_execute
from hybrid_agent import run_hybrid


def capture_and_run(fn, *args, **kwargs):
    """Chạy hàm, bắt stdout, trả về (output_text, result_dict)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


def print_separator():
    print("=" * 80)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print_separator()
    print("  BTVN#3 — ĐÁNH GIÁ HIỆU QUẢ 3 MẪU THIẾT KẾ AGENT")
    print("  SE373 · Agentic AI Engineering · UIT")
    print_separator()

    results = []

    # =====================================================================
    # CHẠY TẤT CẢ 6 CASE (3 patterns × 2 cases)
    # =====================================================================
    runs = [
        ("ReAct",              "success", lambda c: run_react(c)),
        ("ReAct",              "fail",    lambda c: run_react(c)),
        ("Plan-then-Execute",  "success", lambda c: run_plan_execute(c, auto_approve=True)),
        ("Plan-then-Execute",  "fail",    lambda c: run_plan_execute(c, auto_approve=True)),
        ("Hybrid",             "success", lambda c: run_hybrid(c, auto_approve=True)),
        ("Hybrid",             "fail",    lambda c: run_hybrid(c, auto_approve=True)),
    ]

    for pattern_name, case, fn in runs:
        print(f"\n{'#'*80}")
        print(f"#  {pattern_name} · {case}")
        print(f"{'#'*80}")
        output_text, result = capture_and_run(fn, case)
        print(output_text)
        results.append({
            "pattern": pattern_name,
            "case": case,
            **result,
        })

    # =====================================================================
    # BẢNG SO SÁNH
    # =====================================================================
    print(f"\n{'='*80}")
    print("  BẢNG SO SÁNH HIỆU QUẢ")
    print(f"{'='*80}")

    # Header
    print(f"{'Mẫu':<25} {'Case':<10} {'Kết quả':<10} {'Model calls':<13} "
          f"{'Tool calls':<12} {'Replans':<10}")
    print("-" * 80)

    for r in results:
        pattern = r["pattern"]
        case = r["case"]
        result = r["result"]
        model_calls = r.get("model_calls", "N/A")
        tool_calls = r.get("tool_calls", len(r.get("booking", [])))
        replans = r.get("replans", "-")
        print(f"{pattern:<25} {case:<10} {result:<10} {str(model_calls):<13} "
              f"{str(tool_calls):<12} {str(replans):<10}")

    # =====================================================================
    # PHÂN TÍCH ĐÁNH GIÁ
    # =====================================================================
    print(f"\n{'='*80}")
    print("  PHÂN TÍCH ĐÁNH GIÁ")
    print(f"{'='*80}")

    analysis = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ĐÁNH GIÁ HIỆU QUẢ 3 MẪU THIẾT KẾ                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  1. ReAct (Reasoning + Acting)                                             ║
║  ─────────────────────────────                                             ║
║  ✓ Ưu điểm:                                                               ║
║    • Linh hoạt nhất: quyết định từng bước dựa trên observation mới nhất   ║
║    • Có thể thích nghi ngay lập tức khi tình huống thay đổi               ║
║    • Dễ debug: mỗi vòng là một cặp (action → observation)                 ║
║  ✗ Nhược điểm:                                                             ║
║    • Tốn model call nhất (mỗi bước = 1 model call)                        ║
║    • Chi phí token tăng O(n²) theo số vòng (nạp lại toàn bộ lịch sử)     ║
║    • Rủi ro: lặp vô hạn, trôi mục tiêu khi lịch sử dài                  ║
║  → Phù hợp khi: không đoán được số bước, cần thích nghi cao              ║
║                                                                            ║
║  2. Plan-then-Execute                                                      ║
║  ────────────────────                                                      ║
║  ✓ Ưu điểm:                                                               ║
║    • Tiết kiệm model call nhất (chỉ 1-2 call cho cả kế hoạch)            ║
║    • Kế hoạch nhìn thấy TRƯỚC khi chạy → người duyệt được                ║
║    • Ước lượng chi phí trước khi thực thi                                  ║
║    • Có thể dùng model nhỏ hơn cho bước thực thi                          ║
║  ✗ Nhược điểm:                                                             ║
║    • Không thích nghi: lỗi ở bước nào → dừng luôn                         ║
║    • Kế hoạch lỗi thời nếu environment thay đổi                           ║
║    • Tốn thời gian lập kế hoạch ban đầu                                   ║
║  → Phù hợp khi: cần duyệt trước, tác vụ ổn định, ít bất ngờ             ║
║                                                                            ║
║  3. Hybrid (ReAct + Plan)                                                  ║
║  ────────────────────────                                                  ║
║  ✓ Ưu điểm:                                                               ║
║    • Cân bằng: duyệt trước được VÀ thích nghi được                        ║
║    • Replan khi bước thất bại, không dừng cứng như Plan-then-Execute       ║
║    • Số model call trung bình (giữa ReAct và Plan-then-Execute)           ║
║  ✗ Nhược điểm:                                                             ║
║    • Phức tạp nhất: khó debug hơn vì có nhiều vòng lập kế hoạch           ║
║    • Cần thiết kế khi nào replan (trigger condition)                       ║
║    • Vẫn có thể lặp vô hạn nếu replan không hội tụ                       ║
║  → Phù hợp khi: tác vụ dài, environment biến động, cần cả hai            ║
║                                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TỔNG KẾT BẢNG CHỌN MẪU                                                  ║
║  ┌──────────────────┬──────────────────────┬────────────────────────┐      ║
║  │ Mẫu              │ Chọn khi             │ Rủi ro chính           │      ║
║  ├──────────────────┼──────────────────────┼────────────────────────┤      ║
║  │ ReAct            │ Không đoán được bước │ Lặp vô hạn, trôi mục  │      ║
║  │ Plan-then-Exec   │ Cần duyệt trước     │ Kế hoạch lỗi thời     │      ║
║  │ Hybrid           │ Tác vụ dài, biến động│ Khó debug hơn         │      ║
║  └──────────────────┴──────────────────────┴────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(analysis)

    # =====================================================================
    # HARNESS LAYERS SUMMARY
    # =====================================================================
    harness_summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TỔNG KẾT 4 LỚP HARNESS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Layer 1: RÀNG BUỘC LÀ DỮ LIỆU (Constraints)                            ║
║  • Yêu cầu người dùng → dataclass Constraints                             ║
║  • Prompt SINH TỪ data, không hardcode                                     ║
║  • Harness KIỂM data bằng code (is_ok)                                    ║
║  • Model không thể "quên" ràng buộc                                       ║
║                                                                            ║
║  Layer 2: KIỂM QUYỀN (check_permission)                                  ║
║  • Chạy TRƯỚC khi tool thực thi                                           ║
║  • So sánh args của tool với Constraints data                              ║
║  • Block tool call nếu vi phạm → trả denied                               ║
║                                                                            ║
║  Layer 3: TIÊU CHÍ HOÀN THÀNH (is_done)                                  ║
║  • Kiểm bằng CODE, không tin model tự tuyên bố                            ║
║  • Đọc lại booking từ hệ thống (get_booking)                              ║
║  • Kiểm: paid=True AND thỏa tất cả ràng buộc                             ║
║                                                                            ║
║  Layer 4: BÀN GIAO (handoff)                                              ║
║  • Khi agent thất bại → đưa 3 thứ cho con người:                          ║
║    1. done_so_far: đã làm tới đâu                                         ║
║    2. tried: đã thử gì                                                     ║
║    3. question: câu hỏi cụ thể để người trả lời trong 30 giây            ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(harness_summary)


if __name__ == "__main__":
    main()
