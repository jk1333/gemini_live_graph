"""Pure-Python State Graph engine orchestrating Gemini 3.8 Live context updates."""

from __future__ import annotations

from typing import Any

from airline_chatbot.mock_store import MockAirlineStore, SlotExtractor
from airline_chatbot.models import (
    DecisionResult,
    NodeId,
    RootDecision,
    SessionSlots,
    StepDecision,
    ToolResultStatus,
)
from airline_chatbot.prompts_and_tools import (
    NODE_TOOL_NAMES,
    build_system_instruction,
    build_tools,
)


class AirlineStateGraph:
    """State Graph managing active node, extracted slots, and mid-session context updates.

    Key invariants:
    1. Every node exposes exactly ONE tool declaration whose sole parameter is a strict Enum.
    2. `contextUpdate` payloads are staged ONLY when `current_node` actually changes, preserving
       Gemini 3.8 Live prompt prefix caching when retrying within the same node.
    3. Staged `contextUpdate` frames are flushed right after `turnComplete` so in-flight tool
       responses always match the active tool declaration of the turn.
    """

    def __init__(self, store: MockAirlineStore | None = None) -> None:
        self.store = store or MockAirlineStore()
        self.current_node: NodeId = NodeId.ROOT_MENU
        self.slots = SessionSlots()
        self._pending_context_node: NodeId | None = None

    def get_context_config(self, node_id: NodeId | None = None) -> dict[str, Any]:
        """Returns `systemInstruction` and `tools` dictionary for `setup` or `contextUpdate`."""
        target = node_id or self.current_node
        return {
            "systemInstruction": build_system_instruction(target),
            "tools": build_tools(target),
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        """Returns a complete JSON-serializable snapshot of the graph & mock DB state."""
        active_ctx = self.get_context_config(self.current_node)
        active_decl = active_ctx["tools"][0]["functionDeclarations"][0]
        return {
            "current_node": self.current_node.value,
            "pending_context_node": (
                self._pending_context_node.value if self._pending_context_node else None
            ),
            "slots": self.slots.to_dict(),
            "active_tool": active_decl,
            "system_instruction": active_ctx["systemInstruction"]["parts"][0]["text"],
            "flights": self.store.list_all_flights(),
            "bookings": self.store.list_all_bookings(),
        }

    @staticmethod
    def get_topology_metadata() -> dict[str, Any]:
        """Returns static node & edge definitions for the frontend State Graph visualizer."""
        node_labels: dict[NodeId, tuple[str, str]] = {
            NodeId.ROOT_MENU: ("초기 메뉴 분기", "고객 의도 파악 및 3대 기능 분기"),
            NodeId.TICKET_LOOKUP: ("1. 예약 티켓 조회", "user_id + booking_number 검증 및 조회"),
            NodeId.SCHEDULE_LOOKUP: ("2. 비행기 일정 조회", "출도착지/날짜 필터 또는 전체 일정 조회"),
            NodeId.BOOKING_AUTH: ("3-A. 예약 인증", "user_id + 사전 예약번호 검증"),
            NodeId.BOOKING_CONFIRM: ("3-B. 항공편 확정", "항공편 편명 선택 및 좌석 확정"),
        }
        nodes: list[dict[str, Any]] = []
        for node in NodeId:
            title, subtitle = node_labels[node]
            decl = build_tools(node)[0]["functionDeclarations"][0]
            si_text = build_system_instruction(node)["parts"][0]["text"]
            nodes.append(
                {
                    "id": node.value,
                    "title": title,
                    "subtitle": subtitle,
                    "tool": decl,
                    "system_instruction": si_text,
                }
            )

        edges: list[dict[str, str]] = [
            {
                "id": "ROOT_MENU->TICKET_LOOKUP",
                "source": NodeId.ROOT_MENU.value,
                "target": NodeId.TICKET_LOOKUP.value,
                "label": "CHECK_TICKET",
                "kind": "forward",
            },
            {
                "id": "ROOT_MENU->SCHEDULE_LOOKUP",
                "source": NodeId.ROOT_MENU.value,
                "target": NodeId.SCHEDULE_LOOKUP.value,
                "label": "CHECK_SCHEDULE",
                "kind": "forward",
            },
            {
                "id": "ROOT_MENU->BOOKING_AUTH",
                "source": NodeId.ROOT_MENU.value,
                "target": NodeId.BOOKING_AUTH.value,
                "label": "BOOK_TICKET",
                "kind": "forward",
            },
            {
                "id": "BOOKING_AUTH->BOOKING_CONFIRM",
                "source": NodeId.BOOKING_AUTH.value,
                "target": NodeId.BOOKING_CONFIRM.value,
                "label": "EXECUTE (인증 통과)",
                "kind": "forward",
            },
            {
                "id": "TICKET_LOOKUP->ROOT_MENU",
                "source": NodeId.TICKET_LOOKUP.value,
                "target": NodeId.ROOT_MENU.value,
                "label": "완료 / RETURN_TO_ROOT",
                "kind": "return",
            },
            {
                "id": "SCHEDULE_LOOKUP->ROOT_MENU",
                "source": NodeId.SCHEDULE_LOOKUP.value,
                "target": NodeId.ROOT_MENU.value,
                "label": "완료 / RETURN_TO_ROOT",
                "kind": "return",
            },
            {
                "id": "BOOKING_AUTH->ROOT_MENU",
                "source": NodeId.BOOKING_AUTH.value,
                "target": NodeId.ROOT_MENU.value,
                "label": "RETURN_TO_ROOT",
                "kind": "return",
            },
            {
                "id": "BOOKING_CONFIRM->ROOT_MENU",
                "source": NodeId.BOOKING_CONFIRM.value,
                "target": NodeId.ROOT_MENU.value,
                "label": "확정 완료 / RETURN_TO_ROOT",
                "kind": "return",
            },
        ]
        return {"nodes": nodes, "edges": edges}

    def ingest_user_text(self, text: str) -> None:
        """Extracts slot candidates from user input prior to streaming `realtimeInput`."""
        SlotExtractor.extract_into(text, self.slots, self.current_node)

    def should_auto_execute_current_node(self) -> bool:
        """Returns True if the active branch node already has enough extracted slots to execute."""
        if self.current_node in {NodeId.TICKET_LOOKUP, NodeId.BOOKING_AUTH}:
            return bool(self.slots.user_id and self.slots.booking_number)
        if self.current_node == NodeId.SCHEDULE_LOOKUP:
            return bool(
                self.slots.departure_code
                or self.slots.arrival_code
                or self.slots.date
                or self.slots.query_all_schedules
            )
        return False

    @staticmethod
    def build_silent_turn_fallback(result: DecisionResult) -> str:
        """Customer-facing Korean fallback text if the audio model completes silently after a tool call."""
        if result.status == ToolResultStatus.DECISION_ROUTED:
            if result.active_node == NodeId.TICKET_LOOKUP:
                return "네, 예약된 비행기 티켓 조회를 도와드리겠습니다. 사용자 ID와 예약 번호(예: BK-1001)를 입력해 주세요."
            if result.active_node == NodeId.SCHEDULE_LOOKUP:
                return "네, 비행기 운항 일정 조회를 도와드리겠습니다. 원하시는 출발지/도착지 노선이나 전체 일정 조회 여부를 말씀해 주세요."
            if result.active_node == NodeId.BOOKING_AUTH:
                return "네, 비행기 티켓 예약을 도와드리겠습니다. 먼저 사용자 ID와 사전 발급된 예약 번호(예: BK-2001)를 입력해 주세요."
        if result.status == ToolResultStatus.RETURNED_TO_ROOT:
            return "네, 초기 메인 메뉴로 돌아왔습니다. 예약 티켓 조회, 비행기 일정 조회, 티켓 예약 중 어떤 업무를 도와드릴까요?"
        return result.message

    def force_reset_to_root(self) -> dict[str, Any]:
        """Forces an immediate transition back to `ROOT_MENU` (used by CLI `/reset`)."""
        self.slots.reset()
        self.current_node = NodeId.ROOT_MENU
        self._pending_context_node = None
        return self.get_context_config(NodeId.ROOT_MENU)

    def consume_pending_context_update(self) -> tuple[NodeId, dict[str, Any]] | None:
        """Pops a staged `contextUpdate` payload if the node transitioned during the last turn."""
        if self._pending_context_node is None:
            return None
        target_node = self._pending_context_node
        self._pending_context_node = None
        return target_node, self.get_context_config(target_node)

    def _transition_to(self, next_node: NodeId, *, clear_slots: bool = False) -> None:
        prev_node = self.current_node
        if clear_slots:
            self.slots.reset()
        self.current_node = next_node
        if next_node != prev_node:
            self._pending_context_node = next_node

    def handle_tool_call(self, func_name: str, args: dict[str, Any] | None) -> DecisionResult:
        """Evaluates the single Enum tool call for the active Graph Node."""
        args = args or {}
        raw_decision = str(args.get("decision", "")).strip().upper()
        expected_tool = NODE_TOOL_NAMES[self.current_node]

        if func_name != expected_tool:
            return DecisionResult(
                status=ToolResultStatus.INVALID_DECISION,
                decision=raw_decision or "UNKNOWN",
                active_node=self.current_node,
                retryable=True,
                message=(
                    f"현재 단계({self.current_node.value})에서는 '{expected_tool}' 도구만 사용할 수 있습니다."
                ),
            )

        if self.current_node == NodeId.ROOT_MENU:
            return self._handle_root_decision(raw_decision)

        return self._handle_step_decision(raw_decision)

    def _handle_root_decision(self, raw_decision: str) -> DecisionResult:
        try:
            decision = RootDecision(raw_decision)
        except ValueError:
            return DecisionResult(
                status=ToolResultStatus.INVALID_DECISION,
                decision=raw_decision or "EMPTY",
                active_node=self.current_node,
                retryable=True,
                message="유효하지 않은 메뉴 선택값입니다. CHECK_TICKET, CHECK_SCHEDULE, BOOK_TICKET 중 하나여야 합니다.",
            )

        if decision == RootDecision.CHECK_TICKET:
            self._transition_to(NodeId.TICKET_LOOKUP)
            if self.should_auto_execute_current_node():
                msg = "예약된 비행기 티켓 조회 단계로 이동했습니다. 입력해 주신 정보로 바로 확인해 드리겠다고 짧게 한 문장으로 안내하세요."
            else:
                detected_hint = self._format_auth_slot_hint()
                msg = (
                    "예약된 비행기 티켓 조회 단계로 이동했습니다. "
                    f"고객에게 사용자 ID와 예약 번호(예: BK-1001)를 요청하세요.{detected_hint}"
                )
            return DecisionResult(
                status=ToolResultStatus.DECISION_ROUTED,
                decision=decision.value,
                active_node=self.current_node,
                retryable=False,
                message=msg,
                data={"collected_slots": self.slots.to_dict()},
            )

        if decision == RootDecision.CHECK_SCHEDULE:
            self._transition_to(NodeId.SCHEDULE_LOOKUP)
            if self.should_auto_execute_current_node():
                msg = "비행기 일정 조회 단계로 이동했습니다. 요청하신 노선 일정을 바로 확인해 드리겠다고 짧게 한 문장으로 안내하세요."
            else:
                msg = "비행기 일정 조회 단계로 이동했습니다. 조회를 원하는 출발지/도착지 노선이나 전체 일정 조회 여부를 한국어로 물어보세요."
            return DecisionResult(
                status=ToolResultStatus.DECISION_ROUTED,
                decision=decision.value,
                active_node=self.current_node,
                retryable=False,
                message=msg,
                data={"collected_slots": self.slots.to_dict()},
            )

        # RootDecision.BOOK_TICKET
        self._transition_to(NodeId.BOOKING_AUTH)
        if self.should_auto_execute_current_node():
            msg = "비행기 티켓 예약 인증 단계로 이동했습니다. 입력해 주신 계정과 예약 번호를 바로 확인하겠다고 짧게 안내하세요."
        else:
            detected_hint = self._format_auth_slot_hint()
            msg = (
                "비행기 티켓 예약 인증 단계로 이동했습니다. "
                f"예약을 위해 사용자 ID와 사전 발급된 예약 번호(예: BK-2001)를 입력해 달라고 안내하세요.{detected_hint}"
            )
        return DecisionResult(
            status=ToolResultStatus.DECISION_ROUTED,
            decision=decision.value,
            active_node=self.current_node,
            retryable=False,
            message=msg,
            data={"collected_slots": self.slots.to_dict()},
        )

    def _handle_step_decision(self, raw_decision: str) -> DecisionResult:
        try:
            decision = StepDecision(raw_decision)
        except ValueError:
            return DecisionResult(
                status=ToolResultStatus.INVALID_DECISION,
                decision=raw_decision or "EMPTY",
                active_node=self.current_node,
                retryable=True,
                message="유효하지 않은 단계 결정값입니다. EXECUTE 또는 RETURN_TO_ROOT 중 하나여야 합니다.",
            )

        if decision == StepDecision.RETURN_TO_ROOT:
            self._transition_to(NodeId.ROOT_MENU, clear_slots=True)
            return DecisionResult(
                status=ToolResultStatus.RETURNED_TO_ROOT,
                decision=decision.value,
                active_node=self.current_node,
                retryable=False,
                message=(
                    "진행 중이던 작업을 취소하고 초기 메인 메뉴로 돌아왔습니다. "
                    "1) 예약 티켓 조회, 2) 비행기 일정 조회, 3) 비행기 티켓 예약 중 어떤 것을 도와드릴지 안내하세요."
                ),
            )

        # StepDecision.EXECUTE per active node
        if self.current_node == NodeId.TICKET_LOOKUP:
            return self._execute_ticket_lookup()
        if self.current_node == NodeId.SCHEDULE_LOOKUP:
            return self._execute_schedule_lookup()
        if self.current_node == NodeId.BOOKING_AUTH:
            return self._execute_booking_auth()
        if self.current_node == NodeId.BOOKING_CONFIRM:
            return self._execute_booking_confirm()

        return DecisionResult(
            status=ToolResultStatus.INVALID_DECISION,
            decision=decision.value,
            active_node=self.current_node,
            retryable=False,
            message="알 수 없는 그래프 노드 상태입니다.",
        )

    def _execute_ticket_lookup(self) -> DecisionResult:
        missing = self._missing_auth_slots()
        if missing:
            return DecisionResult(
                status=ToolResultStatus.MISSING_SLOTS,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=f"아직 다음 필수 정보가 입력되지 않았습니다: {', '.join(missing)}. 고객에게 해당 정보를 요청하세요.",
                data={"missing_fields": missing, "collected_slots": self.slots.to_dict()},
            )

        assert self.slots.user_id is not None
        assert self.slots.booking_number is not None
        status, msg, payload = self.store.lookup_booked_ticket(
            user_id=self.slots.user_id,
            booking_number=self.slots.booking_number,
        )
        if status != ToolResultStatus.EXECUTED_OK:
            return DecisionResult(
                status=status,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=msg,
                data=payload,
            )

        self._transition_to(NodeId.ROOT_MENU, clear_slots=True)
        return DecisionResult(
            status=ToolResultStatus.EXECUTED_OK,
            decision=StepDecision.EXECUTE.value,
            active_node=self.current_node,
            retryable=False,
            message=(
                f"{msg} "
                "(시스템 안내: 이번 답변 후 상태가 ROOT_MENU로 초기화됩니다. 다음 턴에 고객이 다시 조회/일정/예약을 요청하면 반드시 route_menu 도구를 새로 호출하세요.)"
            ),
            data=payload,
        )

    def _execute_schedule_lookup(self) -> DecisionResult:
        status, msg, payload = self.store.search_schedules(
            departure_code=self.slots.departure_code,
            arrival_code=self.slots.arrival_code,
            date=self.slots.date,
        )
        self._transition_to(NodeId.ROOT_MENU, clear_slots=True)
        return DecisionResult(
            status=status,
            decision=StepDecision.EXECUTE.value,
            active_node=self.current_node,
            retryable=False,
            message=(
                f"{msg} "
                "(시스템 안내: 이번 답변 후 상태가 ROOT_MENU로 초기화됩니다. 다음 턴에 고객이 다시 비행기 일정 조회나 티켓 조회/예약을 요청하면 직접 되묻지 말고 반드시 route_menu 도구를 새로 호출하세요.)"
            ),
            data=payload,
        )

    def _execute_booking_auth(self) -> DecisionResult:
        missing = self._missing_auth_slots()
        if missing:
            return DecisionResult(
                status=ToolResultStatus.MISSING_SLOTS,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=f"티켓 예약을 위해 다음 필수 정보를 추가로 입력받아야 합니다: {', '.join(missing)}.",
                data={"missing_fields": missing, "collected_slots": self.slots.to_dict()},
            )

        assert self.slots.user_id is not None
        assert self.slots.booking_number is not None
        status, msg, payload = self.store.verify_booking_auth(
            user_id=self.slots.user_id,
            booking_number=self.slots.booking_number,
        )
        if status != ToolResultStatus.EXECUTED_OK:
            return DecisionResult(
                status=status,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=msg,
                data=payload,
            )

        self._transition_to(NodeId.BOOKING_CONFIRM)
        return DecisionResult(
            status=ToolResultStatus.EXECUTED_OK,
            decision=StepDecision.EXECUTE.value,
            active_node=self.current_node,
            retryable=False,
            message=msg,
            data=payload,
        )

    def _execute_booking_confirm(self) -> DecisionResult:
        if not self.slots.flight_id:
            return DecisionResult(
                status=ToolResultStatus.MISSING_SLOTS,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=(
                    "아직 예약할 항공편 편명(예: KE101, OZ202, LJ303, TW404)이 선택되지 않았습니다. "
                    "원하시는 항공편 편명을 다시 여쭤보세요."
                ),
                data={"available_flights": self.store.list_all_flights()},
            )

        assert self.slots.user_id is not None
        assert self.slots.booking_number is not None
        status, msg, payload = self.store.confirm_booking(
            user_id=self.slots.user_id,
            booking_number=self.slots.booking_number,
            flight_id=self.slots.flight_id,
        )
        if status != ToolResultStatus.EXECUTED_OK:
            self.slots.flight_id = None
            return DecisionResult(
                status=status,
                decision=StepDecision.EXECUTE.value,
                active_node=self.current_node,
                retryable=True,
                message=msg,
                data=payload,
            )

        self._transition_to(NodeId.ROOT_MENU, clear_slots=True)
        return DecisionResult(
            status=ToolResultStatus.EXECUTED_OK,
            decision=StepDecision.EXECUTE.value,
            active_node=self.current_node,
            retryable=False,
            message=(
                f"{msg} "
                "(시스템 안내: 이번 답변 후 상태가 ROOT_MENU로 초기화됩니다. 다음 턴에 고객이 다시 조회/일정/예약을 요청하면 반드시 route_menu 도구를 새로 호출하세요.)"
            ),
            data=payload,
        )

    def _missing_auth_slots(self) -> list[str]:
        missing: list[str] = []
        if not self.slots.user_id:
            missing.append("사용자 ID(user_id)")
        if not self.slots.booking_number:
            missing.append("예약 번호(booking_number)")
        return missing

    def _format_auth_slot_hint(self) -> str:
        if self.slots.user_id and self.slots.booking_number:
            return (
                f" (참고: 이미 사용자 ID={self.slots.user_id}, 예약번호={self.slots.booking_number}가 감지되었습니다. "
                "맞는지 고객에게 가볍게 확인을 받으세요.)"
            )
        if self.slots.user_id:
            return f" (사용자 ID={self.slots.user_id}는 확인되었으니 예약 번호만 요청하세요.)"
        if self.slots.booking_number:
            return f" (예약 번호={self.slots.booking_number}는 확인되었으니 사용자 ID만 요청하세요.)"
        return ""
