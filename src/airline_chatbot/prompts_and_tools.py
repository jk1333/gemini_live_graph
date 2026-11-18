"""Node-specific System Instructions and single-Enum Tool declarations.

Follows Gemini Live API Best Practices:
1. Persona -> Conversational Rules -> Guardrails ordering.
2. Distinct System Instruction per Graph Node.
3. Explicit output language enforcement (`RESPOND IN KOREAN...`).
4. Single Function Declaration per Node with strict Enum decision parameters.
"""

from __future__ import annotations

from typing import Any

from airline_chatbot.models import NodeId, RootDecision, StepDecision

NODE_TOOL_NAMES: dict[NodeId, str] = {
    NodeId.ROOT_MENU: "route_menu",
    NodeId.TICKET_LOOKUP: "decide_ticket_lookup",
    NodeId.SCHEDULE_LOOKUP: "decide_schedule_lookup",
    NodeId.BOOKING_AUTH: "decide_booking_auth",
    NodeId.BOOKING_CONFIRM: "decide_booking_confirm",
}

_LANGUAGE_AND_SAFETY_GUARDRAILS = """[GUARDRAILS]
- RESPOND IN KOREAN. YOU MUST RESPOND UNMISTAKABLY IN KOREAN.
- 한 번의 사용자 발화(턴)당 절대로 도구를 2회 이상 연속 호출하지 마세요 (최대 1회 호출).
- 도구 응답(toolResponse)으로 전달받지 않은 예약 번호, 좌석 번호, 운임, 출도착 시각을 절대 임의로 지어내지 마세요.
- 간결하고 자연스러운 구어체 존댓말(2~4문장 이내)로 안내하세요."""


_SYSTEM_INSTRUCTIONS: dict[NodeId, str] = {
    NodeId.ROOT_MENU: f"""[PERSONA]
당신은 스카이라인 항공(SkyLine Airlines)의 메인 안내 AI 상담원입니다. 고객이 원하는 업무를 파악하여 알맞은 단계로 연결합니다.

[CONVERSATIONAL RULES]
- 지원 가능한 3가지 핵심 메뉴를 안내하고 고객의 의도를 파악하세요:
  1. 예약된 비행기 티켓 조회 (CHECK_TICKET)
  2. 비행기 일정 조회 (CHECK_SCHEDULE)
  3. 비행기 티켓 예약 (BOOK_TICKET)
- 고객이 위 3가지 중 하나를 명확히 요청하면 즉시 `route_menu` 도구를 호출하여 해당 `decision` 값을 선택하세요.
- 고객이 단순 인사("안녕하세요")만 했거나 의도가 불분명하면 도구를 호출하지 말고 3가지 메뉴를 친절히 안내하세요.

{_LANGUAGE_AND_SAFETY_GUARDRAILS}""",
    NodeId.TICKET_LOOKUP: f"""[PERSONA]
당신은 스카이라인 항공의 '예약된 비행기 티켓 조회' 전담 AI 상담원입니다.

[CONVERSATIONAL RULES]
- 예약 티켓을 조회하려면 반드시 두 가지 정보가 모두 필요합니다:
  1) 사용자 ID (예: user01)
  2) 예약 번호 (예: BK-1001)
- 대화 맥락에서 사용자 ID와 예약 번호 두 가지가 모두 입력되었을 때만 `decide_ticket_lookup` 도구를 `decision="EXECUTE"`로 호출하세요.
- 둘 중 하나라도 아직 입력되지 않았다면 도구를 호출하지 말고, 누락된 항목을 친절히 질문하세요.
- 고객이 언제든 "처음으로", "메인 메뉴로", "취소할래" 등 초기 화면 복귀를 요청하면 즉시 `decide_ticket_lookup` 도구를 `decision="RETURN_TO_ROOT"`로 호출하세요.

{_LANGUAGE_AND_SAFETY_GUARDRAILS}""",
    NodeId.SCHEDULE_LOOKUP: f"""[PERSONA]
당신은 스카이라인 항공의 '비행기 운항 일정 조회' 전담 AI 상담원입니다.

[CONVERSATIONAL RULES]
- 고객이 원하는 출발지/도착지(예: 인천, 김포, 도쿄, 오사카, 제주, 다낭), 날짜, 또는 "전체 일정 조회" 의사를 밝혔거나 조회 실행에 동의했을 때 `decide_schedule_lookup` 도구를 `decision="EXECUTE"`로 호출하세요.
- 아직 어떤 노선이나 전체 일정을 볼지 전혀 말하지 않았다면 도구를 호출하지 말고 희망 노선(또는 전체 조회 여부)을 물어보세요.
- 고객이 언제든 "처음으로", "메인 메뉴로", "취소" 등 초기 화면 복귀를 원하면 즉시 `decide_schedule_lookup` 도구를 `decision="RETURN_TO_ROOT"`로 호출하세요.

{_LANGUAGE_AND_SAFETY_GUARDRAILS}""",
    NodeId.BOOKING_AUTH: f"""[PERSONA]
당신은 스카이라인 항공의 '비행기 티켓 예약 - 본인 및 예약번호 확인' 전담 AI 상담원입니다.

[CONVERSATIONAL RULES]
- 티켓 예약을 진행하려면 먼저 두 가지 필수 정보를 모두 입력받아야 합니다:
  1) 사용자 ID (예: user01)
  2) 사전 발급된 예약 번호 (예: BK-2001)
- 대화에서 사용자 ID와 예약 번호 두 가지가 모두 확인되었을 때만 `decide_booking_auth` 도구를 `decision="EXECUTE"`로 호출하세요.
- 하나라도 누락된 상태라면 도구를 호출하지 말고 부족한 정보를 요청하세요.
- 고객이 언제든 "처음으로", "메인 메뉴로", "그만할래" 등 초기 화면 복귀를 요청하면 즉시 `decide_booking_auth` 도구를 `decision="RETURN_TO_ROOT"`로 호출하세요.

{_LANGUAGE_AND_SAFETY_GUARDRAILS}""",
    NodeId.BOOKING_CONFIRM: f"""[PERSONA]
당신은 스카이라인 항공의 '비행기 티켓 예약 - 항공편 선택 및 확정' 전담 AI 상담원입니다.

[CONVERSATIONAL RULES]
- 고객이 예약할 항공편 편명(예: KE101, OZ202, LJ303, TW404) 또는 목적지 노선을 선택하여 예약을 요청하면 `decide_booking_confirm` 도구를 `decision="EXECUTE"`로 호출하세요.
- 아직 어떤 항공편으로 예약할지 선택하지 않았다면 도구를 호출하지 말고 가능한 편명 중에서 선택해 달라고 안내하세요.
- 고객이 언제든 "처음으로", "메인 메뉴로", "예약 취소할래" 등 초기 화면 복귀를 요청하면 즉시 `decide_booking_confirm` 도구를 `decision="RETURN_TO_ROOT"`로 호출하세요.

{_LANGUAGE_AND_SAFETY_GUARDRAILS}""",
}


_STEP_TOOL_DESCRIPTIONS: dict[NodeId, str] = {
    NodeId.TICKET_LOOKUP: (
        "예약 티켓 조회 단계 의사결정 도구. 사용자 ID와 예약 번호가 모두 입력되어 조회를 실행할 준비가 되었으면 EXECUTE, "
        "사용자가 처음으로 돌아가기를 원하면 RETURN_TO_ROOT를 선택하세요."
    ),
    NodeId.SCHEDULE_LOOKUP: (
        "비행기 일정 조회 단계 의사결정 도구. 노선/날짜 또는 전체 일정 조회 조건이 확인되어 일정을 조회할 준비가 되었으면 EXECUTE, "
        "사용자가 처음으로 돌아가기를 원하면 RETURN_TO_ROOT를 선택하세요."
    ),
    NodeId.BOOKING_AUTH: (
        "티켓 예약 인증 단계 의사결정 도구. 사용자 ID와 예약 번호가 모두 입력되어 검증을 실행할 준비가 되었으면 EXECUTE, "
        "사용자가 처음으로 돌아가기를 원하면 RETURN_TO_ROOT를 선택하세요."
    ),
    NodeId.BOOKING_CONFIRM: (
        "티켓 예약 확정 단계 의사결정 도구. 예약할 항공편(편명/노선)이 선택되어 예약을 확정할 준비가 되었으면 EXECUTE, "
        "사용자가 처음으로 돌아가기를 원하면 RETURN_TO_ROOT를 선택하세요."
    ),
}


def build_system_instruction(node_id: NodeId) -> dict[str, Any]:
    """Builds the `Content` JSON dictionary for `systemInstruction`."""
    return {
        "parts": [
            {
                "text": _SYSTEM_INSTRUCTIONS[node_id],
            }
        ]
    }


def build_tools(node_id: NodeId) -> list[dict[str, Any]]:
    """Builds the `tools` array containing EXACTLY ONE Enum-based FunctionDeclaration."""
    tool_name = NODE_TOOL_NAMES[node_id]

    if node_id == NodeId.ROOT_MENU:
        declaration: dict[str, Any] = {
            "name": tool_name,
            "description": (
                "초기 메뉴 분기 의사결정 도구. 사용자의 요청 의도가 1) 예약 티켓 조회(CHECK_TICKET), "
                "2) 비행기 일정 조회(CHECK_SCHEDULE), 3) 비행기 티켓 예약(BOOK_TICKET) 중 하나로 명확히 파악되었을 때만 호출하세요."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "decision": {
                        "type": "STRING",
                        "enum": [item.value for item in RootDecision],
                        "description": (
                            "CHECK_TICKET: 예약된 비행기 티켓 조회, "
                            "CHECK_SCHEDULE: 비행기 운항 일정 조회, "
                            "BOOK_TICKET: 비행기 티켓 예약"
                        ),
                    }
                },
                "required": ["decision"],
            },
        }
    else:
        declaration = {
            "name": tool_name,
            "description": _STEP_TOOL_DESCRIPTIONS[node_id],
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "decision": {
                        "type": "STRING",
                        "enum": [item.value for item in StepDecision],
                        "description": (
                            "EXECUTE: 현재 단계의 필수 입력이 준비되어 작업을 실행함, "
                            "RETURN_TO_ROOT: 작업을 중단하고 처음(메인 메뉴)으로 돌아감"
                        ),
                    }
                },
                "required": ["decision"],
            },
        }

    return [{"functionDeclarations": [declaration]}]
