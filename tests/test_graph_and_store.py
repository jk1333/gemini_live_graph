"""Unit tests for State Graph transitions, single-Enum tools, and MockAirlineStore."""

from __future__ import annotations

import pytest

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.models import (
    NodeId,
    RootDecision,
    StepDecision,
    ToolResultStatus,
)
from airline_chatbot.prompts_and_tools import (
    NODE_TOOL_NAMES,
    build_system_instruction,
    build_tools,
)


def test_every_node_has_single_enum_tool_and_structured_si() -> None:
    for node in NodeId:
        si = build_system_instruction(node)
        text = si["parts"][0]["text"]
        p_idx = text.index("[PERSONA]")
        r_idx = text.index("[CONVERSATIONAL RULES]")
        g_idx = text.index("[GUARDRAILS]")
        assert p_idx < r_idx < g_idx
        assert "RESPOND IN KOREAN. YOU MUST RESPOND UNMISTAKABLY IN KOREAN." in text

        tools = build_tools(node)
        assert len(tools) == 1
        decls = tools[0]["functionDeclarations"]
        assert len(decls) == 1

        decl = decls[0]
        assert decl["name"] == NODE_TOOL_NAMES[node]
        props = decl["parameters"]["properties"]
        assert list(props.keys()) == ["decision"]
        assert props["decision"]["type"] == "STRING"
        assert decl["parameters"]["required"] == ["decision"]

        if node == NodeId.ROOT_MENU:
            assert props["decision"]["enum"] == [d.value for d in RootDecision]
        else:
            assert props["decision"]["enum"] == [d.value for d in StepDecision]


def test_flow_1_ticket_lookup_partial_invalid_and_success() -> None:
    graph = AirlineStateGraph()
    assert graph.current_node == NodeId.ROOT_MENU

    graph.ingest_user_text("예약된 비행기 티켓 조회할래요")
    res = graph.handle_tool_call("route_menu", {"decision": "CHECK_TICKET"})
    assert res.status == ToolResultStatus.DECISION_ROUTED
    assert graph.current_node == NodeId.TICKET_LOOKUP
    pending = graph.consume_pending_context_update()
    assert pending is not None and pending[0] == NodeId.TICKET_LOOKUP

    graph.ingest_user_text("내 아이디는 user01 이야")
    res_missing = graph.handle_tool_call("decide_ticket_lookup", {"decision": "EXECUTE"})
    assert res_missing.status == ToolResultStatus.MISSING_SLOTS
    assert res_missing.retryable is True
    assert graph.current_node == NodeId.TICKET_LOOKUP
    assert graph.consume_pending_context_update() is None

    graph.ingest_user_text("예약번호 BK-1002")
    res_auth_err = graph.handle_tool_call("decide_ticket_lookup", {"decision": "EXECUTE"})
    assert res_auth_err.status == ToolResultStatus.AUTH_INVALID
    assert res_auth_err.retryable is True
    assert graph.current_node == NodeId.TICKET_LOOKUP
    assert graph.consume_pending_context_update() is None

    graph.ingest_user_text("아 예약번호는 bk1001 이었어")
    res_ok = graph.handle_tool_call("decide_ticket_lookup", {"decision": "EXECUTE"})
    assert res_ok.status == ToolResultStatus.EXECUTED_OK
    assert res_ok.data["ticket"]["flight_id"] == "KE101"
    assert res_ok.data["ticket"]["seat"] == "14A"
    assert graph.current_node == NodeId.ROOT_MENU
    pending_back = graph.consume_pending_context_update()
    assert pending_back is not None and pending_back[0] == NodeId.ROOT_MENU


def test_flow_2_schedule_lookup_with_city_filter_and_all() -> None:
    graph = AirlineStateGraph()

    graph.ingest_user_text("도쿄 가는 비행기 일정 알려줘")
    res_route = graph.handle_tool_call("route_menu", {"decision": "CHECK_SCHEDULE"})
    assert res_route.status == ToolResultStatus.DECISION_ROUTED
    assert graph.current_node == NodeId.SCHEDULE_LOOKUP
    assert graph.slots.arrival_code == "NRT"
    graph.consume_pending_context_update()

    graph.ingest_user_text("응 바로 보여줘")
    res_sched = graph.handle_tool_call("decide_schedule_lookup", {"decision": "EXECUTE"})
    assert res_sched.status == ToolResultStatus.EXECUTED_OK
    matched = res_sched.data["matched_flights"]
    assert len(matched) == 1
    assert matched[0]["flight_id"] == "KE101"
    assert graph.current_node == NodeId.ROOT_MENU
    graph.consume_pending_context_update()

    graph.ingest_user_text("비행기 일정 조회")
    graph.handle_tool_call("route_menu", {"decision": "CHECK_SCHEDULE"})
    graph.consume_pending_context_update()

    graph.ingest_user_text("전체 일정 다 보여줘")
    res_all = graph.handle_tool_call("decide_schedule_lookup", {"decision": "EXECUTE"})
    assert res_all.status == ToolResultStatus.EXECUTED_OK
    assert len(res_all.data["matched_flights"]) == 4


def test_flow_3_booking_end_to_end_and_subsequent_lookup() -> None:
    graph = AirlineStateGraph()

    graph.ingest_user_text("비행기 티켓 예약하고 싶어요")
    res = graph.handle_tool_call("route_menu", {"decision": "BOOK_TICKET"})
    assert res.status == ToolResultStatus.DECISION_ROUTED
    assert graph.current_node == NodeId.BOOKING_AUTH
    graph.consume_pending_context_update()

    graph.ingest_user_text("아이디 user01 이고 예약번호 BK-1001")
    res_dup = graph.handle_tool_call("decide_booking_auth", {"decision": "EXECUTE"})
    assert res_dup.status == ToolResultStatus.ALREADY_CONFIRMED
    assert res_dup.retryable is True
    assert graph.current_node == NodeId.BOOKING_AUTH
    assert graph.consume_pending_context_update() is None

    graph.ingest_user_text("예약번호 BK-2001 로 할게")
    res_auth_ok = graph.handle_tool_call("decide_booking_auth", {"decision": "EXECUTE"})
    assert res_auth_ok.status == ToolResultStatus.EXECUTED_OK
    assert graph.current_node == NodeId.BOOKING_CONFIRM
    pending_confirm = graph.consume_pending_context_update()
    assert pending_confirm is not None and pending_confirm[0] == NodeId.BOOKING_CONFIRM

    res_no_flight = graph.handle_tool_call("decide_booking_confirm", {"decision": "EXECUTE"})
    assert res_no_flight.status == ToolResultStatus.MISSING_SLOTS
    assert graph.current_node == NodeId.BOOKING_CONFIRM

    graph.ingest_user_text("OZ202 편으로 예약 확정해줘")
    res_booked = graph.handle_tool_call("decide_booking_confirm", {"decision": "EXECUTE"})
    assert res_booked.status == ToolResultStatus.EXECUTED_OK
    confirmed = res_booked.data["confirmed_ticket"]
    assert confirmed["booking_number"] == "BK-2001"
    assert confirmed["flight_id"] == "OZ202"
    assert confirmed["flight"]["seats_available"] == 7
    assert graph.current_node == NodeId.ROOT_MENU
    graph.consume_pending_context_update()

    graph.handle_tool_call("route_menu", {"decision": "CHECK_TICKET"})
    graph.consume_pending_context_update()
    graph.ingest_user_text("user01 BK-2001")
    res_verify = graph.handle_tool_call("decide_ticket_lookup", {"decision": "EXECUTE"})
    assert res_verify.status == ToolResultStatus.EXECUTED_OK
    assert res_verify.data["ticket"]["flight_id"] == "OZ202"


@pytest.mark.parametrize(
    ("route_decision", "sub_node", "tool_name"),
    [
        ("CHECK_TICKET", NodeId.TICKET_LOOKUP, "decide_ticket_lookup"),
        ("CHECK_SCHEDULE", NodeId.SCHEDULE_LOOKUP, "decide_schedule_lookup"),
        ("BOOK_TICKET", NodeId.BOOKING_AUTH, "decide_booking_auth"),
    ],
)
def test_return_to_root_from_any_branch_clears_slots(
    route_decision: str,
    sub_node: NodeId,
    tool_name: str,
) -> None:
    graph = AirlineStateGraph()
    graph.handle_tool_call("route_menu", {"decision": route_decision})
    assert graph.current_node == sub_node
    graph.consume_pending_context_update()

    graph.ingest_user_text("아이디 user01 인데 아 그냥 처음으로 돌아갈래")
    res = graph.handle_tool_call(tool_name, {"decision": "RETURN_TO_ROOT"})
    assert res.status == ToolResultStatus.RETURNED_TO_ROOT
    assert graph.current_node == NodeId.ROOT_MENU
    assert graph.slots.user_id is None
    pending = graph.consume_pending_context_update()
    assert pending is not None and pending[0] == NodeId.ROOT_MENU


def test_return_to_root_from_booking_confirm() -> None:
    graph = AirlineStateGraph()
    graph.handle_tool_call("route_menu", {"decision": "BOOK_TICKET"})
    graph.consume_pending_context_update()

    graph.ingest_user_text("user02 BK-2002")
    graph.handle_tool_call("decide_booking_auth", {"decision": "EXECUTE"})
    assert graph.current_node == NodeId.BOOKING_CONFIRM
    graph.consume_pending_context_update()

    res = graph.handle_tool_call("decide_booking_confirm", {"decision": "RETURN_TO_ROOT"})
    assert res.status == ToolResultStatus.RETURNED_TO_ROOT
    assert graph.current_node == NodeId.ROOT_MENU
    assert graph.slots.user_id is None
    assert graph.slots.booking_number is None


def test_endpoint_resolution_vertex_and_developer_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import airline_chatbot.live_client as lc

    monkeypatch.setattr(lc, "_fetch_adc_access_token", lambda: "mock-bearer-token")

    vertex_cfg = lc.LiveConnectionConfig(
        model="gemini-3.8-live",
        project_id="demo-proj",
        location="us-central1",
    )
    url, model_res, headers = vertex_cfg.resolve_endpoint_and_headers()
    assert url == (
        "wss://us-central1-aiplatform.googleapis.com/ws/"
        "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
    )
    assert (
        model_res
        == "projects/demo-proj/locations/us-central1/publishers/google/models/gemini-3.8-live"
    )
    assert headers == {"Authorization": "Bearer mock-bearer-token"}

    dev_cfg = lc.LiveConnectionConfig(
        model="gemini-3.8-live",
        api_key="test-api-key",
    )
    dev_url, dev_model, dev_headers = dev_cfg.resolve_endpoint_and_headers()
    assert "generativelanguage.googleapis.com" in dev_url
    assert "key=test-api-key" in dev_url
    assert dev_model == "models/gemini-3.8-live"
    assert dev_headers == {}


def test_safety_settings_builder_and_setup_payload() -> None:
    from airline_chatbot.cli import resolve_cli_args
    from airline_chatbot.live_client import (
        GeminiLiveWebSocketSession,
        LiveConnectionConfig,
        build_safety_settings,
    )

    # 1. DEFAULT omits safetySettings from setup payload
    assert build_safety_settings("DEFAULT") == []
    default_session = GeminiLiveWebSocketSession(
        graph=AirlineStateGraph(),
        config=LiveConnectionConfig(model="gemini-3.8-live"),
    )
    setup_default = default_session.build_setup_payload("models/gemini-3.8-live")
    assert "safetySettings" not in setup_default["setup"]

    # 2. Preset expands across all 4 core HarmCategories
    preset = build_safety_settings(threshold="BLOCK_LOW_AND_ABOVE")
    assert len(preset) == 4
    assert {item["category"] for item in preset} == {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    assert all(item["threshold"] == "BLOCK_LOW_AND_ABOVE" for item in preset)

    strict_session = GeminiLiveWebSocketSession(
        graph=AirlineStateGraph(),
        config=LiveConnectionConfig(model="gemini-3.8-live", safety_settings=preset),
    )
    setup_strict = strict_session.build_setup_payload("models/gemini-3.8-live")
    assert setup_strict["setup"]["safetySettings"] == preset

    # 3. Custom per-category JSON & CLI flag parsing
    custom_json = (
        '[{"category": "HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"}, '
        '{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}, '
        '{"category": "DANGEROUS_CONTENT", "threshold": "DEFAULT"}]'
    )
    _, parsed_cfg, parsed_gr = resolve_cli_args(
        [
            "--project",
            "demo",
            "--safety-settings-json",
            custom_json,
            "--no-guardrail",
        ]
    )
    assert parsed_gr.enabled is False
    assert parsed_cfg.safety_settings == [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ]

    # 4. Invalid threshold raises ValueError
    with pytest.raises(ValueError):
        build_safety_settings(threshold="INVALID_THRESHOLD")

