"""Integration tests verifying the Gemini 3.8 Live WebSocket wire protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import json
import socket
from typing import Any, TypeVar

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.client import ClientProtocol
from websockets.frames import Frame, Opcode
from websockets.protocol import State
from websockets.server import ServerProtocol
from websockets.uri import parse_uri

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.live_client import GeminiLiveWebSocketSession, LiveConnectionConfig
from airline_chatbot.models import NodeId

_T = TypeVar("_T")
_FRAME_TIMEOUT_SEC = 3.0
_OVERALL_TEST_TIMEOUT_SEC = 10.0


def _can_connect_loopback_tcp(timeout_sec: float = 0.3) -> bool:
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssock: socket.socket | None = None
    try:
        lsock.settimeout(timeout_sec)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(1)
        addr, port = lsock.getsockname()[:2]

        csock.setblocking(False)
        try:
            csock.connect((addr, port))
        except (BlockingIOError, InterruptedError):
            pass

        ssock, _ = lsock.accept()
        return True
    except OSError:
        return False
    finally:
        if ssock is not None:
            ssock.close()
        csock.close()
        lsock.close()


def _drive_sync_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    finally:
        coro.close()
    raise AssertionError("Sans-I/O coroutine unexpectedly suspended on I/O")


def _assert_setup_frame(setup_msg: dict[str, Any]) -> None:
    assert "setup" in setup_msg
    assert setup_msg["setup"]["model"] == "models/gemini-3.8-live"
    assert setup_msg["setup"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert "outputAudioTranscription" in setup_msg["setup"]
    assert (
        setup_msg["setup"]["tools"][0]["functionDeclarations"][0]["name"]
        == "route_menu"
    )


def _assert_turn1_tool_response(tool_resp_1: dict[str, Any]) -> None:
    assert "toolResponse" in tool_resp_1
    fn_resp_1 = tool_resp_1["toolResponse"]["functionResponses"][0]
    assert fn_resp_1["id"] == "call_root_1"
    assert fn_resp_1["response"]["status"] == "DECISION_ROUTED"
    assert fn_resp_1["response"]["active_node"] == "TICKET_LOOKUP"


def _assert_turn1_context_update(ctx_update_1: dict[str, Any]) -> None:
    assert "contextUpdate" in ctx_update_1
    tools_wrapper = ctx_update_1["contextUpdate"]["tools"]
    assert isinstance(tools_wrapper, dict)
    assert isinstance(tools_wrapper["tools"], list)
    assert (
        tools_wrapper["tools"][0]["functionDeclarations"][0]["name"]
        == "decide_ticket_lookup"
    )


def _assert_turn2_tool_response(tool_resp_2: dict[str, Any]) -> None:
    assert "toolResponse" in tool_resp_2
    fn_resp_2 = tool_resp_2["toolResponse"]["functionResponses"][0]
    assert fn_resp_2["response"]["status"] == "EXECUTED_OK"
    assert fn_resp_2["response"]["data"]["ticket"]["flight_id"] == "KE101"


def _assert_turn2_context_update(ctx_update_2: dict[str, Any]) -> None:
    assert "contextUpdate" in ctx_update_2
    tools_wrapper = ctx_update_2["contextUpdate"]["tools"]
    assert isinstance(tools_wrapper, dict)
    assert isinstance(tools_wrapper["tools"], list)
    assert (
        tools_wrapper["tools"][0]["functionDeclarations"][0]["name"]
        == "route_menu"
    )


def _assert_session_outcomes(
    *,
    received_frames: list[dict[str, Any]],
    context_transitions: list[NodeId],
    transcripts: list[str],
) -> None:
    assert context_transitions == [NodeId.TICKET_LOOKUP, NodeId.ROOT_MENU]
    joined = "".join(transcripts)
    assert "네, 사용자 ID와 예약 번호를 알려주세요." in joined
    assert "KE101편 도쿄행 티켓" in joined
    assert len(received_frames) == 7


def _exercise_live_websocket_protocol_sans_io() -> None:
    client_proto = ClientProtocol(parse_uri("ws://127.0.0.1:8765/ws"))
    server_proto = ServerProtocol()

    handshake_req = client_proto.connect()
    client_proto.send_request(handshake_req)
    server_proto.receive_data(b"".join(client_proto.data_to_send()))
    req_events = server_proto.events_received()
    assert len(req_events) == 1
    handshake_resp = server_proto.accept(req_events[0])  # type: ignore[arg-type]
    server_proto.send_response(handshake_resp)
    client_proto.receive_data(b"".join(server_proto.data_to_send()))
    client_proto.events_received()
    assert client_proto.state is State.OPEN
    assert server_proto.state is State.OPEN

    received_frames: list[dict[str, Any]] = []
    server_inbox: list[dict[str, Any]] = []

    class _SansIoClientWs:
        async def send(self, data: str) -> None:
            client_proto.send_text(data.encode("utf-8"))
            wire_bytes = b"".join(client_proto.data_to_send())
            server_proto.receive_data(wire_bytes)
            for event in server_proto.events_received():
                if isinstance(event, Frame) and event.opcode is Opcode.TEXT:
                    msg = json.loads(event.data.decode("utf-8"))
                    received_frames.append(msg)
                    server_inbox.append(msg)

        async def close(self) -> None:
            client_proto.send_close(1000)

    def server_send(payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            server_proto.send_binary(payload)
        else:
            server_proto.send_text(payload.encode("utf-8"))
        wire_bytes = b"".join(server_proto.data_to_send())
        client_proto.receive_data(wire_bytes)
        for event in client_proto.events_received():
            if isinstance(event, Frame) and event.opcode in (Opcode.TEXT, Opcode.BINARY):
                _drive_sync_coro(session.process_incoming_frame(event.data))

    graph = AirlineStateGraph()
    transcripts: list[str] = []
    context_transitions: list[NodeId] = []
    config = LiveConnectionConfig(
        model="gemini-3.8-live",
        ws_url_override="ws://127.0.0.1:8765/ws",
    )
    session = GeminiLiveWebSocketSession(
        graph=graph,
        config=config,
        on_text_chunk=transcripts.append,
        on_context_update=context_transitions.append,
    )
    session._ws = _SansIoClientWs()
    _, model_resource, _ = config.resolve_endpoint_and_headers()

    _drive_sync_coro(session._send_json(session.build_setup_payload(model_resource)))
    _assert_setup_frame(server_inbox.pop(0))
    server_send(json.dumps({"setupComplete": {}}).encode("utf-8"))
    assert session._setup_complete.is_set()

    turn1_payload = session.prepare_user_turn_payload("예약 티켓 조회할래요")
    session._turn_complete.clear()
    _drive_sync_coro(session._send_json(turn1_payload))
    assert server_inbox.pop(0) == {"realtimeInput": {"text": "예약 티켓 조회할래요"}}

    server_send(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_root_1",
                            "name": "route_menu",
                            "args": {"decision": "CHECK_TICKET"},
                        }
                    ]
                }
            }
        ).encode("utf-8")
    )
    assert len(server_inbox) == 1
    _assert_turn1_tool_response(server_inbox.pop(0))

    server_send(
        json.dumps(
            {
                "serverContent": {
                    "outputTranscription": {
                        "text": "네, 사용자 ID와 예약 번호를 알려주세요.",
                        "languageCode": "ko-KR",
                    },
                    "turnComplete": False,
                }
            }
        )
    )
    assert server_inbox == []
    server_send(json.dumps({"serverContent": {"turnComplete": True}}))
    assert session._turn_complete.is_set()
    assert graph.current_node == NodeId.TICKET_LOOKUP
    _assert_turn1_context_update(server_inbox.pop(0))

    turn2_payload = session.prepare_user_turn_payload("아이디 user01 예약번호 BK-1001")
    session._turn_complete.clear()
    _drive_sync_coro(session._send_json(turn2_payload))
    assert server_inbox.pop(0) == {"realtimeInput": {"text": "아이디 user01 예약번호 BK-1001"}}

    server_send(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_lookup_1",
                            "name": "decide_ticket_lookup",
                            "args": {"decision": "EXECUTE"},
                        }
                    ]
                }
            }
        )
    )
    assert len(server_inbox) == 1
    _assert_turn2_tool_response(server_inbox.pop(0))

    server_send(
        json.dumps(
            {
                "serverContent": {
                    "outputTranscription": {
                        "text": "김민수님의 KE101편 도쿄행 티켓(좌석 14A)이 확인되었습니다.",
                    },
                    "turnComplete": True,
                }
            }
        )
    )
    assert session._turn_complete.is_set()
    assert graph.current_node == NodeId.ROOT_MENU
    _assert_turn2_context_update(server_inbox.pop(0))

    _assert_session_outcomes(
        received_frames=received_frames,
        context_transitions=context_transitions,
        transcripts=transcripts,
    )


async def _recv_json(ws: ServerConnection) -> dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout=_FRAME_TIMEOUT_SEC)
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return json.loads(text)


async def _exercise_live_websocket_protocol() -> None:
    received_frames: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    server_done: asyncio.Future[None] = loop.create_future()

    async def mock_gemini_live_server(ws: ServerConnection) -> None:
        try:
            setup_msg = await _recv_json(ws)
            received_frames.append(setup_msg)
            _assert_setup_frame(setup_msg)

            await ws.send(json.dumps({"setupComplete": {}}).encode("utf-8"))

            turn1_msg = await _recv_json(ws)
            received_frames.append(turn1_msg)
            assert turn1_msg == {"realtimeInput": {"text": "예약 티켓 조회할래요"}}

            await ws.send(
                json.dumps(
                    {
                        "toolCall": {
                            "functionCalls": [
                                {
                                    "id": "call_root_1",
                                    "name": "route_menu",
                                    "args": {"decision": "CHECK_TICKET"},
                                }
                            ]
                        }
                    }
                ).encode("utf-8")
            )

            tool_resp_1 = await _recv_json(ws)
            received_frames.append(tool_resp_1)
            _assert_turn1_tool_response(tool_resp_1)

            await ws.send(
                json.dumps(
                    {
                        "serverContent": {
                            "outputTranscription": {
                                "text": "네, 사용자 ID와 예약 번호를 알려주세요.",
                                "languageCode": "ko-KR",
                            },
                            "turnComplete": False,
                        }
                    }
                )
            )
            await ws.send(json.dumps({"serverContent": {"turnComplete": True}}))

            ctx_update_1 = await _recv_json(ws)
            received_frames.append(ctx_update_1)
            _assert_turn1_context_update(ctx_update_1)

            turn2_msg = await _recv_json(ws)
            received_frames.append(turn2_msg)
            assert turn2_msg == {"realtimeInput": {"text": "아이디 user01 예약번호 BK-1001"}}

            await ws.send(
                json.dumps(
                    {
                        "toolCall": {
                            "functionCalls": [
                                {
                                    "id": "call_lookup_1",
                                    "name": "decide_ticket_lookup",
                                    "args": {"decision": "EXECUTE"},
                                }
                            ]
                        }
                    }
                )
            )

            tool_resp_2 = await _recv_json(ws)
            received_frames.append(tool_resp_2)
            _assert_turn2_tool_response(tool_resp_2)

            await ws.send(
                json.dumps(
                    {
                        "serverContent": {
                            "outputTranscription": {
                                "text": "김민수님의 KE101편 도쿄행 티켓(좌석 14A)이 확인되었습니다.",
                            },
                            "turnComplete": True,
                        }
                    }
                )
            )

            ctx_update_2 = await _recv_json(ws)
            received_frames.append(ctx_update_2)
            _assert_turn2_context_update(ctx_update_2)

            if not server_done.done():
                server_done.set_result(None)
        except BaseException as exc:
            if not server_done.done():
                server_done.set_exception(exc)
            raise

    async with websockets.serve(
        mock_gemini_live_server,
        "127.0.0.1",
        0,
        open_timeout=5.0,
        close_timeout=2.0,
        ping_interval=None,
    ) as server:
        port = server.sockets[0].getsockname()[1]
        ws_url = f"ws://127.0.0.1:{port}"

        graph = AirlineStateGraph()
        transcripts: list[str] = []
        context_transitions: list[NodeId] = []

        config = LiveConnectionConfig(
            model="gemini-3.8-live",
            ws_url_override=ws_url,
            setup_timeout_sec=5.0,
            turn_timeout_sec=5.0,
        )

        async with GeminiLiveWebSocketSession(
            graph=graph,
            config=config,
            on_text_chunk=transcripts.append,
            on_context_update=context_transitions.append,
        ) as session:
            await session.send_user_turn("예약 티켓 조회할래요")
            assert graph.current_node == NodeId.TICKET_LOOKUP

            await session.send_user_turn("아이디 user01 예약번호 BK-1001")
            assert graph.current_node == NodeId.ROOT_MENU

            await asyncio.wait_for(server_done, timeout=_FRAME_TIMEOUT_SEC)

    _assert_session_outcomes(
        received_frames=received_frames,
        context_transitions=context_transitions,
        transcripts=transcripts,
    )


def test_live_websocket_wire_framing_sans_io() -> None:
    _exercise_live_websocket_protocol_sans_io()


def test_live_websocket_session_handshake_toolcall_and_context_update() -> None:
    if not _can_connect_loopback_tcp():
        _exercise_live_websocket_protocol_sans_io()
        return

    asyncio.run(
        asyncio.wait_for(
            _exercise_live_websocket_protocol(),
            timeout=_OVERALL_TEST_TIMEOUT_SEC,
        )
    )


def test_vertex_intermediate_turn_complete_and_danang_schedule_regression() -> None:
    sent_by_client: list[dict[str, Any]] = []

    class _DummyWs:
        async def send(self, data: str) -> None:
            sent_by_client.append(json.loads(data))

        async def close(self) -> None:
            return None

    graph = AirlineStateGraph()
    transcripts: list[str] = []
    transitions: list[NodeId] = []
    session = GeminiLiveWebSocketSession(
        graph=graph,
        config=LiveConnectionConfig(
            model="gemini-3.8-live",
            ws_url_override="ws://127.0.0.1:8765/ws",
        ),
        on_text_chunk=transcripts.append,
        on_context_update=transitions.append,
    )
    session._ws = _DummyWs()

    turn1 = session.prepare_user_turn_payload("비행기 일정 조회")
    session._turn_complete.clear()
    _drive_sync_coro(session._send_json(turn1))
    sent_by_client.clear()

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call_sched_route",
                                "name": "route_menu",
                                "args": {"decision": "CHECK_SCHEDULE"},
                            }
                        ]
                    }
                }
            )
        )
    )
    assert len(sent_by_client) == 1
    assert "toolResponse" in sent_by_client[0]

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "serverContent": {
                        "turnComplete": True,
                        "interactionStatus": "REQUIRES_ACTION",
                    }
                }
            )
        )
    )
    assert not session._turn_complete.is_set()
    assert len(sent_by_client) == 1

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps({"serverContent": {"turnComplete": True}})
        )
    )
    assert not session._turn_complete.is_set()
    assert len(sent_by_client) == 1

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "serverContent": {
                        "outputTranscription": {
                            "text": "어느 노선의 비행기 일정을 조회해 드릴까요?",
                        },
                        "interactionStatus": "IN_PROGRESS",
                        "turnComplete": False,
                    }
                }
            )
        )
    )
    assert not session._turn_complete.is_set()

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "serverContent": {
                        "turnComplete": True,
                        "interactionStatus": "IDLE",
                    }
                }
            )
        )
    )
    assert session._turn_complete.is_set()
    assert len(sent_by_client) == 2
    ctx_frame = sent_by_client[1]
    assert ctx_frame["contextUpdate"]["tools"] == {
        "tools": graph.get_context_config(NodeId.SCHEDULE_LOOKUP)["tools"]
    }
    assert transitions == [NodeId.SCHEDULE_LOOKUP]

    sent_by_client.clear()
    turn2 = session.prepare_user_turn_payload("다낭행 항공편을 알려줘")
    session._turn_complete.clear()
    _drive_sync_coro(session._send_json(turn2))

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call_sched_exec",
                                "name": "decide_schedule_lookup",
                                "args": {"decision": "EXECUTE"},
                            }
                        ]
                    }
                }
            )
        )
    )
    tool_resp_2 = sent_by_client[1]["toolResponse"]["functionResponses"][0]["response"]
    assert tool_resp_2["status"] == "EXECUTED_OK"
    assert tool_resp_2["data"]["matched_flights"][0]["flight_id"] == "TW404"

    _drive_sync_coro(
        session.process_incoming_frame(
            json.dumps(
                {
                    "serverContent": {
                        "outputTranscription": {
                            "text": "인천발 다낭행 TW404편이 19시 40분에 출발합니다.",
                        },
                        "turnComplete": True,
                        "interactionStatus": "IDLE",
                    }
                }
            )
        )
    )
    assert session._turn_complete.is_set()
    assert transitions == [NodeId.SCHEDULE_LOOKUP, NodeId.ROOT_MENU]
    assert (
        sent_by_client[2]["contextUpdate"]["tools"]["tools"][0]["functionDeclarations"][0]["name"]
        == "route_menu"
    )
