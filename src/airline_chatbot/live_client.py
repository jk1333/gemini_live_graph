"""Raw WebSocket client for Gemini 3.8 Live (`BidiGenerateContent`).

Supports both:
- Google Cloud Vertex AI (`LlmBidiService/BidiGenerateContent` via ADC Bearer token)
- Gemini Developer API (`GenerativeService.BidiGenerateContent` via `GEMINI_API_KEY`)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.parse import urlparse

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import websockets
from websockets.asyncio.client import ClientConnection

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.models import DecisionResult, NodeId


@dataclass
class LiveConnectionConfig:
    """Configuration for connecting to Gemini 3.8 Live over WebSockets."""

    model: str = "gemini-3.8-live"
    project_id: str | None = None
    location: str = "us-central1"
    api_key: str | None = None
    ws_url_override: str | None = None
    language_code: str = "ko-KR"
    setup_timeout_sec: float = 20.0
    turn_timeout_sec: float = 60.0

    @classmethod
    def from_env(cls) -> LiveConnectionConfig:
        return cls(
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live").strip(),
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip(),
            api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            ws_url_override=os.getenv("GEMINI_WS_URL_OVERRIDE", "").strip() or None,
        )

    def resolve_endpoint_and_headers(self) -> tuple[str, str, dict[str, str]]:
        """Returns `(ws_url, full_model_resource, extra_headers)`."""
        if self.ws_url_override:
            model_res = (
                self.model
                if self.model.startswith(("models/", "projects/"))
                else f"models/{self.model}"
            )
            return self.ws_url_override, model_res, {}

        if self.project_id:
            host = (
                "aiplatform.googleapis.com"
                if self.location == "global"
                else f"{self.location}-aiplatform.googleapis.com"
            )
            ws_url = (
                f"wss://{host}/ws/google.cloud.aiplatform.v1beta1."
                "LlmBidiService/BidiGenerateContent"
            )
            model_res = (
                self.model
                if self.model.startswith("projects/")
                else (
                    f"projects/{self.project_id}/locations/{self.location}/"
                    f"publishers/google/models/{self.model}"
                )
            )
            token = _fetch_adc_access_token()
            return ws_url, model_res, {"Authorization": f"Bearer {token}"}

        if self.api_key:
            ws_url = (
                "wss://generativelanguage.googleapis.com/ws/"
                "google.ai.generativelanguage.v1beta.GenerativeService."
                f"BidiGenerateContent?key={self.api_key}"
            )
            model_res = (
                self.model
                if self.model.startswith("models/")
                else f"models/{self.model}"
            )
            return ws_url, model_res, {}

        raise RuntimeError(
            "인증 설정이 없습니다. 환경 변수 GOOGLE_CLOUD_PROJECT (Vertex AI + ADC) "
            "또는 GEMINI_API_KEY (Gemini Developer API) 중 하나를 설정해 주세요."
        )


def _fetch_adc_access_token() -> str:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(GoogleAuthRequest())
    if not credentials.token:
        raise RuntimeError(
            "Google Cloud ADC 액세스 토큰을 발급받지 못했습니다. "
            "`gcloud auth application-default login`을 먼저 실행해 주세요."
        )
    return str(credentials.token)


def _is_loopback_ws_url(ws_url: str) -> bool:
    hostname = (urlparse(ws_url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


class GeminiLiveWebSocketSession:
    """Manages an active bidirectional Gemini 3.8 Live WebSocket session."""

    def __init__(
        self,
        graph: AirlineStateGraph,
        config: LiveConnectionConfig,
        *,
        on_text_chunk: Callable[[str], None] | None = None,
        on_tool_event: Callable[[str, dict[str, Any], DecisionResult], None] | None = None,
        on_context_update: Callable[[NodeId], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self._on_text_chunk = on_text_chunk or (lambda _: None)
        self._on_tool_event = on_tool_event or (lambda *_: None)
        self._on_context_update = on_context_update or (lambda _: None)
        self._on_status = on_status or (lambda _: None)

        self._ws: Any = None
        self._recv_task: asyncio.Task[None] | None = None
        self._setup_complete = asyncio.Event()
        self._turn_complete = asyncio.Event()
        self._closed_error: BaseException | None = None
        self._closing = False
        self._emitted_transcription = False
        self._awaiting_post_tool_turn = False
        self._received_post_tool_content = False

    async def __aenter__(self) -> GeminiLiveWebSocketSession:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def build_setup_payload(self, model_resource: str) -> dict[str, Any]:
        """Builds the initial `setup` frame payload for `ROOT_MENU`."""
        initial_ctx = self.graph.get_context_config(NodeId.ROOT_MENU)
        return {
            "setup": {
                "model": model_resource,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "languageCode": self.config.language_code,
                    },
                },
                "outputAudioTranscription": {},
                "contextWindowCompression": {
                    "slidingWindow": {},
                },
                "systemInstruction": initial_ctx["systemInstruction"],
                "tools": initial_ctx["tools"],
            }
        }

    @staticmethod
    def build_context_update_payload(context_config: dict[str, Any]) -> dict[str, Any]:
        """Builds `BidiGenerateContentContextUpdate` with `Tools` message wrapper.

        In `BidiGenerateContentContextUpdate`, `tools` is a singular `Tools` wrapper
        message (`{"tools": [...]}`) rather than a bare repeated list so the server can
        distinguish an unset `tools` field from clearing all tools (`{"tools": []}`).
        """
        return {
            "contextUpdate": {
                "systemInstruction": context_config["systemInstruction"],
                "tools": {
                    "tools": context_config["tools"],
                },
            }
        }

    async def connect(self) -> None:
        needs_adc_thread = bool(
            self.config.project_id and not self.config.ws_url_override
        )
        if needs_adc_thread:
            ws_url, model_resource, headers = await asyncio.to_thread(
                self.config.resolve_endpoint_and_headers
            )
        else:
            ws_url, model_resource, headers = self.config.resolve_endpoint_and_headers()

        connect_kwargs: dict[str, Any] = {
            "additional_headers": headers or None,
            "max_size": 16 * 1024 * 1024,
            "open_timeout": self.config.setup_timeout_sec,
            "close_timeout": min(self.config.setup_timeout_sec, 5.0),
        }
        if _is_loopback_ws_url(ws_url):
            connect_kwargs["proxy"] = None

        self._closing = False
        self._closed_error = None
        self._ws = await websockets.connect(ws_url, **connect_kwargs)
        self._emitted_transcription = False
        self._awaiting_post_tool_turn = False
        self._received_post_tool_content = False
        self._recv_task = asyncio.create_task(self._receive_loop())

        try:
            await self._send_json(self.build_setup_payload(model_resource))
            await asyncio.wait_for(
                self._setup_complete.wait(),
                timeout=self.config.setup_timeout_sec,
            )
            if self._closed_error is not None:
                raise RuntimeError(
                    f"세션 초기화 중 연결이 종료되었습니다: {self._closed_error}"
                )
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._closing = True
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        self._recv_task = None

        if self._ws is not None:
            ws: ClientConnection | Any = self._ws
            self._ws = None
            try:
                await asyncio.wait_for(ws.close(), timeout=5.0)
            except (TimeoutError, OSError):
                pass

    def prepare_user_turn_payload(self, text: str) -> dict[str, Any]:
        """Extracts slots from user text and builds the `realtimeInput` JSON frame."""
        self.graph.ingest_user_text(text)
        self._awaiting_post_tool_turn = False
        self._received_post_tool_content = False
        return {
            "realtimeInput": {
                "text": text,
            }
        }

    async def send_user_turn(self, text: str) -> None:
        """Extracts slots, sends `realtimeInput.text`, and waits for turn + contextUpdate."""
        if self._ws is None:
            raise RuntimeError("WebSocket 세션이 연결되어 있지 않습니다.")
        if self._closed_error is not None:
            raise RuntimeError(f"세션이 이미 종료되었습니다: {self._closed_error}")

        payload = self.prepare_user_turn_payload(text)
        self._turn_complete.clear()

        await self._send_json(payload)

        await asyncio.wait_for(
            self._turn_complete.wait(),
            timeout=self.config.turn_timeout_sec,
        )
        if self._closed_error is not None:
            raise RuntimeError(f"응답 대기 중 연결 오류 발생: {self._closed_error}")

    async def send_manual_reset(self) -> None:
        """Forces an immediate mid-session `contextUpdate` back to `ROOT_MENU`."""
        ctx = self.graph.force_reset_to_root()
        await self.send_context_update(ctx)
        self._on_context_update(NodeId.ROOT_MENU)

    async def send_context_update(self, context_config: dict[str, Any]) -> None:
        """Sends a Gemini 3.8 Live `BidiGenerateContentContextUpdate` frame."""
        await self._send_json(self.build_context_update_payload(context_config))

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket 연결이 없습니다.")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def process_incoming_frame(self, raw_frame: str | bytes | bytearray) -> None:
        """Decodes and processes a single incoming server WebSocket frame."""
        text_frame = (
            raw_frame.decode("utf-8")
            if isinstance(raw_frame, (bytes, bytearray))
            else raw_frame
        )
        msg: dict[str, Any] = json.loads(text_frame)

        if "setupComplete" in msg or "setup_complete" in msg:
            self._setup_complete.set()
            return

        if "goAway" in msg or "go_away" in msg:
            self._on_status("[Server] go_away 수신: 곧 세션 재연결이 필요할 수 있습니다.")
            return

        server_content = msg.get("serverContent") or msg.get("server_content")
        tool_call = msg.get("toolCall") or msg.get("tool_call")

        if server_content:
            out_trans = (
                server_content.get("outputTranscription")
                or server_content.get("output_transcription")
            )
            model_turn = (
                server_content.get("modelTurn")
                or server_content.get("model_turn")
            )
            has_payload_content = bool(
                (out_trans and out_trans.get("text"))
                or (model_turn and model_turn.get("parts"))
            )
            if has_payload_content and self._awaiting_post_tool_turn and not tool_call:
                self._received_post_tool_content = True

            if out_trans and out_trans.get("text"):
                self._emitted_transcription = True
                self._on_text_chunk(str(out_trans["text"]))
            elif not self._emitted_transcription and model_turn:
                for part in model_turn.get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        self._on_text_chunk(str(part["text"]))

        if tool_call:
            await self._handle_tool_call_frame(tool_call)
            return

        if server_content and self._is_interaction_finished(msg, server_content):
            self._emitted_transcription = False
            self._awaiting_post_tool_turn = False
            self._received_post_tool_content = False
            pending = self.graph.consume_pending_context_update()
            if pending is not None:
                next_node, next_ctx = pending
                await self.send_context_update(next_ctx)
                self._on_context_update(next_node)
            self._turn_complete.set()

    def _is_interaction_finished(
        self,
        msg: dict[str, Any],
        server_content: dict[str, Any],
    ) -> bool:
        """Determines whether the full turn (including any tool roundtrip) is done."""
        turn_done = bool(
            server_content.get("turnComplete")
            or server_content.get("turn_complete")
        )
        explicit_turn_false = (
            server_content.get("turnComplete") is False
            or server_content.get("turn_complete") is False
        )

        raw_reason = (
            server_content.get("turnCompleteReason")
            or server_content.get("turn_complete_reason")
            or ""
        )
        norm_reason = str(raw_reason).strip().upper()
        has_abnormal_reason = bool(
            norm_reason and norm_reason != "TURN_COMPLETE_REASON_UNSPECIFIED"
        )
        if has_abnormal_reason:
            self._on_status(f"[Server] 턴 종료 사유: {norm_reason}")

        raw_status = (
            server_content.get("interactionStatus")
            or server_content.get("interaction_status")
            or msg.get("interactionStatus")
            or msg.get("interaction_status")
            or ""
        )
        norm_status = str(raw_status).strip().upper()

        if norm_status.endswith(("IN_PROGRESS", "REQUIRES_ACTION")):
            return False

        if norm_status.endswith("IDLE"):
            return not explicit_turn_false

        if not turn_done:
            return False

        if (
            self._awaiting_post_tool_turn
            and not self._received_post_tool_content
            and not has_abnormal_reason
        ):
            return False

        return True

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        self._emitted_transcription = False

        try:
            async for raw_frame in self._ws:
                await self.process_incoming_frame(raw_frame)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            self._closed_error = exc
            if not self._closing:
                self._on_status(f"[연결 종료] {exc}")
            self._setup_complete.set()
            self._turn_complete.set()

    async def _handle_tool_call_frame(self, tool_call: dict[str, Any]) -> None:
        calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or []
        function_responses: list[dict[str, Any]] = []

        for call in calls:
            call_id = call.get("id", "")
            func_name = call.get("name", "")
            args = call.get("args") or {}

            result = self.graph.handle_tool_call(func_name, args)
            self._on_tool_event(func_name, args, result)

            resp_entry: dict[str, Any] = {
                "name": func_name,
                "response": result.to_response_dict(),
            }
            if call_id:
                resp_entry["id"] = call_id
            function_responses.append(resp_entry)

        if function_responses:
            self._awaiting_post_tool_turn = True
            self._received_post_tool_content = False
            self._emitted_transcription = False
            await self._send_json(
                {
                    "toolResponse": {
                        "functionResponses": function_responses,
                    }
                }
            )
