"""FastAPI + WebSocket web server bridging the HTML UI to Gemini 3.8 Live & Model Armor."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
import sys
import threading
from typing import Any
import webbrowser

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.guardrail import (
    ModelArmorGuardrailChecker,
    ModelArmorGuardrailConfig,
)
from airline_chatbot.live_client import (
    GeminiLiveWebSocketSession,
    LiveConnectionConfig,
    build_safety_settings,
)
from airline_chatbot.models import (
    CORE_HARM_CATEGORIES,
    DecisionResult,
    NodeId,
    SafetyThresholdOption,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _describe_backend(config: LiveConnectionConfig) -> str:
    if config.ws_url_override:
        return f"Custom ({config.ws_url_override})"
    if config.project_id:
        return f"Vertex AI ({config.project_id} / {config.location})"
    if config.api_key:
        return "Gemini Developer API"
    return "Unconfigured (GOOGLE_CLOUD_PROJECT 또는 GEMINI_API_KEY 필요)"


def _safety_options_metadata() -> dict[str, Any]:
    return {
        "thresholds": [opt.value for opt in SafetyThresholdOption],
        "categories": [cat.value for cat in CORE_HARM_CATEGORIES],
    }


class WebChatBridge:
    """Coordinates real-time Graph State, Model Armor Guardrail & Gemini Live events."""

    def __init__(
        self,
        session_cfg: LiveConnectionConfig,
        send_event: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        guardrail_cfg: ModelArmorGuardrailConfig | None = None,
        session_factory: Callable[..., Any] = GeminiLiveWebSocketSession,
        guardrail_factory: Callable[[ModelArmorGuardrailConfig], Any] = ModelArmorGuardrailChecker,
    ) -> None:
        self.session_cfg = session_cfg
        self._send_event = send_event
        self._session_factory = session_factory
        resolved_gr_cfg = guardrail_cfg or ModelArmorGuardrailConfig.from_env(
            fallback_project_id=session_cfg.project_id,
            fallback_location=session_cfg.location,
        )
        self.guardrail = guardrail_factory(resolved_gr_cfg)
        self.graph = AirlineStateGraph()
        self._tracked_node: str = self.graph.current_node.value
        self._pending_events: deque[dict[str, Any]] = deque()
        self._live_session: Any = None

    async def close(self) -> None:
        await self.guardrail.close()

    def _enqueue(self, event: dict[str, Any]) -> None:
        event.setdefault("timestamp", _now_hms())
        self._pending_events.append(event)

    async def flush(self) -> None:
        while self._pending_events:
            item = self._pending_events.popleft()
            await self._send_event(item)

    def build_init_event(self) -> dict[str, Any]:
        return {
            "type": "init_state",
            "timestamp": _now_hms(),
            "topology": AirlineStateGraph.get_topology_metadata(),
            "snapshot": self.graph.get_state_snapshot(),
            "model": self.session_cfg.model,
            "backend": _describe_backend(self.session_cfg),
            "safety_settings": self.session_cfg.safety_settings,
            "safety_options": _safety_options_metadata(),
            "guardrail": self.guardrail.config.to_public_dict(),
        }

    def create_live_session(
        self,
        *,
        notify_flush: Callable[[], None] | None = None,
    ) -> Any:
        notify = notify_flush or (lambda: None)

        def on_slots_updated() -> None:
            self._enqueue(
                {
                    "type": "slots_updated",
                    "snapshot": self.graph.get_state_snapshot(),
                }
            )
            notify()

        def on_text_chunk(chunk: str) -> None:
            self._enqueue(
                {
                    "type": "assistant_chunk",
                    "text": chunk,
                }
            )
            notify()

        def on_tool_event(
            func_name: str,
            args: dict[str, Any],
            result: DecisionResult,
        ) -> None:
            source_node = self._tracked_node
            target_node = result.active_node.value
            self._tracked_node = target_node
            self._enqueue(
                {
                    "type": "tool_call",
                    "from_node": source_node,
                    "to_node": target_node,
                    "tool_name": func_name,
                    "decision": str(args.get("decision", "")),
                    "result": result.to_response_dict(),
                    "cache_preserved": source_node == target_node,
                    "snapshot": self.graph.get_state_snapshot(),
                }
            )
            notify()

        def on_context_update(next_node: NodeId) -> None:
            self._tracked_node = next_node.value
            self._enqueue(
                {
                    "type": "context_update",
                    "target_node": next_node.value,
                    "snapshot": self.graph.get_state_snapshot(),
                }
            )
            notify()

        def on_status(message: str) -> None:
            self._enqueue(
                {
                    "type": "status",
                    "message": message,
                }
            )
            notify()

        self._live_session = self._session_factory(
            graph=self.graph,
            config=self.session_cfg,
            on_text_chunk=on_text_chunk,
            on_tool_event=on_tool_event,
            on_context_update=on_context_update,
            on_slots_updated=on_slots_updated,
            on_status=on_status,
        )
        return self._live_session

    async def handle_client_message(self, raw_msg: dict[str, Any]) -> None:
        if self._live_session is None:
            raise RuntimeError("Live 세션이 초기화되지 않았습니다.")

        msg_type = str(raw_msg.get("type", "")).strip()
        if msg_type == "set_guardrail":
            enabled = raw_msg.get("enabled")
            template_id = raw_msg.get("template_id")
            location = raw_msg.get("location")
            auto_create = raw_msg.get("auto_create_template")
            self.guardrail.update_settings(
                enabled=bool(enabled) if enabled is not None else None,
                template_id=str(template_id) if template_id is not None else None,
                location=str(location) if location is not None else None,
                auto_create_template=bool(auto_create) if auto_create is not None else None,
            )
            self._enqueue(
                {
                    "type": "guardrail_config_updated",
                    "guardrail": self.guardrail.config.to_public_dict(),
                }
            )
            await self.flush()
            return

        if msg_type == "user_message":
            user_text = str(raw_msg.get("text", "")).strip()
            if not user_text:
                return
            self._tracked_node = self.graph.current_node.value
            self._enqueue(
                {
                    "type": "turn_started",
                    "text": user_text,
                    "active_node": self._tracked_node,
                }
            )
            await self.flush()

            # Step 1: Google Cloud Model Armor (`sanitizeUserPrompt`) Check
            check_res = await self.guardrail.check_user_input(
                user_text=user_text,
                active_node=self.graph.current_node,
            )
            self._enqueue(
                {
                    "type": "guardrail_checked",
                    "check": check_res.to_dict(),
                    "snapshot": self.graph.get_state_snapshot(),
                }
            )
            await self.flush()

            if not check_res.allowed:
                matched_str = ", ".join(check_res.matched_filters) or check_res.category
                rejection_msg = (
                    "죄송합니다. 입력하신 메시지가 Google Cloud Model Armor 보안 정책에 의해 차단되었습니다.\n"
                    f"• 감지된 필터: {matched_str} ({check_res.filter_match_state})\n"
                    f"• 상세 사유: {check_res.reason}\n"
                    f"• 검사 소요 시간: {check_res.latency_ms:.1f} ms"
                )
                self._enqueue(
                    {
                        "type": "assistant_chunk",
                        "text": rejection_msg,
                    }
                )
                self._enqueue(
                    {
                        "type": "turn_complete",
                        "snapshot": self.graph.get_state_snapshot(),
                    }
                )
                await self.flush()
                return

            # Step 2: Forward compliant input to Gemini 3.8 Live WebSocket session
            try:
                await self._live_session.send_user_turn(user_text)
            except Exception as turn_exc:  # noqa: BLE001
                self._enqueue(
                    {
                        "type": "error",
                        "message": str(turn_exc),
                        "snapshot": self.graph.get_state_snapshot(),
                    }
                )
            else:
                self._tracked_node = self.graph.current_node.value
                self._enqueue(
                    {
                        "type": "turn_complete",
                        "snapshot": self.graph.get_state_snapshot(),
                    }
                )
            await self.flush()

        elif msg_type == "manual_reset":
            prev_node = self.graph.current_node.value
            await self._live_session.send_manual_reset()
            self._tracked_node = self.graph.current_node.value
            self._enqueue(
                {
                    "type": "manual_reset_done",
                    "from_node": prev_node,
                    "to_node": self._tracked_node,
                    "snapshot": self.graph.get_state_snapshot(),
                }
            )
            await self.flush()


def create_web_app(
    default_config: LiveConnectionConfig | None = None,
    *,
    default_guardrail_config: ModelArmorGuardrailConfig | None = None,
    session_factory: Callable[..., Any] = GeminiLiveWebSocketSession,
    guardrail_factory: Callable[[ModelArmorGuardrailConfig], Any] = ModelArmorGuardrailChecker,
) -> FastAPI:
    """Creates the FastAPI application serving the HTML dashboard and `/ws/chat`."""
    base_cfg = default_config or LiveConnectionConfig.from_env()
    base_gr_cfg = default_guardrail_config or ModelArmorGuardrailConfig.from_env(
        fallback_project_id=base_cfg.project_id,
        fallback_location=base_cfg.location,
    )
    app = FastAPI(
        title="SkyLine Airlines Live Graph Studio",
        version="0.1.0",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_content = INDEX_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)

    @app.get("/api/topology", response_class=JSONResponse)
    async def topology() -> JSONResponse:
        preview_graph = AirlineStateGraph()
        return JSONResponse(
            content={
                "topology": AirlineStateGraph.get_topology_metadata(),
                "initial_snapshot": preview_graph.get_state_snapshot(),
                "safety_options": _safety_options_metadata(),
                "guardrail": base_gr_cfg.to_public_dict(),
                "connection": {
                    "model": base_cfg.model,
                    "project_id": base_cfg.project_id,
                    "location": base_cfg.location,
                    "backend": _describe_backend(base_cfg),
                    "safety_settings": base_cfg.safety_settings,
                },
            }
        )

    @app.websocket("/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        await websocket.accept()

        qp = websocket.query_params
        try:
            if "safety_json" in qp:
                resolved_safety = build_safety_settings(
                    custom_settings=qp.get("safety_json", "")
                )
            elif "safety_threshold" in qp:
                resolved_safety = build_safety_settings(
                    threshold=qp.get("safety_threshold", "")
                )
            else:
                resolved_safety = [dict(item) for item in base_cfg.safety_settings]
        except Exception as safety_exc:  # noqa: BLE001
            await websocket.send_json(
                {
                    "type": "error",
                    "timestamp": _now_hms(),
                    "message": f"Safety 설정 오류: {safety_exc}",
                }
            )
            await websocket.close()
            return

        session_cfg = LiveConnectionConfig(
            model=(qp.get("model") or base_cfg.model).strip(),
            project_id=(qp.get("project") or base_cfg.project_id or "").strip() or None,
            location=(qp.get("location") or base_cfg.location or "us-central1").strip(),
            api_key=base_cfg.api_key,
            ws_url_override=base_cfg.ws_url_override,
            language_code=base_cfg.language_code,
            safety_settings=resolved_safety,
            setup_timeout_sec=base_cfg.setup_timeout_sec,
            turn_timeout_sec=base_cfg.turn_timeout_sec,
        )

        tab_gr_cfg = ModelArmorGuardrailConfig(
            enabled=(
                qp.get("guardrail_enabled", "").lower() in {"1", "true", "on"}
                if "guardrail_enabled" in qp
                else base_gr_cfg.enabled
            ),
            project_id=session_cfg.project_id or base_gr_cfg.project_id,
            location=(qp.get("model_armor_location") or base_gr_cfg.location).strip(),
            template_id=(qp.get("model_armor_template") or base_gr_cfg.template_id).strip(),
            auto_create_template=base_gr_cfg.auto_create_template,
            endpoint_override=base_gr_cfg.endpoint_override,
            timeout_sec=base_gr_cfg.timeout_sec,
        )

        flush_wakeup = asyncio.Event()
        stop_flusher = False
        bridge = WebChatBridge(
            session_cfg=session_cfg,
            send_event=websocket.send_json,
            guardrail_cfg=tab_gr_cfg,
            session_factory=session_factory,
            guardrail_factory=guardrail_factory,
        )

        async def background_flusher() -> None:
            while not stop_flusher:
                await flush_wakeup.wait()
                flush_wakeup.clear()
                await bridge.flush()

        flusher_task = asyncio.create_task(background_flusher())

        await websocket.send_json(bridge.build_init_event())

        try:
            live_session = bridge.create_live_session(notify_flush=flush_wakeup.set)
            async with live_session:
                await websocket.send_json(
                    {
                        "type": "session_ready",
                        "timestamp": _now_hms(),
                        "model": session_cfg.model,
                        "backend": _describe_backend(session_cfg),
                        "safety_settings": session_cfg.safety_settings,
                        "guardrail": bridge.guardrail.config.to_public_dict(),
                        "snapshot": bridge.graph.get_state_snapshot(),
                    }
                )

                while True:
                    raw_msg = await websocket.receive_json()
                    await bridge.handle_client_message(raw_msg)

        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "timestamp": _now_hms(),
                        "message": str(exc),
                        "snapshot": bridge.graph.get_state_snapshot(),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            stop_flusher = True
            flush_wakeup.set()
            try:
                await flusher_task
            except Exception:  # noqa: BLE001
                pass
            await bridge.close()

    return app


def build_web_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airline-web",
        description="Gemini 3.8 Live Airline Ticket Chatbot HTML Web App + Real-Time Graph Tracker",
    )
    parser.add_argument(
        "--project",
        help="Google Cloud Project ID (기본값: 환경변수 GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--location",
        help="Vertex AI 및 Model Armor 리전 (기본값: 환경변수 GOOGLE_CLOUD_LOCATION 또는 us-central1)",
    )
    parser.add_argument(
        "--model",
        help="사용할 Gemini Live 모델명 (기본값: 환경변수 GEMINI_LIVE_MODEL 또는 gemini-3.8-live)",
    )
    parser.add_argument(
        "--safety-threshold",
        choices=[opt.value for opt in SafetyThresholdOption],
        help=(
            "초기 setup 시 4대 유해 카테고리에 일괄 적용할 Safety Filter 임계치 "
            "(DEFAULT, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH, BLOCK_NONE, OFF)"
        ),
    )
    parser.add_argument(
        "--safety-settings-json",
        help=(
            "카테고리별 개별 Safety Filter JSON 배열 문자열 "
            '(예: \'[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}]\')'
        ),
    )
    parser.add_argument(
        "--guardrail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Google Cloud Model Armor 사용자 입력 보안 검사 활성화/비활성화 (--guardrail / --no-guardrail)",
    )
    parser.add_argument(
        "--model-armor-template",
        help="사용할 Model Armor Template ID (기본값: airline-chatbot-guardrail, 미존재 시 자동 생성)",
    )
    parser.add_argument(
        "--model-armor-location",
        help="Model Armor 템플릿 리전 (기본값: --location 또는 us-central1)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="바인딩할 호스트 주소 (기본값: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="바인딩할 포트 번호 (기본값: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="서버 시작 시 브라우저 자동 열기 비활성화",
    )
    return parser


def run_web_server(
    config: LiveConnectionConfig,
    *,
    guardrail_config: ModelArmorGuardrailConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    url = f"http://{host}:{port}"
    gr_cfg = guardrail_config or ModelArmorGuardrailConfig.from_env(
        fallback_project_id=config.project_id,
        fallback_location=config.location,
    )
    safety_summary = (
        f"{len(config.safety_settings)}개 카테고리 적용"
        if config.safety_settings
        else "DEFAULT (서버 기본값)"
    )
    gr_summary = (
        f"ON (Model Armor template={gr_cfg.template_id}, location={gr_cfg.location})"
        if gr_cfg.enabled
        else "OFF"
    )
    print("========================================================================")
    print("  SkyLine Airlines Live Graph Studio (HTML Web App)")
    print("========================================================================")
    print(f"  웹 대시보드 주소 : {url}")
    print(f"  Gemini 모델     : {config.model}")
    print(f"  연동 백엔드     : {_describe_backend(config)}")
    print(f"  초기 Safety 설정: {safety_summary}")
    print(f"  Model Armor     : {gr_summary}")
    print("========================================================================")

    app = create_web_app(config, default_guardrail_config=gr_cfg)

    if open_browser:
        timer = threading.Timer(0.7, webbrowser.open, args=[url])
        timer.daemon = True
        timer.start()

    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv()
    args = build_web_arg_parser().parse_args(argv)
    config = LiveConnectionConfig.from_env()
    if args.project:
        config.project_id = args.project.strip()
    if args.location:
        config.location = args.location.strip()
    if args.model:
        config.model = args.model.strip()
    if args.safety_settings_json is not None:
        config.safety_settings = build_safety_settings(
            custom_settings=args.safety_settings_json
        )
    elif args.safety_threshold:
        config.safety_settings = build_safety_settings(threshold=args.safety_threshold)

    gr_cfg = ModelArmorGuardrailConfig.from_env(
        fallback_project_id=config.project_id,
        fallback_location=config.location,
    )
    if args.guardrail is not None:
        gr_cfg.enabled = bool(args.guardrail)
    if args.model_armor_template:
        gr_cfg.template_id = args.model_armor_template.strip()
    if args.model_armor_location:
        gr_cfg.location = args.model_armor_location.strip()

    run_web_server(
        config,
        guardrail_config=gr_cfg,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
