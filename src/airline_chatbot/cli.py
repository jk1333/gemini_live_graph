"""Entrypoint for the Gemini 3.8 Live Airline Ticket Chatbot (HTML Web App default + CLI mode)."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import sys
from typing import Any

from dotenv import load_dotenv

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.guardrail import ModelArmorGuardrailChecker, ModelArmorGuardrailConfig
from airline_chatbot.live_client import (
    GeminiLiveWebSocketSession,
    LiveConnectionConfig,
    build_safety_settings,
)
from airline_chatbot.models import DecisionResult, NodeId, SafetyThresholdOption
from airline_chatbot.web import run_web_server

BANNER = """========================================================================
  SkyLine Airlines Chatbot (Gemini 3.8 Live WebSocket + State Graph)
========================================================================
지원 메뉴:
  1. 예약된 비행기 티켓 조회 (필수: user_id + booking_number)
  2. 비행기 일정 조회       (노선/일자 조건 또는 전체 조회)
  3. 비행기 티켓 예약       (필수: user_id + booking_number -> 편명 확정)
  * 어느 단계에서든 "처음으로 돌아갈래"라고 입력하면 초기 메뉴로 복귀합니다.

[테스트용 Mock 데이터 치트시트]
  - 기확정 예약 조회용 (1번):
      user01 / BK-1001 (KE101 인천->도쿄, 좌석 14A)
      user02 / BK-1002 (LJ303 김포->제주, 좌석 7C)
  - 신규 예약 확정용 (3번):
      user01 / BK-2001 (미확정 사전 예약코드)
      user02 / BK-2002 (미확정 사전 예약코드)
  - 운항 항공편 목록 (2번 & 3번):
      KE101 (인천->도쿄), OZ202 (인천->오사카), LJ303 (김포->제주), TW404 (인천->다낭)

로컬 보조 명령어: /state (상태 보기), /guardrail on|off, /reset (루트 복귀), /quit (종료)
========================================================================"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airline-chat",
        description="Gemini 3.8 Live WebSocket + State Graph Airline Ticket Chatbot (HTML Web App & CLI)",
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
        help="HTML 웹 앱 호스트 주소 (기본값: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTML 웹 앱 포트 번호 (기본값: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="웹 서버 시작 시 브라우저 자동 실행 비활성화",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="HTML 웹 앱 대신 터미널 텍스트 CLI 모드로 실행",
    )
    return parser


def resolve_cli_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, LiveConnectionConfig, ModelArmorGuardrailConfig]:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)
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

    return args, config, gr_cfg


async def run_cli(
    config: LiveConnectionConfig,
    guardrail_config: ModelArmorGuardrailConfig | None = None,
) -> int:
    graph = AirlineStateGraph()
    gr_cfg = guardrail_config or ModelArmorGuardrailConfig.from_env(
        fallback_project_id=config.project_id,
        fallback_location=config.location,
    )
    guardrail = ModelArmorGuardrailChecker(gr_cfg)

    printed_header_in_turn = False

    def on_text_chunk(chunk: str) -> None:
        nonlocal printed_header_in_turn
        if not printed_header_in_turn:
            sys.stdout.write("\nAssistant > ")
            printed_header_in_turn = True
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_tool_event(
        func_name: str,
        args: dict[str, Any],
        result: DecisionResult,
    ) -> None:
        nonlocal printed_header_in_turn
        printed_header_in_turn = False
        decision_val = args.get("decision", "")
        sys.stdout.write(
            f"\n  [ToolCall] {func_name}(decision={decision_val}) "
            f"-> status={result.status.value}, node={result.active_node.value}\n"
        )
        sys.stdout.flush()

    def on_context_update(next_node: NodeId) -> None:
        sys.stdout.write(
            f"\n  [ContextUpdate] Gemini 3.8 Live 컨텍스트 전환 완료 -> [{next_node.value}]"
        )
        sys.stdout.flush()

    def on_status(message: str) -> None:
        sys.stdout.write(f"\n  {message}\n")
        sys.stdout.flush()

    print(BANNER)
    backend_desc = (
        f"Custom ({config.ws_url_override})"
        if config.ws_url_override
        else (
            f"Vertex AI ({config.project_id} / {config.location})"
            if config.project_id
            else "Gemini Developer API"
        )
    )
    safety_desc = (
        f"{len(config.safety_settings)}개 카테고리 설정됨"
        if config.safety_settings
        else "DEFAULT (서버 기본값)"
    )
    gr_desc = f"ON ({guardrail.config.template_id})" if guardrail.config.enabled else "OFF"
    print(
        f"연결 중... 모델={config.model} | 백엔드={backend_desc} | "
        f"Safety={safety_desc} | Model Armor={gr_desc}"
    )

    try:
        async with GeminiLiveWebSocketSession(
            graph=graph,
            config=config,
            on_text_chunk=on_text_chunk,
            on_tool_event=on_tool_event,
            on_context_update=on_context_update,
            on_status=on_status,
        ) as session:
            print("연결 완료! 대화를 시작해 보세요.\n")

            while True:
                prompt_label = f"[{graph.current_node.value}] You > "
                try:
                    user_text = await asyncio.to_thread(input, prompt_label)
                except (EOFError, KeyboardInterrupt):
                    print("\n챗봇을 종료합니다.")
                    break

                stripped = user_text.strip()
                if not stripped:
                    continue

                lowered = stripped.lower()
                if lowered in {"/quit", "/exit", "exit", "quit"}:
                    print("챗봇을 종료합니다.")
                    break

                if lowered == "/state":
                    print(f"  현재 노드: {graph.current_node.value}")
                    print(f"  수집 슬롯: {graph.slots.to_dict()}")
                    print(f"  Model Armor: {'ON' if guardrail.config.enabled else 'OFF'}\n")
                    continue

                if lowered in {"/guardrail on", "/guardrail off"}:
                    turn_on = lowered.endswith("on")
                    guardrail.update_settings(enabled=turn_on)
                    print(f"  [Model Armor] 입력 보안 검사가 {'ON' if turn_on else 'OFF'} 되었습니다.\n")
                    continue

                if lowered == "/reset":
                    await session.send_manual_reset()
                    print("\n  수동으로 초기 메뉴(ROOT_MENU)로 복귀했습니다.\n")
                    continue

                check = await guardrail.check_user_input(stripped, graph.current_node)
                if check.enabled:
                    verdict_str = "PASS" if check.allowed else "BLOCK"
                    print(
                        f"  [Model Armor] {verdict_str} | state={check.filter_match_state} | "
                        f"category={check.category} | latency={check.latency_ms:.1f}ms"
                    )
                    if check.error:
                        print(f"  [Model Armor 참고] {check.error}")
                    if not check.allowed:
                        print(f"\nAssistant > {check.reason}\n")
                        continue

                printed_header_in_turn = False
                await session.send_user_turn(stripped)
                sys.stdout.write("\n\n")
                sys.stdout.flush()

    except Exception as exc:  # noqa: BLE001
        print(f"\n[오류] {exc}", file=sys.stderr)
        return 1
    finally:
        await guardrail.close()

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args, config, gr_cfg = resolve_cli_args(argv)
    if args.cli:
        raise SystemExit(asyncio.run(run_cli(config, gr_cfg)))

    run_web_server(
        config,
        guardrail_config=gr_cfg,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
