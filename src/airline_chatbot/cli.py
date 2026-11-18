"""Interactive CLI for the Gemini 3.8 Live Airline Ticket Graph Chatbot."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import sys
from typing import Any

from dotenv import load_dotenv

from airline_chatbot.graph import AirlineStateGraph
from airline_chatbot.live_client import GeminiLiveWebSocketSession, LiveConnectionConfig
from airline_chatbot.models import DecisionResult, NodeId

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

로컬 보조 명령어: /state (현재 그래프 상태 보기), /reset (강제 루트 복귀), /quit (종료)
========================================================================"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airline-chat",
        description="Gemini 3.8 Live WebSocket + State Graph Airline Ticket Chatbot CLI",
    )
    parser.add_argument(
        "--project",
        help="Google Cloud Project ID (기본값: 환경변수 GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--location",
        help="Vertex AI 리전 (기본값: 환경변수 GOOGLE_CLOUD_LOCATION 또는 us-central1)",
    )
    parser.add_argument(
        "--model",
        help="사용할 Gemini Live 모델명 (기본값: 환경변수 GEMINI_LIVE_MODEL 또는 gemini-3.8-live)",
    )
    return parser


def resolve_cli_config(argv: Sequence[str] | None = None) -> LiveConnectionConfig:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)
    config = LiveConnectionConfig.from_env()
    if args.project:
        config.project_id = args.project.strip()
    if args.location:
        config.location = args.location.strip()
    if args.model:
        config.model = args.model.strip()
    return config


async def run_cli(config: LiveConnectionConfig) -> int:
    graph = AirlineStateGraph()

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
    print(f"연결 중... 모델={config.model} | 백엔드={backend_desc}")

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

                if stripped.lower() in {"/quit", "/exit", "exit", "quit"}:
                    print("챗봇을 종료합니다.")
                    break

                if stripped.lower() == "/state":
                    print(f"  현재 노드: {graph.current_node.value}")
                    print(f"  수집 슬롯: {graph.slots.to_dict()}\n")
                    continue

                if stripped.lower() == "/reset":
                    await session.send_manual_reset()
                    print("\n  수동으로 초기 메뉴(ROOT_MENU)로 복귀했습니다.\n")
                    continue

                printed_header_in_turn = False
                await session.send_user_turn(stripped)
                sys.stdout.write("\n\n")
                sys.stdout.flush()

    except Exception as exc:  # noqa: BLE001
        print(f"\n[오류] {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    config = resolve_cli_config(argv)
    raise SystemExit(asyncio.run(run_cli(config)))


if __name__ == "__main__":
    main()
