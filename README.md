# Gemini 3.8 Live Airline Ticket Graph Chatbot

**Gemini 3.8 Live (`gemini-3.8-live`)** 모델을 **Raw WebSocket (`BidiGenerateContent`)** 으로 연결하고, **순수 Python State Graph**로 대화 컨텍스트를 제어하는 CLI 기반 항공사 티켓 챗봇 샘플입니다.

## 주요 특징

1. **3대 핵심 기능 + 언제든 초기 메뉴 복귀**
   - `1. 예약된 비행기 티켓 조회`: `user_id` + `booking_number` 검증 후 확정 티켓 안내
   - `2. 비행기 일정 조회`: 출발지/도착지/날짜 조건 또는 전체 운항 일정 조회
   - `3. 비행기 티켓 예약`: `user_id` + 사전 발급 `booking_number` 인증 후 항공편 선택 및 예약 확정
   - 어느 단계에서든 `"처음으로 돌아갈래"` 발화 시 즉시 `ROOT_MENU`로 복귀
2. **노드당 단 1개의 Enum 기반 의사결정 Tool**
   - 루트 노드(`ROOT_MENU`): `route_menu(decision: CHECK_TICKET | CHECK_SCHEDULE | BOOK_TICKET)` 단일 분기 툴
   - 하위 단계 노드: `decide_*(decision: EXECUTE | RETURN_TO_ROOT)` 단일 게이트 툴
   - 사용자 입력 값(`user_id`, `booking_number`, `flight_id` 등)은 `SlotExtractor`가 상태 슬롯에 추출하므로 모델이 문자열 인자를 생성하지 않아 응답 지연과 할루시네이션이 최소화됩니다.
3. **세션 중 동적 컨텍스트 업데이트 (`contextUpdate`)**
   - 그래프 노드가 전이될 때만 `BidiGenerateContentContextUpdate` 프레임을 전송해 WebSocket 재연결 없이 `systemInstruction`과 `tools`를 즉시 교체합니다.
4. **공식 시스템 지시사항 Best Practices 준수**
   - 노드별 시스템 지시사항을 `Persona -> Conversational Rules -> Guardrails` 순서로 구성하고 한국어 출력 강제(`RESPOND IN KOREAN...`) 및 구조화된 비공백 `DecisionResult` 응답을 적용했습니다.

## 설치 및 실행 방법

### 1. 의존성 설치 (`uv`)

```powershell
uv sync --all-extras
```

### 2. 인증 환경 변수 설정

#### 옵션 A: Google Cloud Vertex AI (기본 권장)
```powershell
gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
```

#### 옵션 B: Gemini Developer API
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
```

### 3. CLI 실행

```powershell
uv run airline-chat
```

## 테스트용 Mock 데이터 치트시트

| 구분 | `user_id` | `booking_number` | 설명 |
| --- | --- | --- | --- |
| 1번 조회용 (확정 완료) | `user01` | `BK-1001` | `KE101` (인천 -> 도쿄, 좌석 `14A`) |
| 1번 조회용 (확정 완료) | `user02` | `BK-1002` | `LJ303` (김포 -> 제주, 좌석 `7C`) |
| 3번 예약용 (미확정 코드) | `user01` | `BK-2001` | 항공편 선택 시 확정 처리 (이후 1번에서 조회 가능) |
| 3번 예약용 (미확정 코드) | `user02` | `BK-2002` | 항공편 선택 시 확정 처리 |

- **운항 항공편**: `KE101` (인천->도쿄), `OZ202` (인천->오사카), `LJ303` (김포->제주), `TW404` (인천->다낭)

## 자동화 테스트 실행

```powershell
uv run pytest -v
```
