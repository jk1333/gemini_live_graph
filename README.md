# Gemini 3.8 Live Airline Ticket Graph Chatbot (HTML Web App & CLI)

**Gemini 3.8 Live (`gemini-3.8-live`)** 모델을 **Raw WebSocket (`BidiGenerateContent`)** 으로 연결하고, **순수 Python State Graph**로 대화 컨텍스트를 제어하는 항공사 티켓 챗봇 샘플입니다. 브라우저에서 대화와 **실시간 Graph State 전이 / 슬롯 추출 / `contextUpdate` / Mock DB 상태**를 한눈에 추적할 수 있는 HTML 웹 대시보드를 기본 제공합니다.

## 주요 특징

1. **실시간 Graph State 추적 HTML 대시보드**
   - **인터랙티브 SVG State Graph**: 현재 활성 노드(`ROOT_MENU`, `TICKET_LOOKUP`, `SCHEDULE_LOOKUP`, `BOOKING_AUTH`, `BOOKING_CONFIRM`)와 방금 통과한 분기/복귀 엣지를 실시간 하이라이트
   - **실시간 SessionSlots 모니터**: 사용자 입력에서 추출된 `user_id`, `booking_number`, `flight_id`, 출도착지, 날짜를 즉시 표시
   - **단일 Enum 의사결정 Tool & System Instruction 뷰어**: 각 노드를 클릭하여 해당 노드에 노출되는 단 1개의 Tool JSON과 `Persona -> Rules -> Guardrails` 시스템 프롬프트 확인 가능
   - **실시간 전이/프로토콜 로그 & Mock DB 테이블**: `ToolCall`, `BidiGenerateContentContextUpdate`, Prefix Cache 유지 여부 및 예약 확정 시 잔여석 감소를 실시간 반영
2. **3대 핵심 기능 + 언제든 초기 메뉴 복귀**
   - `1. 예약된 비행기 티켓 조회`: `user_id` + `booking_number` 검증 후 확정 티켓 안내
   - `2. 비행기 일정 조회`: 출발지/도착지/날짜 조건 또는 전체 운항 일정 조회
   - `3. 비행기 티켓 예약`: `user_id` + 사전 발급 `booking_number` 인증 후 항공편 선택 및 예약 확정
   - 어느 단계에서든 `"처음으로 돌아갈래"` 발화 또는 상단 `처음으로 리셋` 버튼으로 즉시 `ROOT_MENU` 복귀

## 실행 방법

### 1. HTML 웹 앱 실행 (기본 모드)

```powershell
# Vertex AI (Google Cloud ADC) 연동 실행 -> http://127.0.0.1:8000 자동 오픈
.\.venv\Scripts\airline-chat.exe --project sandbox-373102 --location us-central1

# 초기 Safety Filter 임계치 지정 실행 (DEFAULT | BLOCK_LOW_AND_ABOVE | BLOCK_MEDIUM_AND_ABOVE | BLOCK_ONLY_HIGH | BLOCK_NONE | OFF)
.\.venv\Scripts\airline-chat.exe --project sandbox-373102 --location us-central1 --safety-threshold BLOCK_MEDIUM_AND_ABOVE

# 카테고리별 개별 Safety Filter JSON 지정 예시
.\.venv\Scripts\airline-chat.exe --project sandbox-373102 --location us-central1 --safety-settings-json '[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}]'

# Google Cloud Model Armor 입력 보안 Guardrail 옵션 지정 예시
.\.venv\Scripts\airline-chat.exe --project sandbox-373102 --location us-central1 --guardrail --model-armor-template airline-chatbot-guardrail
```

*(웹 앱 실행 후 브라우저 우측 상단의 **`🛡️ Model Armor: ON/OFF`** 버튼으로 세션 재연결 없이 즉시 Guardrail을 켜고 끌 수 있으며, 매 입력마다 측정된 **Latency (`ms`)** 와 4대 보안 필터(`PI/Jailbreak · RAI · SDP · Malicious URI`) 결과가 실시간으로 표시됩니다.)*

`GEMINI_API_KEY`를 사용하는 경우:
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
.\.venv\Scripts\airline-chat.exe
```

### 2. 터미널 전용 CLI 모드로 실행하고 싶은 경우

```powershell
.\.venv\Scripts\airline-chat.exe --cli --project sandbox-373102 --location us-central1
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
.\.venv\Scripts\python.exe -m pytest -v
```
