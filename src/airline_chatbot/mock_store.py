"""In-memory mock airline data store and deterministic slot extractor."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import re
from typing import Any

from airline_chatbot.models import (
    BookingRecord,
    BookingStatus,
    FlightSchedule,
    NodeId,
    SessionSlots,
    ToolResultStatus,
)


class MockAirlineStore:
    """In-memory mock store for flights, pre-issued booking codes, and confirmed tickets."""

    def __init__(self) -> None:
        self._flights: dict[str, FlightSchedule] = {
            "KE101": FlightSchedule(
                flight_id="KE101",
                airline="대한항공 (Korean Air)",
                departure_city="서울/인천",
                departure_code="ICN",
                arrival_city="도쿄/나리타",
                arrival_code="NRT",
                departure_time="2026-12-01 09:30",
                arrival_time="2026-12-01 11:55",
                price_krw=385000,
                seats_available=12,
            ),
            "OZ202": FlightSchedule(
                flight_id="OZ202",
                airline="아시아나항공 (Asiana)",
                departure_city="서울/인천",
                departure_code="ICN",
                arrival_city="오사카/간사이",
                arrival_code="KIX",
                departure_time="2026-12-01 13:15",
                arrival_time="2026-12-01 15:05",
                price_krw=320000,
                seats_available=8,
            ),
            "LJ303": FlightSchedule(
                flight_id="LJ303",
                airline="진에어 (Jin Air)",
                departure_city="서울/김포",
                departure_code="GMP",
                arrival_city="제주",
                arrival_code="CJU",
                departure_time="2026-12-02 08:00",
                arrival_time="2026-12-02 09:10",
                price_krw=89000,
                seats_available=25,
            ),
            "TW404": FlightSchedule(
                flight_id="TW404",
                airline="티웨이항공 (T'way Air)",
                departure_city="서울/인천",
                departure_code="ICN",
                arrival_city="다낭",
                arrival_code="DAD",
                departure_time="2026-12-03 19:40",
                arrival_time="2026-12-03 22:35",
                price_krw=410000,
                seats_available=5,
            ),
        }

        self._bookings: dict[str, BookingRecord] = {
            "BK-1001": BookingRecord(
                booking_number="BK-1001",
                user_id="user01",
                passenger_name="김민수",
                status=BookingStatus.CONFIRMED,
                flight_id="KE101",
                seat="14A",
                booked_at="2026-11-10 14:20",
            ),
            "BK-1002": BookingRecord(
                booking_number="BK-1002",
                user_id="user02",
                passenger_name="이지은",
                status=BookingStatus.CONFIRMED,
                flight_id="LJ303",
                seat="7C",
                booked_at="2026-11-12 10:05",
            ),
            "BK-2001": BookingRecord(
                booking_number="BK-2001",
                user_id="user01",
                passenger_name="김민수",
                status=BookingStatus.PENDING_FLIGHT,
            ),
            "BK-2002": BookingRecord(
                booking_number="BK-2002",
                user_id="user02",
                passenger_name="이지은",
                status=BookingStatus.PENDING_FLIGHT,
            ),
        }

    def list_all_flights(self) -> list[dict[str, Any]]:
        return [flight.to_dict() for flight in self._flights.values()]

    def lookup_booked_ticket(
        self,
        user_id: str,
        booking_number: str,
    ) -> tuple[ToolResultStatus, str, dict[str, Any]]:
        """1. 예약된 비행기 티켓 조회 (user_id + booking_number 검증)."""
        normalized_bn = booking_number.strip().upper()
        normalized_uid = user_id.strip().lower()

        record = self._bookings.get(normalized_bn)
        if record is None:
            return (
                ToolResultStatus.NOT_FOUND,
                f"예약 번호 '{normalized_bn}'에 해당하는 내역을 찾을 수 없습니다. 번호를 다시 확인해 주세요.",
                {"user_id": normalized_uid, "booking_number": normalized_bn},
            )

        if record.user_id.lower() != normalized_uid:
            return (
                ToolResultStatus.AUTH_INVALID,
                f"사용자 ID '{normalized_uid}'와 예약 번호 '{normalized_bn}'의 명의가 일치하지 않습니다.",
                {"user_id": normalized_uid, "booking_number": normalized_bn},
            )

        if record.status != BookingStatus.CONFIRMED or not record.flight_id:
            return (
                ToolResultStatus.NOT_FOUND,
                (
                    f"예약 번호 '{normalized_bn}'는 아직 항공편이 확정되지 않은 사전 예약 코드입니다. "
                    "항공편 예약을 원하시면 초기 메뉴의 '3. 비행기 티켓 예약'을 이용해 주세요."
                ),
                {"booking": record.to_dict()},
            )

        flight = self._flights.get(record.flight_id)
        return (
            ToolResultStatus.EXECUTED_OK,
            "예약된 비행기 티켓 조회에 성공했습니다. 아래 티켓 상세 정보를 고객에게 안내한 뒤 추가로 도울 일이 있는지 물어보세요.",
            {"ticket": record.to_dict(flight=flight)},
        )

    def search_schedules(
        self,
        departure_code: str | None = None,
        arrival_code: str | None = None,
        date: str | None = None,
    ) -> tuple[ToolResultStatus, str, dict[str, Any]]:
        """2. 비행기 일정 조회."""
        results: list[FlightSchedule] = []
        for flight in self._flights.values():
            if departure_code:
                if departure_code == "SEL":
                    if flight.departure_code not in {"ICN", "GMP"}:
                        continue
                elif flight.departure_code != departure_code:
                    continue

            if arrival_code:
                if arrival_code == "SEL":
                    if flight.arrival_code not in {"ICN", "GMP"}:
                        continue
                elif flight.arrival_code != arrival_code:
                    continue

            if date and not flight.departure_time.startswith(date):
                continue

            results.append(flight)

        filters = {
            "departure_code": departure_code,
            "arrival_code": arrival_code,
            "date": date,
        }
        if not results:
            return (
                ToolResultStatus.NOT_FOUND,
                "요청하신 조건에 일치하는 운항 일정이 없습니다. 전체 운항 일정을 참고용으로 함께 전달하니 안내해 주세요.",
                {
                    "applied_filters": filters,
                    "matched_flights": [],
                    "all_available_flights": self.list_all_flights(),
                },
            )

        return (
            ToolResultStatus.EXECUTED_OK,
            f"총 {len(results)}건의 비행기 운항 일정을 조회했습니다. 고객에게 편명, 출도착 시각, 운임, 잔여석을 친절히 안내하세요.",
            {
                "applied_filters": filters,
                "matched_flights": [f.to_dict() for f in results],
            },
        )

    def verify_booking_auth(
        self,
        user_id: str,
        booking_number: str,
    ) -> tuple[ToolResultStatus, str, dict[str, Any]]:
        """3-A. 비행기 티켓 예약 전 사용자 ID 및 예약 번호 검증."""
        normalized_bn = booking_number.strip().upper()
        normalized_uid = user_id.strip().lower()

        record = self._bookings.get(normalized_bn)
        if record is None:
            return (
                ToolResultStatus.NOT_FOUND,
                f"예약 번호 '{normalized_bn}'가 존재하지 않습니다. 올바른 사전 예약 번호(예: BK-2001)를 다시 요청하세요.",
                {"user_id": normalized_uid, "booking_number": normalized_bn},
            )

        if record.user_id.lower() != normalized_uid:
            return (
                ToolResultStatus.AUTH_INVALID,
                f"사용자 ID '{normalized_uid}'와 예약 번호 '{normalized_bn}' 정보가 일치하지 않습니다. 다시 확인해 달라고 안내하세요.",
                {"user_id": normalized_uid, "booking_number": normalized_bn},
            )

        if record.status == BookingStatus.CONFIRMED:
            flight = self._flights.get(record.flight_id or "")
            return (
                ToolResultStatus.ALREADY_CONFIRMED,
                (
                    f"예약 번호 '{normalized_bn}'는 이미 '{record.flight_id}' 항공편으로 예약이 완료된 번호입니다. "
                    "미확정 예약 번호(예: BK-2001, BK-2002)를 입력하거나 처음으로 돌아갈지 물어보세요."
                ),
                {"existing_ticket": record.to_dict(flight=flight)},
            )

        return (
            ToolResultStatus.EXECUTED_OK,
            (
                f"{record.passenger_name}({normalized_uid})님의 예약 번호 '{normalized_bn}' 인증이 완료되었습니다. "
                "예약 가능한 항공편 목록을 안내하고 어떤 항공편(편명, 예: KE101, OZ202)으로 확정할지 질문하세요."
            ),
            {
                "verified_booking": record.to_dict(),
                "available_flights": self.list_all_flights(),
            },
        )

    def confirm_booking(
        self,
        user_id: str,
        booking_number: str,
        flight_id: str,
    ) -> tuple[ToolResultStatus, str, dict[str, Any]]:
        """3-B. 선택한 항공편으로 티켓 예약 확정."""
        normalized_bn = booking_number.strip().upper()
        normalized_uid = user_id.strip().lower()
        normalized_fid = flight_id.strip().upper().replace("-", "")

        record = self._bookings.get(normalized_bn)
        if record is None or record.user_id.lower() != normalized_uid:
            return (
                ToolResultStatus.AUTH_INVALID,
                "인증된 사용자 ID 또는 예약 번호 정보를 확인할 수 없습니다.",
                {"user_id": normalized_uid, "booking_number": normalized_bn},
            )

        flight = self._flights.get(normalized_fid)
        if flight is None:
            return (
                ToolResultStatus.NOT_FOUND,
                f"선택하신 편명 '{normalized_fid}'는 존재하지 않는 항공편입니다. 가능한 편명({', '.join(self._flights.keys())}) 중 하나를 다시 선택해 달라고 안내하세요.",
                {
                    "requested_flight_id": normalized_fid,
                    "available_flights": self.list_all_flights(),
                },
            )

        if flight.seats_available <= 0:
            return (
                ToolResultStatus.NOT_FOUND,
                f"'{normalized_fid}' 항공편은 매진되었습니다. 다른 항공편을 선택해 주세요.",
                {"available_flights": self.list_all_flights()},
            )

        assigned_seat = f"{15 + (25 - flight.seats_available) % 15}B"
        updated_flight = replace(flight, seats_available=flight.seats_available - 1)
        self._flights[normalized_fid] = updated_flight

        record.status = BookingStatus.CONFIRMED
        record.flight_id = normalized_fid
        record.seat = assigned_seat
        record.booked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            ToolResultStatus.EXECUTED_OK,
            "항공권 예약이 최종 확정되었습니다! 확정된 편명, 출도착 정보, 배정 좌석을 안내하고 추가로 도와드릴 업무가 있는지 물어보세요.",
            {"confirmed_ticket": record.to_dict(flight=updated_flight)},
        )


class SlotExtractor:
    """Extracts structured slots from user CLI text so tools only need Enum decisions."""

    _BOOKING_EXPLICIT_RE = re.compile(
        r"(?:예약\s*번호|예약\s*코드|booking\s*(?:number|no|id|code)?)\s*(?:는|은|이|가|:|=)?\s*([A-Za-z]{1,4}-?\d{3,6}|[A-Za-z0-9-]{5,20})",
        re.IGNORECASE,
    )
    _BOOKING_CODE_RE = re.compile(r"\b(BK-?\d{3,6})\b", re.IGNORECASE)

    _FLIGHT_CODE_RE = re.compile(r"\b(KE|OZ|LJ|TW|7C|BX)\s*-?\s*(\d{3,4})\b", re.IGNORECASE)

    _USER_EXPLICIT_RE = re.compile(
        r"(?:사용자\s*(?:아이디|id)|아이디|유저\s*id|user\s*id|\bid)\s*(?:는|은|이|가|:|=)?\s*([A-Za-z][A-Za-z0-9_-]{2,24})",
        re.IGNORECASE,
    )
    _USER_TOKEN_RE = re.compile(r"\b((?:user|guest|member|tester)\d{1,6})\b", re.IGNORECASE)
    _BARE_ALNUM_ID_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]{2,20})\s*$")

    _DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
    _KR_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")

    _RESERVED_WORDS = {
        "exit",
        "quit",
        "reset",
        "menu",
        "home",
        "root",
        "help",
        "yes",
        "no",
        "icn",
        "gmp",
        "nrt",
        "kix",
        "cju",
        "dad",
    }

    _CITY_ALIASES: dict[str, tuple[str, ...]] = {
        "ICN": ("인천", "나리타발 인천", "icn"),
        "GMP": ("김포", "gmp"),
        "SEL": ("서울",),
        "NRT": ("도쿄", "나리타", "동경", "tokyo", "narita", "nrt"),
        "KIX": ("오사카", "간사이", "osaka", "kansai", "kix"),
        "CJU": ("제주", "제주도", "jeju", "cju"),
        "DAD": ("다낭", "danang", "dad"),
    }

    # Fallback destination-to-flight mapping when user chooses by destination name in BOOKING_CONFIRM
    _ARRIVAL_TO_FLIGHT: dict[str, str] = {
        "NRT": "KE101",
        "KIX": "OZ202",
        "CJU": "LJ303",
        "DAD": "TW404",
    }

    @classmethod
    def extract_into(cls, text: str, slots: SessionSlots, current_node: NodeId) -> None:
        """Mutates `slots` in-place with any entities detected in `text`."""
        cleaned = text.strip()
        if not cleaned:
            return

        # 1. Booking Number (e.g., BK-1001, bk1001, or explicit "예약번호 XXX")
        bk_code_match = cls._BOOKING_CODE_RE.search(cleaned)
        if bk_code_match:
            raw_bk = bk_code_match.group(1).upper()
            if not raw_bk.startswith("BK-"):
                raw_bk = "BK-" + raw_bk[2:]
            slots.booking_number = raw_bk
        else:
            bk_exp_match = cls._BOOKING_EXPLICIT_RE.search(cleaned)
            if bk_exp_match:
                candidate = bk_exp_match.group(1).strip().upper()
                # Strip trailing Korean particles if attached (e.g., BK-1001이야)
                candidate = re.sub(r"[가-힣]+$", "", candidate)
                if re.match(r"^BK\d+$", candidate):
                    candidate = "BK-" + candidate[2:]
                if candidate:
                    slots.booking_number = candidate

        # 2. Flight ID (e.g., KE101, OZ-202)
        flight_match = cls._FLIGHT_CODE_RE.search(cleaned)
        if flight_match:
            slots.flight_id = f"{flight_match.group(1).upper()}{flight_match.group(2)}"

        # 3. User ID
        user_exp_match = cls._USER_EXPLICIT_RE.search(cleaned)
        if user_exp_match:
            candidate_uid = re.sub(r"[가-힣]+$", "", user_exp_match.group(1))
            if (
                candidate_uid
                and not cls._BOOKING_CODE_RE.match(candidate_uid)
                and not cls._FLIGHT_CODE_RE.match(candidate_uid)
            ):
                slots.user_id = candidate_uid.lower()
        else:
            user_tok_match = cls._USER_TOKEN_RE.search(cleaned)
            if user_tok_match:
                slots.user_id = user_tok_match.group(1).lower()
            elif current_node in {NodeId.TICKET_LOOKUP, NodeId.BOOKING_AUTH}:
                # Support bare user_id answer like "wronguser" or "kim123" when prompted
                bare_match = cls._BARE_ALNUM_ID_RE.match(cleaned)
                if bare_match:
                    token = bare_match.group(1)
                    if (
                        token.lower() not in cls._RESERVED_WORDS
                        and not token.upper().startswith("BK-")
                        and not re.match(r"^BK\d+$", token.upper())
                        and not cls._FLIGHT_CODE_RE.match(token)
                    ):
                        slots.user_id = token.lower()

        # 4. Date (YYYY-MM-DD or M월 D일)
        date_match = cls._DATE_RE.search(cleaned)
        if date_match:
            slots.date = date_match.group(1)
        else:
            md_match = cls._KR_MONTH_DAY_RE.search(cleaned)
            if md_match:
                month = int(md_match.group(1))
                day = int(md_match.group(2))
                slots.date = f"2026-{month:02d}-{day:02d}"

        # 5. Route / City filters ("전체" resets filters)
        lowered = cleaned.lower()
        if any(kw in lowered for kw in ("전체", "모두", "전부", "다 보여", "모든 일정", "all")):
            slots.query_all_schedules = True
            slots.departure_code = None
            slots.arrival_code = None

        detected_cities: list[tuple[int, str]] = []
        for code, aliases in cls._CITY_ALIASES.items():
            for alias in aliases:
                idx = lowered.find(alias)
                if idx != -1:
                    detected_cities.append((idx, code))
                    break

        detected_cities.sort(key=lambda item: item[0])
        if len(detected_cities) >= 2:
            slots.departure_code = detected_cities[0][1]
            slots.arrival_code = detected_cities[1][1]
        elif len(detected_cities) == 1:
            only_code = detected_cities[0][1]
            if any(
                marker in lowered
                for marker in ("출발", "에서 출발")
            ) and only_code in {"ICN", "GMP", "SEL"}:
                slots.departure_code = only_code
            elif only_code in {"NRT", "KIX", "CJU", "DAD"} or any(
                marker in lowered for marker in ("행", "가는", "도착", "갈래", "가고")
            ):
                slots.arrival_code = only_code
            else:
                slots.departure_code = only_code

        # In BOOKING_CONFIRM, if user said e.g. "오사카행 비행기로 할게" without flight code, map destination -> flight_id
        if (
            current_node == NodeId.BOOKING_CONFIRM
            and slots.flight_id is None
            and slots.arrival_code in cls._ARRIVAL_TO_FLIGHT
        ):
            slots.flight_id = cls._ARRIVAL_TO_FLIGHT[slots.arrival_code]
