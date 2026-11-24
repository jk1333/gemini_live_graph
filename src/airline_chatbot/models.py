"""Domain models and strict Enum schemas for the Airline State Graph Chatbot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeId(str, Enum):
    """Graph states representing distinct conversational stages."""

    ROOT_MENU = "ROOT_MENU"
    TICKET_LOOKUP = "TICKET_LOOKUP"
    SCHEDULE_LOOKUP = "SCHEDULE_LOOKUP"
    BOOKING_AUTH = "BOOKING_AUTH"
    BOOKING_CONFIRM = "BOOKING_CONFIRM"


class RootDecision(str, Enum):
    """Enum decision values for the single routing tool in ROOT_MENU."""

    CHECK_TICKET = "CHECK_TICKET"
    CHECK_SCHEDULE = "CHECK_SCHEDULE"
    BOOK_TICKET = "BOOK_TICKET"


class StepDecision(str, Enum):
    """Enum decision values for the single gate tool in all non-root nodes."""

    EXECUTE = "EXECUTE"
    RETURN_TO_ROOT = "RETURN_TO_ROOT"


class BookingStatus(str, Enum):
    """Lifecycle status of a mock booking record."""

    CONFIRMED = "CONFIRMED"
    PENDING_FLIGHT = "PENDING_FLIGHT"


class ToolResultStatus(str, Enum):
    """Enum status codes returned in structured tool responses."""

    DECISION_ROUTED = "DECISION_ROUTED"
    EXECUTED_OK = "EXECUTED_OK"
    MISSING_SLOTS = "MISSING_SLOTS"
    AUTH_INVALID = "AUTH_INVALID"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_CONFIRMED = "ALREADY_CONFIRMED"
    INVALID_DECISION = "INVALID_DECISION"
    RETURNED_TO_ROOT = "RETURNED_TO_ROOT"


class HarmCategoryOption(str, Enum):
    """Supported HarmCategory values for Gemini Live `setup.safetySettings`."""

    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    CIVIC_INTEGRITY = "HARM_CATEGORY_CIVIC_INTEGRITY"


class SafetyThresholdOption(str, Enum):
    """Supported HarmBlockThreshold values (`DEFAULT` omits `safetySettings`)."""

    DEFAULT = "DEFAULT"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"
    OFF = "OFF"


CORE_HARM_CATEGORIES: tuple[HarmCategoryOption, ...] = (
    HarmCategoryOption.HARASSMENT,
    HarmCategoryOption.HATE_SPEECH,
    HarmCategoryOption.SEXUALLY_EXPLICIT,
    HarmCategoryOption.DANGEROUS_CONTENT,
)


@dataclass(frozen=True)
class FlightSchedule:
    """Mock flight schedule item."""

    flight_id: str
    airline: str
    departure_city: str
    departure_code: str
    arrival_city: str
    arrival_code: str
    departure_time: str
    arrival_time: str
    price_krw: int
    seats_available: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BookingRecord:
    """In-memory booking/voucher record associated with a user_id."""

    booking_number: str
    user_id: str
    passenger_name: str
    status: BookingStatus
    flight_id: str | None = None
    seat: str | None = None
    booked_at: str | None = None

    def to_dict(self, flight: FlightSchedule | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "booking_number": self.booking_number,
            "user_id": self.user_id,
            "passenger_name": self.passenger_name,
            "status": self.status.value,
            "flight_id": self.flight_id,
            "seat": self.seat,
            "booked_at": self.booked_at,
        }
        if flight is not None:
            payload["flight"] = flight.to_dict()
        return payload


@dataclass
class SessionSlots:
    """Extracted user parameters maintained outside LLM tool arguments."""

    user_id: str | None = None
    booking_number: str | None = None
    flight_id: str | None = None
    departure_code: str | None = None
    arrival_code: str | None = None
    date: str | None = None
    query_all_schedules: bool = False

    def reset(self) -> None:
        self.user_id = None
        self.booking_number = None
        self.flight_id = None
        self.departure_code = None
        self.arrival_code = None
        self.date = None
        self.query_all_schedules = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    """Structured non-empty tool output following Gemini Live API best practices."""

    status: ToolResultStatus
    decision: str
    active_node: NodeId
    retryable: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_response_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "decision": self.decision,
            "active_node": self.active_node.value,
            "retryable": self.retryable,
            "message": self.message,
            "data": self.data,
        }
