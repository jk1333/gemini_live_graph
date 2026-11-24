"""User input guardrail powered by Google Cloud Model Armor (`sanitizeUserPrompt`)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import os
import time
from typing import Any

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

from airline_chatbot.models import NodeId

DEFAULT_MODEL_ARMOR_TEMPLATE_ID = "airline-chatbot-guardrail"
DEFAULT_MODEL_ARMOR_LOCATION = "us-central1"

FILTER_REASON_KO: dict[str, str] = {
    "CLEAN": "모든 Model Armor 보안·안전 필터를 통과한 정상 입력입니다.",
    "PI_AND_JAILBREAK": "프롬프트 인젝션 또는 탈옥(Jailbreak) 시도가 Model Armor에서 감지되어 차단되었습니다.",
    "RAI:HATE_SPEECH": "혐오 표현(Hate Speech)이 감지되어 차단되었습니다.",
    "RAI:HARASSMENT": "괴롭힘·폭언(Harassment) 표현이 감지되어 차단되었습니다.",
    "RAI:DANGEROUS": "위험하거나 유해한 행위(Dangerous Content) 요청이 감지되어 차단되었습니다.",
    "RAI:SEXUALLY_EXPLICIT": "성적 노골성(Sexually Explicit) 콘텐츠가 감지되어 차단되었습니다.",
    "RAI": "Responsible AI 안전 정책 위반이 감지되어 차단되었습니다.",
    "SDP_SENSITIVE_DATA": "민감 개인·금융정보(SDP/PII)가 포함되어 차단되었습니다.",
    "MALICIOUS_URI": "악성 또는 피싱 의심 URL이 감지되어 차단되었습니다.",
    "CSAM": "불법 유해 콘텐츠가 감지되어 차단되었습니다.",
    "POLICY_VIOLATION": "Model Armor 보안 정책 위반이 감지되어 입력이 차단되었습니다.",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def build_default_template_body(
    *,
    include_sdp: bool = True,
    include_multilang: bool = True,
) -> dict[str, Any]:
    """Builds a standard Model Armor Template configuration for airline chatbot protection."""
    filter_config: dict[str, Any] = {
        "raiSettings": {
            "raiFilters": [
                {"filterType": "HATE_SPEECH", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                {"filterType": "HARASSMENT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                {"filterType": "DANGEROUS", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                {"filterType": "SEXUALLY_EXPLICIT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
            ]
        },
        "piAndJailbreakFilterSettings": {
            "filterEnforcement": "ENABLED",
            "confidenceLevel": "LOW_AND_ABOVE",
        },
        "maliciousUriFilterSettings": {
            "filterEnforcement": "ENABLED",
        },
    }
    if include_sdp:
        filter_config["sdpSettings"] = {
            "basicConfig": {
                "filterEnforcement": "ENABLED",
            }
        }
    body: dict[str, Any] = {"filterConfig": filter_config}
    if include_multilang:
        body["templateMetadata"] = {
            "multiLanguageDetection": {
                "enableMultiLanguageDetection": True,
            }
        }
    return body


@dataclass
class ModelArmorGuardrailConfig:
    """Runtime-mutable configuration for Google Cloud Model Armor `sanitizeUserPrompt`."""

    enabled: bool = True
    project_id: str | None = None
    location: str = DEFAULT_MODEL_ARMOR_LOCATION
    template_id: str = DEFAULT_MODEL_ARMOR_TEMPLATE_ID
    auto_create_template: bool = True
    endpoint_override: str | None = None
    timeout_sec: float = 10.0

    @classmethod
    def from_env(
        cls,
        *,
        fallback_project_id: str | None = None,
        fallback_location: str | None = None,
    ) -> ModelArmorGuardrailConfig:
        enabled = _env_bool("MODEL_ARMOR_ENABLED", True)
        project_id = (
            fallback_project_id
            or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
            or None
        )
        raw_loc = (
            os.getenv("MODEL_ARMOR_LOCATION", "").strip()
            or fallback_location
            or os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
            or DEFAULT_MODEL_ARMOR_LOCATION
        )
        # Model Armor uses regional endpoints (e.g., us-central1); map 'global' -> 'us-central1'
        location = DEFAULT_MODEL_ARMOR_LOCATION if raw_loc == "global" else raw_loc
        template_id = (
            os.getenv("MODEL_ARMOR_TEMPLATE_ID", "").strip()
            or DEFAULT_MODEL_ARMOR_TEMPLATE_ID
        )
        auto_create = _env_bool("MODEL_ARMOR_AUTO_CREATE_TEMPLATE", True)
        endpoint_override = os.getenv("MODEL_ARMOR_ENDPOINT_OVERRIDE", "").strip() or None
        return cls(
            enabled=enabled,
            project_id=project_id,
            location=location,
            template_id=template_id,
            auto_create_template=auto_create,
            endpoint_override=endpoint_override,
        )

    def base_api_url(self) -> str:
        if self.endpoint_override:
            return self.endpoint_override.rstrip("/")
        return f"https://modelarmor.{self.location}.rep.googleapis.com"

    def sanitize_prompt_url(self) -> str:
        if not self.project_id:
            raise RuntimeError(
                "Model Armor 사용을 위해 GOOGLE_CLOUD_PROJECT(--project) 설정이 필요합니다."
            )
        return (
            f"{self.base_api_url()}/v1/projects/{self.project_id}/"
            f"locations/{self.location}/templates/{self.template_id}:sanitizeUserPrompt"
        )

    def create_template_url(self) -> str:
        if not self.project_id:
            raise RuntimeError(
                "Model Armor 템플릿 생성을 위해 GOOGLE_CLOUD_PROJECT(--project) 설정이 필요합니다."
            )
        return (
            f"{self.base_api_url()}/v1/projects/{self.project_id}/"
            f"locations/{self.location}/templates"
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "provider": "model_armor",
            "enabled": self.enabled,
            "project_id": self.project_id,
            "location": self.location,
            "template_id": self.template_id,
            "auto_create_template": self.auto_create_template,
        }


@dataclass
class GuardrailCheckResult:
    """Structured outcome of a single Model Armor `sanitizeUserPrompt` inspection."""

    enabled: bool
    allowed: bool
    latency_ms: float
    filter_match_state: str = "NO_MATCH_FOUND"
    invocation_result: str = "SUCCESS"
    category: str = "CLEAN"
    matched_filters: list[str] = field(default_factory=list)
    filter_statuses: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    template_id: str = DEFAULT_MODEL_ARMOR_TEMPLATE_ID
    template_auto_created: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fetch_adc_bearer_token() -> str:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(GoogleAuthRequest())
    if not credentials.token:
        raise RuntimeError(
            "Google Cloud ADC 액세스 토큰을 가져오지 못했습니다 (`gcloud auth application-default login` 필요)."
        )
    return str(credentials.token)


class ModelArmorGuardrailChecker:
    """Evaluates user prompts via Google Cloud Model Armor (`sanitizeUserPrompt`)."""

    def __init__(
        self,
        config: ModelArmorGuardrailConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Any = _fetch_adc_bearer_token,
    ) -> None:
        self.config = config
        self._external_client = http_client is not None
        self._client = http_client
        self._token_provider = token_provider
        self._ensured_templates: set[str] = set()

    async def close(self) -> None:
        if self._client is not None and not self._external_client:
            await self._client.aclose()
            self._client = None

    def update_settings(
        self,
        *,
        enabled: bool | None = None,
        template_id: str | None = None,
        location: str | None = None,
        auto_create_template: bool | None = None,
    ) -> None:
        if enabled is not None:
            self.config.enabled = bool(enabled)
        if template_id is not None and template_id.strip():
            self.config.template_id = template_id.strip()
        if location is not None and location.strip():
            loc = location.strip()
            self.config.location = (
                DEFAULT_MODEL_ARMOR_LOCATION if loc == "global" else loc
            )
        if auto_create_template is not None:
            self.config.auto_create_template = bool(auto_create_template)

    @staticmethod
    def build_sanitize_payload(user_text: str) -> dict[str, Any]:
        return {
            "userPromptData": {
                "text": user_text,
            }
        }

    async def _resolve_headers(self) -> dict[str, str]:
        if not self.config.project_id:
            raise RuntimeError(
                "Model Armor 호출에는 Google Cloud 프로젝트 ID(--project 또는 GOOGLE_CLOUD_PROJECT)가 필요합니다."
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            token = self._token_provider()
        else:
            token = await asyncio.to_thread(self._token_provider)

        return {
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": self.config.project_id,
            "Content-Type": "application/json",
        }

    def parse_sanitization_response(
        self,
        body: dict[str, Any],
        *,
        latency_ms: float,
        template_auto_created: bool = False,
    ) -> GuardrailCheckResult:
        san_res = body.get("sanitizationResult") or body.get("sanitization_result") or {}
        overall_match = str(
            san_res.get("filterMatchState")
            or san_res.get("filter_match_state")
            or "NO_MATCH_FOUND"
        ).upper()
        invocation_res = str(
            san_res.get("invocationResult")
            or san_res.get("invocation_result")
            or "SUCCESS"
        ).upper()
        filter_results = san_res.get("filterResults") or san_res.get("filter_results") or {}

        matched_filters: list[str] = []
        filter_statuses: dict[str, str] = {
            "pi_and_jailbreak": "OFF",
            "rai": "OFF",
            "sdp": "OFF",
            "malicious_uris": "OFF",
        }

        # 1. Prompt Injection & Jailbreak
        pi_wrap = filter_results.get("pi_and_jailbreak") or {}
        pi_res = (
            pi_wrap.get("piAndJailbreakFilterResult")
            or pi_wrap.get("pi_and_jailbreak_filter_result")
            or {}
        )
        if pi_res:
            pi_match = str(pi_res.get("matchState") or pi_res.get("match_state") or "").upper()
            filter_statuses["pi_and_jailbreak"] = (
                "MATCH" if pi_match == "MATCH_FOUND" else "PASS"
            )
            if pi_match == "MATCH_FOUND":
                matched_filters.append("PI_AND_JAILBREAK")

        # 2. Responsible AI (HATE_SPEECH, HARASSMENT, DANGEROUS, SEXUALLY_EXPLICIT)
        rai_wrap = filter_results.get("rai") or {}
        rai_res = rai_wrap.get("raiFilterResult") or rai_wrap.get("rai_filter_result") or {}
        if rai_res:
            rai_match = str(rai_res.get("matchState") or rai_res.get("match_state") or "").upper()
            filter_statuses["rai"] = "MATCH" if rai_match == "MATCH_FOUND" else "PASS"
            if rai_match == "MATCH_FOUND":
                sub_results = (
                    rai_res.get("raiFilterTypeResults")
                    or rai_res.get("rai_filter_type_results")
                    or {}
                )
                sub_matched = False
                for sub_name, sub_obj in sub_results.items():
                    if isinstance(sub_obj, dict):
                        sm = str(
                            sub_obj.get("matchState") or sub_obj.get("match_state") or ""
                        ).upper()
                        if sm == "MATCH_FOUND":
                            matched_filters.append(f"RAI:{sub_name.upper()}")
                            sub_matched = True
                if not sub_matched:
                    matched_filters.append("RAI")

        # 3. Sensitive Data Protection (SDP / DLP)
        sdp_wrap = filter_results.get("sdp") or {}
        sdp_res = sdp_wrap.get("sdpFilterResult") or sdp_wrap.get("sdp_filter_result") or {}
        if sdp_res:
            inspect_res = (
                sdp_res.get("inspectResult")
                or sdp_res.get("inspect_result")
                or sdp_res.get("deidentifyResult")
                or {}
            )
            sdp_match = str(
                inspect_res.get("matchState") or inspect_res.get("match_state") or ""
            ).upper()
            filter_statuses["sdp"] = "MATCH" if sdp_match == "MATCH_FOUND" else "PASS"
            if sdp_match == "MATCH_FOUND":
                matched_filters.append("SDP_SENSITIVE_DATA")

        # 4. Malicious URIs
        uri_wrap = filter_results.get("malicious_uris") or {}
        uri_res = (
            uri_wrap.get("maliciousUriFilterResult")
            or uri_wrap.get("malicious_uri_filter_result")
            or {}
        )
        if uri_res:
            uri_match = str(uri_res.get("matchState") or uri_res.get("match_state") or "").upper()
            filter_statuses["malicious_uris"] = (
                "MATCH" if uri_match == "MATCH_FOUND" else "PASS"
            )
            if uri_match == "MATCH_FOUND":
                matched_filters.append("MALICIOUS_URI")

        # 5. CSAM
        csam_wrap = filter_results.get("csam") or {}
        csam_res = (
            csam_wrap.get("csamFilterFilterResult")
            or csam_wrap.get("csam_filter_filter_result")
            or {}
        )
        if csam_res:
            csam_match = str(
                csam_res.get("matchState") or csam_res.get("match_state") or ""
            ).upper()
            if csam_match == "MATCH_FOUND":
                matched_filters.append("CSAM")

        is_blocked = overall_match == "MATCH_FOUND" or bool(matched_filters)
        if is_blocked:
            primary_cat = matched_filters[0] if matched_filters else "POLICY_VIOLATION"
            reason = FILTER_REASON_KO.get(
                primary_cat,
                FILTER_REASON_KO["POLICY_VIOLATION"],
            )
        else:
            primary_cat = "CLEAN"
            reason = FILTER_REASON_KO["CLEAN"]

        return GuardrailCheckResult(
            enabled=True,
            allowed=not is_blocked,
            latency_ms=round(latency_ms, 1),
            filter_match_state="MATCH_FOUND" if is_blocked else overall_match,
            invocation_result=invocation_res,
            category=primary_cat,
            matched_filters=matched_filters,
            filter_statuses=filter_statuses,
            reason=reason,
            template_id=self.config.template_id,
            template_auto_created=template_auto_created,
        )

    def _template_cache_key(self) -> str:
        return f"{self.config.location}/{self.config.template_id}"

    @staticmethod
    def _is_template_not_found(response: httpx.Response) -> bool:
        """Detects missing Model Armor template on both HTTP 404 and HTTP 400 INVALID_ARGUMENT."""
        if response.status_code == 404:
            return True
        if response.status_code == 400:
            lower = response.text.lower()
            if "template" in lower and (
                "was not found" in lower
                or "not found" in lower
                or "does not exist" in lower
            ):
                return True
        return False

    async def _wait_if_lro(
        self,
        resp: httpx.Response,
        headers: dict[str, str],
    ) -> None:
        """If CreateTemplate returns a Long-Running Operation, polls briefly until done."""
        assert self._client is not None
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return
        if not isinstance(data, dict):
            return
        op_name = str(data.get("name") or "").strip("/")
        if "/operations/" not in op_name or data.get("done") is True:
            return

        op_url = f"{self.config.base_api_url()}/v1/{op_name}"
        for _ in range(10):
            await asyncio.sleep(0.4)
            poll_resp = await self._client.get(op_url, headers=headers)
            if poll_resp.status_code >= 400:
                return
            try:
                poll_data = poll_resp.json()
            except Exception:  # noqa: BLE001
                return
            if isinstance(poll_data, dict) and poll_data.get("done") is True:
                return

    async def _auto_provision_template(
        self,
        headers: dict[str, str],
    ) -> tuple[bool, str | None]:
        """Creates the default Model Armor template if it does not yet exist."""
        assert self._client is not None
        create_url = self.config.create_template_url()
        params = {"template_id": self.config.template_id}

        candidates = [
            build_default_template_body(include_sdp=True, include_multilang=True),
            build_default_template_body(include_sdp=False, include_multilang=True),
            build_default_template_body(include_sdp=False, include_multilang=False),
        ]
        last_err: str | None = None
        for body in candidates:
            resp = await self._client.post(
                create_url,
                params=params,
                headers=headers,
                json=body,
            )
            if resp.status_code in {200, 201, 409}:
                if resp.status_code in {200, 201}:
                    await self._wait_if_lro(resp, headers)
                self._ensured_templates.add(self._template_cache_key())
                return True, None
            last_err = f"HTTP {resp.status_code}: {resp.text[:220]}"
            if resp.status_code in {401, 403}:
                # Permission/auth failure will not be resolved by payload fallbacks
                break

        return False, last_err

    async def check_user_input(
        self,
        user_text: str,
        active_node: NodeId | None = None,
    ) -> GuardrailCheckResult:
        """Evaluates `user_text` via Model Armor `sanitizeUserPrompt`, measuring latency."""
        del active_node  # Model Armor evaluates prompt against configured GCP template
        if not self.config.enabled:
            return GuardrailCheckResult(
                enabled=False,
                allowed=True,
                latency_ms=0.0,
                filter_match_state="SKIPPED",
                category="SKIPPED",
                reason="Model Armor Guardrail 비활성화 상태 (OFF)",
                template_id=self.config.template_id,
            )

        try:
            headers = await self._resolve_headers()
            sanitize_url = self.config.sanitize_prompt_url()
        except Exception as cfg_exc:  # noqa: BLE001
            return GuardrailCheckResult(
                enabled=True,
                allowed=True,
                latency_ms=0.0,
                filter_match_state="ERROR_OPEN",
                category="ERROR_OPEN",
                reason=str(cfg_exc),
                template_id=self.config.template_id,
                error=str(cfg_exc),
            )

        payload = self.build_sanitize_payload(user_text)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_sec)

        auto_created = False
        create_err: str | None = None
        t0 = time.perf_counter()
        try:
            response = await self._client.post(
                sanitize_url,
                headers=headers,
                json=payload,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Note: Model Armor `:sanitizeUserPrompt` returns HTTP 400 INVALID_ARGUMENT
            # ("The template named ... was not found") or HTTP 404 when a template is missing.
            if (
                self._is_template_not_found(response)
                and self.config.auto_create_template
                and self._template_cache_key() not in self._ensured_templates
            ):
                created_ok, create_err = await self._auto_provision_template(headers)
                if created_ok:
                    auto_created = True
                    for attempt in range(3):
                        t_retry = time.perf_counter()
                        response = await self._client.post(
                            sanitize_url,
                            headers=headers,
                            json=payload,
                        )
                        latency_ms = (time.perf_counter() - t_retry) * 1000.0
                        if not self._is_template_not_found(response):
                            break
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return GuardrailCheckResult(
                enabled=True,
                allowed=True,
                latency_ms=round(latency_ms, 1),
                filter_match_state="ERROR_OPEN",
                category="ERROR_OPEN",
                reason=f"Model Armor 네트워크 오류로 검사를 건너뛰었습니다: {exc}",
                template_id=self.config.template_id,
                error=str(exc),
            )

        if response.status_code >= 400:
            raw_text = response.text
            lower = raw_text.lower()
            if response.status_code == 403 and (
                "modelarmor.googleapis.com" in lower or "has not been used" in lower or "disabled" in lower
            ):
                friendly_err = (
                    f"HTTP 403 (Model Armor API 미활성화): 터미널에서 "
                    f"`gcloud services enable modelarmor.googleapis.com --project={self.config.project_id}` "
                    "명령으로 API를 활성화해 주세요."
                )
            elif self._is_template_not_found(response):
                if create_err:
                    friendly_err = (
                        f"템플릿 '{self.config.template_id}'({self.config.location})이 존재하지 않으며 "
                        f"자동 생성에도 실패했습니다 ({create_err}). "
                        "IAM 권한(`roles/modelarmor.admin`)을 확인하거나 상단 ⚙️ 설정에서 기존 Template ID를 지정해 주세요."
                    )
                else:
                    friendly_err = (
                        f"템플릿 '{self.config.template_id}'({self.config.location})을 찾을 수 없습니다. "
                        "상단 ⚙️ 설정에서 존재하는 Template ID를 입력해 주세요."
                    )
            else:
                friendly_err = f"HTTP {response.status_code}: {raw_text[:280]}"

            return GuardrailCheckResult(
                enabled=True,
                allowed=True,
                latency_ms=round(latency_ms, 1),
                filter_match_state="ERROR_OPEN",
                category="ERROR_OPEN",
                reason=f"Model Armor API 응답 오류 (HTTP {response.status_code})로 검사를 건너뛰었습니다.",
                template_id=self.config.template_id,
                error=friendly_err,
            )

        body = response.json()
        return self.parse_sanitization_response(
            body,
            latency_ms=latency_ms,
            template_auto_created=auto_created,
        )
