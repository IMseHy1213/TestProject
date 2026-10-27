"""Real Estate Recommendation Agent - 오케스트레이터.

에이전트 루프
  1. 관찰(Observe)  : 매물 폴더·플랫폼 내보내기 파일·공공데이터를 수집 (sync)
  2. 정제(Clean)    : 오류 매물 제외, 중복 매물 병합
  3. 이해(Reason)   : 손님 문의를 LLM 으로 구조화 (실패 시 규칙 기반)
  4. 계획/실행(Act) : 점수화·순위화, 후보가 부족하면 스스로 조건 완화
  5. 반성(Reflect)  : 손님 피드백("더 싼 거")을 조건에 반영해 재순위, 거절 매물 기억
  6. 선제 제안      : 새 매물이 들어오면 저장된 손님 조건과 대조해 먼저 알림 (watch)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from .cleaner import CleanResult, clean
from .collectors import LocalFolderCollector
from .collectors.public_data import LandTradePriceClient, VWorldGeocoder
from .config import DEFAULT_DATA_DIR, Settings
from .listing_editor import apply_edits
from .llm import LLMRouter, LLMUnavailable
from .models import CustomerRequirement, LandListing, ScoredListing, format_price
from .recommender import RecommendationSession, RecommendOptions, score_listing, hard_filter
from .requirement_parser import RequirementParser
from .preference import PreferenceModel
from .storage import OFFICE_MODEL, Storage

log = logging.getLogger(__name__)

EXPLAIN_SYSTEM = """당신은 토지 전문 공인중개사의 상담 보조 AI 입니다.
손님 조건과 추천 매물 정보를 보고, 손님에게 보여줄 2~4문장의 친절한 한국어 추천 설명을 쓰세요.
- 조건과 맞는 점을 먼저, 확인이 필요한 점(전용허가, 맹지, 예산 초과 등)을 솔직하게 덧붙이세요.
- 주어진 정보에 없는 사실(개발 호재, 시세 등)을 지어내지 마세요.
- 소유자 연락처 같은 개인정보는 쓰지 마세요."""


@dataclass
class SyncReport:
    clean: CleanResult
    new: list[LandListing] = field(default_factory=list)
    changed: list[LandListing] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)
    files_read: int = 0
    folder_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"파일 {self.files_read}개 읽음 | {self.clean.report().splitlines()[0]}\n"
            f"신규 {len(self.new)}건, 변경 {len(self.changed)}건, 사라진 매물 {len(self.removed_ids)}건"
            + ("\n폴더별 매물: " + ", ".join(f"{k} {v}건" for k, v in self.folder_counts.items())
               if self.folder_counts else "")
            + ("".join(f"\n  [FILE] {e}" for e in self.file_errors))
        )


@dataclass
class ProactiveAlert:
    customer: str
    scored: ScoredListing
    message: str


class RealEstateAgent:
    def __init__(self, settings: Settings, storage: Optional[Storage] = None, llm: Optional[LLMRouter] = None):
        self.settings = settings
        self.storage = storage or Storage(settings.db_path)
        self.llm = llm if llm is not None else LLMRouter(settings.llm_providers, settings.anthropic_model,
                                                         settings.openai_model)
        self.collector = LocalFolderCollector(settings.data_dir)
        self.market_client = LandTradePriceClient(settings.data_go_kr_key)
        self.geocoder = VWorldGeocoder(settings.vworld_key)
        self.market_price: dict[str, dict[str, float]] = {}
        self._llm_built_at = time.time()
        self._listings: Optional[list[LandListing]] = None
        self._office_model: Optional[PreferenceModel] = None
        self.hidden_listings: list[LandListing] = []  # 관리자가 목록에서 숨긴 매물
        self._originals: dict[str, LandListing] = {}  # 관리자가 고치기 전 원본 (되돌리기용)
        self.last_sync: Optional[SyncReport] = None
        self.last_sync_at: Optional[datetime] = None

    # ------------------------------------------------------------------ 1~2. 수집·정제
    def sync(self, today: Optional[date] = None) -> SyncReport:
        if self.settings.data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
            from .template import prepare_data_folder

            prepare_data_folder(self.settings.data_dir)  # 프로젝트 '매물' 폴더는 없으면 자동으로 만듦
        records = self.collector.collect()
        result = clean(records, today)
        # 관리자 화면에서 고친 내용 덮어쓰기 (원본 엑셀은 그대로)
        result.listings, self.hidden_listings, self._originals = apply_edits(result.listings,
                                                                             self.storage.listing_edits())
        self._enrich(result.listings)
        diff = self.storage.upsert_listings(result.listings)
        self._listings = result.listings
        report = SyncReport(clean=result, new=diff["new"], changed=diff["changed"], removed_ids=diff["removed_ids"],
                            file_errors=list(self.collector.errors), files_read=len(self.collector.files_read),
                            folder_counts=self.collector.folder_counts(result.listings))
        self._write_quality_log(report)
        self.last_sync, self.last_sync_at = report, datetime.now()
        return report

    def _enrich(self, listings: list[LandListing]) -> None:
        """공공데이터/GIS 로 보강 (API 키가 있을 때만)."""
        if self.geocoder.enabled:
            for l in listings:
                if l.lat is None:
                    pt = self.geocoder.geocode(l.address)
                    if pt:
                        l.lat, l.lon = pt
        lawd = os.getenv("REA_LAWD_CODES", "")  # 예: "양평군:41830,여주시:41670"
        if self.market_client.enabled and lawd and not self.market_price:
            today = date.today()
            months = []
            y, m = today.year, today.month
            for _ in range(6):
                m -= 1
                if m == 0:
                    y, m = y - 1, 12
                months.append(f"{y}{m:02d}")
            for pair in lawd.split(","):
                if ":" not in pair:
                    continue
                name, code = (x.strip() for x in pair.split(":", 1))
                prices = self.market_client.market_price_per_m2(code, months)
                if prices:
                    self.market_price[name] = prices

    def listings(self) -> list[LandListing]:
        if self._listings is None:
            self._listings = self.storage.active_listings()
        return self._listings

    # ------------------------------------------------------------------ 관리자: 매물 추가·수정
    def find_listing(self, listing_id: str) -> Optional[LandListing]:
        return next((l for l in self.listings() + self.hidden_listings if l.listing_id == listing_id), None)

    def original_listing(self, listing_id: str) -> Optional[LandListing]:
        """관리자가 고치기 전 원본 (고친 적 없으면 지금 매물)."""
        return self._originals.get(listing_id) or self.find_listing(listing_id)

    def listing_edit(self, listing_id: str) -> dict:
        return self.storage.listing_edits().get(listing_id, {})

    def check_listing(self, candidate: LandListing, ignore_id: str = "") -> tuple[list[str], list[str]]:
        """저장 전 검사: (오류, 경고). 오류가 있으면 저장하지 않음."""
        from .cleaner import is_duplicate, validate

        errors, warnings = validate(copy.deepcopy(candidate))
        for other in self.listings() + self.hidden_listings:
            if other.listing_id != ignore_id and other.listing_id != candidate.listing_id \
                    and is_duplicate(candidate, other):
                errors.append(f"이미 같은 매물이 있습니다 (매물번호 {other.listing_id}). 그 매물을 수정해 주세요.")
                break
        return errors, warnings

    def add_listing(self, values: dict) -> tuple[str, list[str]]:
        """새 매물을 매물 폴더의 웹입력 엑셀에 추가하고 다시 읽음. (매물번호, 경고) 반환. 오류면 ValueError."""
        from .listing_editor import append_listing, new_listing_id, preview_listing, web_file

        existing = {l.listing_id for l in self.listings() + self.hidden_listings}
        listing_id = new_listing_id(existing)
        errors, warnings = self.check_listing(preview_listing(listing_id, values, web_file(self.settings.data_dir)))
        if errors:
            raise ValueError(" / ".join(errors))
        try:
            append_listing(self.settings.data_dir, listing_id, values)
        except PermissionError as e:
            raise ValueError("웹입력매물.xlsx 가 엑셀에서 열려 있어 저장하지 못했습니다. 엑셀을 닫고 다시 저장하세요.") from e
        self.sync()
        if self.find_listing(listing_id) is None:
            raise ValueError("저장했지만 매물로 읽히지 않았습니다. 관리자 화면의 매물 정리 보고서를 확인하세요.")
        return listing_id, warnings

    def edit_listing(self, listing_id: str, values: dict, hidden: Optional[bool] = None) -> list[str]:
        """기존 매물 수정: 원본과 달라진 항목만 저장해 덮어씀. 경고 목록 반환. 오류면 ValueError."""
        from .listing_editor import HIDDEN, apply_fields, diff_fields

        current = self.find_listing(listing_id)
        if current is None:
            raise ValueError("매물을 찾을 수 없습니다. (매물 폴더에서 지워졌을 수 있습니다)")
        original = self.original_listing(listing_id)
        fields = diff_fields(original, values) if values else {
            k: v for k, v in self.listing_edit(listing_id).items() if k != HIDDEN}
        hide = bool(self.listing_edit(listing_id).get(HIDDEN)) if hidden is None else hidden
        candidate = copy.deepcopy(original)
        apply_fields(candidate, fields)
        errors, warnings = self.check_listing(candidate, ignore_id=listing_id) if values else ([], [])
        if errors:
            raise ValueError(" / ".join(errors))
        if hide:
            fields[HIDDEN] = True
        if fields:
            self.storage.save_listing_edit(listing_id, fields)
            self._originals.setdefault(listing_id, copy.deepcopy(original))
        else:
            self.storage.delete_listing_edit(listing_id)
        self._replace_listing(listing_id, candidate if fields else copy.deepcopy(original), hide)
        return warnings

    def revert_listing(self, listing_id: str) -> None:
        """관리자가 고친 내용을 모두 지우고 원본 엑셀 내용으로 되돌림 (숨김도 해제)."""
        original = self.original_listing(listing_id)
        if original is None:
            raise ValueError("매물을 찾을 수 없습니다.")
        self.storage.delete_listing_edit(listing_id)
        self._replace_listing(listing_id, copy.deepcopy(original), hide=False)
        self._originals.pop(listing_id, None)

    def _replace_listing(self, listing_id: str, new: LandListing, hide: bool) -> None:
        visible = [l for l in self.listings() if l.listing_id != listing_id]
        self.hidden_listings = [l for l in self.hidden_listings if l.listing_id != listing_id]
        (self.hidden_listings if hide else visible).append(new)
        self._listings = visible
        self.storage.upsert_listings(visible)

    def known_regions(self) -> list[str]:
        regions = set()
        for l in self.listings():
            for tok in l.address.split():
                if tok[-1:] in {"시", "군", "구", "읍", "면", "동", "리"} and len(tok) >= 2:
                    regions.add(tok)
        return sorted(regions)

    # ------------------------------------------------------------------ 3~5. 상담·추천
    def parser(self) -> RequirementParser:
        return RequirementParser(self.llm, self.known_regions(), [l.address for l in self.listings()])

    def understand(self, inquiry: str, customer_name: str = "") -> CustomerRequirement:
        return self.parser().parse(inquiry, customer_name)

    @property
    def office_model(self) -> PreferenceModel:
        """사무실 공통 학습 모델. 여러 상담(웹 접속자)이 같은 객체를 함께 학습시킴."""
        if self._office_model is None:
            self._office_model = self.storage.load_preference(OFFICE_MODEL)
        return self._office_model

    def _options(self, customer: str) -> RecommendOptions:
        """손님별 학습 모델 + 사무실 공통 모델을 불러와 추천 옵션 구성."""
        pref = self.storage.load_preference(customer) if customer else PreferenceModel()
        return RecommendOptions(market_price_per_m2=self.market_price, pref=pref, prior=self.office_model)

    def _save_learning(self, session: RecommendationSession) -> None:
        if session.opt.prior is not None:
            self.storage.save_preference(OFFICE_MODEL, session.opt.prior)
        if session.req.customer_name and session.opt.pref is not None:
            self.storage.save_preference(session.req.customer_name, session.opt.pref)

    def recommend(self, req: CustomerRequirement) -> RecommendationSession:
        exclude = self.storage.rejected_ids(req.customer_name)
        return RecommendationSession(self.listings(), req, self._options(req.customer_name), exclude_ids=exclude)

    def reject(self, session: RecommendationSession, reason: str = "", ai: bool = True) -> Optional[ScoredListing]:
        """현재 매물이 마음에 안 들 때: 거절 기록 + 취향 학습(규칙 + 생성형 AI) 후 다음 순위 매물."""
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "reject", reason)
        session.next(reason)
        if ai:
            self.apply_ai_learning(session, self.ai_learn(session))
        self._save_learning(session)
        return session.current()

    # ------------------------------------------------------------------ AI 연결
    def reconnect_llm(self) -> str:
        """.env / my_api_key.py 를 다시 읽고 GPT/Claude 연결을 새로 만듦 (서버 재시작 없이 키 변경 반영)."""
        from .config import load_keys

        load_keys()
        s = self.settings
        s.openai_model = os.getenv("OPENAI_MODEL", s.openai_model) or s.openai_model
        s.anthropic_model = os.getenv("ANTHROPIC_MODEL", s.anthropic_model) or s.anthropic_model
        self.llm = LLMRouter(s.llm_providers, s.anthropic_model, s.openai_model)
        self._llm_built_at = time.time()
        return self.llm.status()

    def maybe_reconnect_llm(self, every_sec: int = 180) -> None:
        """AI 가 꺼져 있으면 몇 분마다 한 번씩 다시 연결 시도 (키를 고친 뒤 재시작하지 않아도 되도록)."""
        if self.llm.mode() != "off" or not any(p not in {"rule"} for p in self.settings.llm_providers):
            return
        if time.time() - getattr(self, "_llm_built_at", 0) >= every_sec:
            self.reconnect_llm()

    def test_llm(self) -> tuple[bool, str]:
        """실제로 한 번 물어봐서 연결 확인."""
        if not self.llm.available:
            return False, self.llm.off_reason()
        try:
            answer = self.llm.complete("짧게 답하는 도우미입니다.", "연결 확인입니다. '확인' 이라고만 답하세요.",
                                       max_tokens=200)
            model = next((getattr(c, "model", "") for c in self.llm.clients if c.name == self.llm.last_used), "")
            return True, f"{self.llm.last_used}({model}) 응답: {answer.strip()[:40]}"
        except LLMUnavailable as e:
            return False, str(e)

    def test_llm_each(self) -> list[tuple[bool, str]]:
        """연결된 GPT·Claude 각각에 한 번씩 물어봐서 확인 (관리자 화면)."""
        label = {"openai": "GPT", "anthropic": "Claude"}
        out = []
        for c in list(self.llm.clients):
            name = f"{label.get(c.name, c.name)}({getattr(c, 'model', '')})"
            try:
                answer = c.complete("짧게 답하는 도우미입니다.", "연결 확인입니다. '확인' 이라고만 답하세요.",
                                    max_tokens=200)
                out.append((True, f"{name} 연결 성공: {answer.strip()[:20]}"))
            except LLMUnavailable as e:
                out.append((False, f"{name} 연결 실패: {e}"))
        for name, reason in self.llm.unavailable.items():
            if name in {"openai", "anthropic"} and "키가 없습니다" not in reason and "설정되지 않았습니다" not in reason:
                out.append((False, f"{label.get(name, name)} 사용 안 함: {reason}"))
        return out

    # ------------------------------------------------------------------ 생성형 AI 학습
    def ai_learn(self, session: RecommendationSession) -> Optional[dict]:
        """손님 반응 기록을 GPT/Claude 에 보내 취향을 추론 (느릴 수 있으므로 웹에서는 잠금 밖에서 호출)."""
        from .ai_learning import ask_ai

        return ask_ai(self.llm, session.req.raw_text or session.req.describe(), session.events)

    def apply_ai_learning(self, session: RecommendationSession, plan: Optional[dict]) -> list[str]:
        session.ai_changes = []
        if not plan:
            return []
        from .ai_learning import apply_plan

        session.ai_changes = apply_plan(session, plan)
        session.rerank_keep_order_of_seen()
        self._save_learning(session)
        return session.ai_changes

    def refine(self, session: RecommendationSession, feedback: str) -> CustomerRequirement:
        """피드백을 조건에 반영 후 재순위 (현재 매물도 거절·학습 처리)."""
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "reject", feedback)
        session.reject_current(feedback)
        new_req = self.parser().apply_feedback(session.req, feedback)
        session.update_requirement(new_req)
        self.apply_ai_learning(session, self.ai_learn(session))
        self._save_learning(session)
        return session.req

    def save_customer(self, session: RecommendationSession) -> None:
        """손님 조건을 저장(선제 알림 대상)하고, 이름이 정해지기 전에 거절한 매물도 그 손님 기록으로 옮김."""
        name = session.req.customer_name
        self.storage.save_customer(session.req)
        self._save_learning(session)  # 이름 없이 상담하며 학습한 취향도 이 손님 것으로 저장
        already = self.storage.rejected_ids(name)
        for listing_id, reason in session.rejected.items():
            if listing_id not in already:
                self.storage.add_feedback(name, listing_id, "reject", reason)

    def like(self, session: RecommendationSession, ai: bool = True) -> None:
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "like")
        session.like()
        if ai:
            self.apply_ai_learning(session, self.ai_learn(session))
            if cur:
                session._bring_to_front(cur.listing.listing_id)  # 관심 표시한 매물은 계속 보여줌
        self._save_learning(session)

    def respond(self, session: RecommendationSession, text: str) -> str:
        """손님이 한 말을 해석해 처리. '마음에 안 들어요/너무 비싸요' -> 'reject' (다음 순위),
        '좀 더 싼 걸로/여주 쪽도' -> 'refine' (조건 변경 후 재추천)."""
        text = text.strip()
        if is_condition_change(text):
            self.refine(session, text)
            return "refine"
        self.reject(session, text)
        return "reject"

    def explain(self, req: CustomerRequirement, s: ScoredListing, use_llm: bool = True,
                public: bool = False) -> str:
        """추천 설명. public=True 면 손님용으로 지번을 가림."""
        address = mask_address(s.listing.address) if public else s.listing.address
        if use_llm and self.llm.available:
            l = s.listing
            info = {
                "주소": address, "지목": l.jimok, "용도지역": l.zoning, "면적_평": round(l.area_pyeong),
                "면적_㎡": l.area_m2, "가격": format_price(l.price_manwon), "평당가_만원": round(l.price_per_pyeong),
                "도로": l.road_access, "특이사항": l.description, "점수": s.score,
                "장점": s.reasons, "확인필요": s.penalties,
            }
            try:
                return self.llm.complete(
                    EXPLAIN_SYSTEM,
                    f"손님 조건: {req.describe()}\n원문: {req.raw_text}\n\n매물:\n{json.dumps(info, ensure_ascii=False)}",
                    max_tokens=600, optional=True,
                ).strip()
            except LLMUnavailable as e:
                log.info("LLM 설명 생성 실패: %s", e)
        return template_explanation(s, address)

    # ------------------------------------------------------------------ 6. 선제 제안
    def proactive_scan(self, candidates: Optional[list[LandListing]] = None,
                       threshold: Optional[float] = None, use_llm: bool = False) -> list[ProactiveAlert]:
        """저장된 손님 조건과 (신규) 매물을 대조해 아직 알리지 않은 고득점 매물을 찾음."""
        threshold = self.settings.proactive_threshold if threshold is None else threshold
        candidates = self.listings() if candidates is None else candidates
        alerts = []
        for req in self.storage.customers():
            opt = self._options(req.customer_name)
            rejected = self.storage.rejected_ids(req.customer_name)
            for l in candidates:
                if l.listing_id in rejected or self.storage.was_notified(req.customer_name, l.listing_id):
                    continue
                if hard_filter(l, req, opt):
                    continue
                s = score_listing(l, req, opt)
                if s.score < threshold:
                    continue
                msg = self.explain(req, s, use_llm=use_llm)
                self.storage.add_notification(req.customer_name, l.listing_id, s.score, msg)
                alerts.append(ProactiveAlert(req.customer_name, s, msg))
        alerts.sort(key=lambda a: -a.scored.score)
        self._write_alert_log(alerts)
        return alerts

    def watch(self, on_alert: Callable[[ProactiveAlert], None], on_sync: Callable[[SyncReport], None] | None = None,
              interval: Optional[int] = None, iterations: Optional[int] = None) -> None:
        """매물 폴더를 주기적으로 다시 읽고, 새로 들어오거나 바뀐 매물을 손님들에게 선제 제안."""
        interval = interval or self.settings.watch_interval_sec
        n = 0
        while iterations is None or n < iterations:
            report = self.sync()
            if on_sync:
                on_sync(report)
            fresh = report.new + report.changed
            if fresh:
                for alert in self.proactive_scan(fresh, use_llm=True):
                    on_alert(alert)
            n += 1
            if iterations is None or n < iterations:
                time.sleep(interval)

    # ------------------------------------------------------------------ 로그
    def _log_path(self, name: str) -> Path:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.log_dir / name

    def _write_quality_log(self, report: SyncReport) -> None:
        try:
            with open(self._log_path("quality_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S} 매물 정제 보고서\n")
                f.write(report.summary() + "\n\n" + report.clean.report() + "\n")
        except OSError as e:
            log.warning("품질 보고서 저장 실패: %s", e)

    def _write_alert_log(self, alerts: list[ProactiveAlert]) -> None:
        if not alerts:
            return
        try:
            with open(self._log_path("proactive_alerts.log"), "a", encoding="utf-8") as f:
                for a in alerts:
                    f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{a.customer}\t{a.scored.score}\t"
                            f"{a.scored.listing.summary()}\n")
        except OSError as e:
            log.warning("알림 로그 저장 실패: %s", e)


# '이 매물이 별로' 라는 평가 (-> 다음 순위)
_DISLIKE = r"(별로|싫|마음에\s*안|맘에\s*안|안\s*좋|아니|비싸|좁|작아|멀|어둡|그늘|불편|애매|글쎄|패스|다른\s*(거|것|매물))"
# 조건을 바꾸려는 말인지 판단하는 단서
_CONDITION_CUES = r"(더\s|좀\s*더|으로|로\s*(해|바꿔|보여)|바꿔|말고|쪽|이상|이하|이내|평|억|만원|지역|도로\s*(있|접|붙))"


def is_condition_change(text: str) -> bool:
    """True: 조건을 바꾸려는 말 / False: 지금 매물이 마음에 안 든다는 말."""
    from .requirement_parser import parse_rule_based

    if re.search(_DISLIKE, text) and not re.search(_CONDITION_CUES, text):
        return False
    return bool(re.search(_CONDITION_CUES, text)) or parse_rule_based(text).describe() != "(조건 없음)"


def public_region(address: str) -> str:
    """손님에게 보여줄 지역: 시/도 + 시/군/구 까지만. '음성군 금왕읍 각회리 210-3' -> '충청북도 음성군'
    (시/도가 안 적혀 있으면 시/군 이름으로 채움, '청주시 흥덕구' 처럼 시 아래 구는 함께)"""
    from .regions import SIDO_ALIASES, SIGUNGU_TO_SIDO, sido_of

    sido, parts = "", []
    for tok in address.split():
        if not sido and not parts and sido_of(tok):
            sido = max(SIDO_ALIASES[sido_of(tok)], key=len)  # '충북' -> '충청북도'
            continue
        if re.fullmatch(r"[가-힣]{1,10}(시|군|구)", tok) and len(parts) < 2:
            if parts and not (parts[0].endswith("시") and tok.endswith("구")):
                break
            parts.append(tok)
            continue
        break
    if parts and not sido:
        owners = SIGUNGU_TO_SIDO.get(parts[0], set())
        if len(owners) == 1:
            sido = max(SIDO_ALIASES[next(iter(owners))], key=len)
    return " ".join(([sido] if sido else []) + parts)


def mask_address(address: str) -> str:
    """손님용 화면: 시/도 + 시/군/구 까지만 보이고 나머지(읍·면·동·리, 도로명, 지번, 상세주소)는 가림.
    '음성군 금왕읍 각회리 210-3' -> '충청북도 음성군 ***'"""
    region = public_region(address.strip())
    return f"{region} ***" if region else "***"


def public_office_no(office_no: str) -> str:
    """손님·AI 에게 보여줄 매물번호: 'S.24250409(이재순님)' -> 'S.24250409' (괄호 안 이름·메모는 가림)"""
    return re.sub(r"\s*[(\[（].*$", "", office_no or "").strip()


def template_explanation(s: ScoredListing, address: Optional[str] = None) -> str:
    l = s.listing
    text = (f"{(address or l.address).replace(' ***', '')} 소재 {l.jimok or '토지'} {l.area_pyeong:,.0f}평 매물로, "
            f"매매가는 {format_price(l.price_manwon)}(평당 약 {l.price_per_pyeong:,.0f}만원)입니다.")
    if s.reasons:
        text += " 장점: " + ", ".join(s.reasons[:4]) + "."
    if s.penalties:
        text += " 확인 필요: " + ", ".join(s.penalties[:3]) + "."
    return text
