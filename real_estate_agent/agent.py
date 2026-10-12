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
        self._listings: Optional[list[LandListing]] = None
        self._office_model: Optional[PreferenceModel] = None
        self.last_sync: Optional[SyncReport] = None
        self.last_sync_at: Optional[datetime] = None

    # ------------------------------------------------------------------ 1~2. 수집·정제
    def sync(self, today: Optional[date] = None) -> SyncReport:
        if self.settings.data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
            from .template import prepare_data_folder

            prepare_data_folder(self.settings.data_dir)  # 프로젝트 '매물' 폴더는 없으면 자동으로 만듦
        records = self.collector.collect()
        result = clean(records, today)
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

    def known_regions(self) -> list[str]:
        regions = set()
        for l in self.listings():
            for tok in l.address.split():
                if tok[-1:] in {"시", "군", "구", "읍", "면", "동", "리"} and len(tok) >= 2:
                    regions.add(tok)
        return sorted(regions)

    # ------------------------------------------------------------------ 3~5. 상담·추천
    def parser(self) -> RequirementParser:
        return RequirementParser(self.llm, self.known_regions())

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

    def reject(self, session: RecommendationSession, reason: str = "") -> Optional[ScoredListing]:
        """현재 매물이 마음에 안 들 때: 거절 기록 + 취향 학습 후 다음 순위 매물."""
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "reject", reason)
        nxt = session.next(reason)
        self._save_learning(session)
        return nxt

    def refine(self, session: RecommendationSession, feedback: str) -> CustomerRequirement:
        """피드백을 조건에 반영 후 재순위 (현재 매물도 거절·학습 처리)."""
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "reject", feedback)
        session.reject_current(feedback)
        new_req = self.parser().apply_feedback(session.req, feedback)
        session.update_requirement(new_req)
        self._save_learning(session)
        return new_req

    def save_customer(self, session: RecommendationSession) -> None:
        """손님 조건을 저장(선제 알림 대상)하고, 이름이 정해지기 전에 거절한 매물도 그 손님 기록으로 옮김."""
        name = session.req.customer_name
        self.storage.save_customer(session.req)
        self._save_learning(session)  # 이름 없이 상담하며 학습한 취향도 이 손님 것으로 저장
        already = self.storage.rejected_ids(name)
        for listing_id, reason in session.rejected.items():
            if listing_id not in already:
                self.storage.add_feedback(name, listing_id, "reject", reason)

    def like(self, session: RecommendationSession) -> None:
        cur = session.current()
        if cur:
            self.storage.add_feedback(session.req.customer_name, cur.listing.listing_id, "like")
        session.like()
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
                    max_tokens=600,
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


def mask_address(address: str) -> str:
    """손님용 화면: 정확한 지번은 가리고 리/동까지만 표시. '음성군 금왕읍 무극리 210-3' -> '... 무극리 ***'"""
    address = address.strip()
    multi = re.search(r"\s*외\s*\d+\s*필지.*$", address)  # '417-2 외 36필지' -> '*** 외 36필지'
    tail = multi.group(0).strip() if multi else ""
    core = address[:multi.start()] if multi else address
    masked = re.sub(r"\s*(산\s*)?\d+(-\d+)?(번지)?\s*$", "", core).strip()
    if masked == core.strip():
        return address
    return f"{masked} ***" + (f" {tail}" if tail else "")


def template_explanation(s: ScoredListing, address: Optional[str] = None) -> str:
    l = s.listing
    text = (f"{address or l.address}의 {l.jimok or '토지'} {l.area_pyeong:,.0f}평 매물로, "
            f"매매가는 {format_price(l.price_manwon)}(평당 약 {l.price_per_pyeong:,.0f}만원)입니다.")
    if s.reasons:
        text += " 장점: " + ", ".join(s.reasons[:4]) + "."
    if s.penalties:
        text += " 확인 필요: " + ", ".join(s.penalties[:3]) + "."
    return text
