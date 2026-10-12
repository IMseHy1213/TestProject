"""정부 공공데이터 / GIS 연동 (선택 기능 - API 키가 있을 때만 동작).

1) 국토교통부 토지 매매 실거래가 (공공데이터포털 data.go.kr)
   - 지역 시세(㎡당 거래가 중앙값)를 구해 '시세 대비 저렴한 매물' 점수에 사용합니다.
   - 활용 신청: https://www.data.go.kr  ->  '국토교통부_토지 매매 실거래가 자료'
2) 브이월드(VWorld) 지오코딩 (국토교통부 공간정보 오픈플랫폼)
   - 주소 -> 좌표 변환. 좌표가 없는 매물에 위·경도를 채워 지도 표시/거리 계산에 씁니다.
   - 키 발급: https://www.vworld.kr

API 사양이 바뀔 수 있으니 실제 응답 필드는 발급 후 한 번 확인하세요.
"""

from __future__ import annotations

import logging
import statistics
import xml.etree.ElementTree as ET
from typing import Optional

log = logging.getLogger(__name__)

LAND_TRADE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcLandTrade/getRTMSDataSvcLandTrade"
VWORLD_GEOCODE_URL = "https://api.vworld.kr/req/address"

# 응답 필드명은 영문/국문 두 가지 버전이 있어 모두 시도합니다.
_AMOUNT_TAGS = ("dealAmount", "거래금액")
_AREA_TAGS = ("dealArea", "거래면적")
_JIMOK_TAGS = ("jimok", "지목")
_UMD_TAGS = ("umdNm", "법정동")


def _first(item: ET.Element, tags: tuple[str, ...]) -> Optional[str]:
    for t in tags:
        el = item.find(t)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return None


class LandTradePriceClient:
    """국토부 토지 매매 실거래가 조회."""

    def __init__(self, service_key: str, timeout: float = 10.0):
        self.service_key = service_key
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.service_key)

    def fetch(self, lawd_cd: str, deal_ym: str, rows: int = 1000) -> list[dict]:
        """lawd_cd: 법정동 시군구코드 5자리 (예: 양평군 41830), deal_ym: 'YYYYMM'."""
        import requests

        resp = requests.get(
            LAND_TRADE_URL,
            params={
                "serviceKey": self.service_key,
                "LAWD_CD": lawd_cd,
                "DEAL_YMD": deal_ym,
                "numOfRows": rows,
                "pageNo": 1,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        # XXE 공격 방지를 위해 defusedxml 로 파싱 (이 기능을 쓸 때만 필요하므로 여기서 import)
        from defusedxml.ElementTree import fromstring as safe_fromstring

        root = safe_fromstring(resp.content)
        out = []
        for item in root.iter("item"):
            amount = _first(item, _AMOUNT_TAGS)
            area = _first(item, _AREA_TAGS)
            if not amount or not area:
                continue
            try:
                out.append({
                    "price_manwon": int(amount.replace(",", "")),
                    "area_m2": float(area.replace(",", "")),
                    "jimok": _first(item, _JIMOK_TAGS) or "",
                    "umd": _first(item, _UMD_TAGS) or "",
                })
            except ValueError:
                continue
        return out

    def market_price_per_m2(self, lawd_cd: str, months: list[str]) -> dict[str, float]:
        """지목별 ㎡당 거래가(만원) 중앙값. key '*' 는 전체."""
        trades: list[dict] = []
        for ym in months:
            try:
                trades.extend(self.fetch(lawd_cd, ym))
            except Exception as e:
                log.warning("실거래가 조회 실패 %s %s: %s", lawd_cd, ym, e)
        by_jimok: dict[str, list[float]] = {"*": []}
        for t in trades:
            if t["area_m2"] <= 0:
                continue
            ppm = t["price_manwon"] / t["area_m2"]
            by_jimok["*"].append(ppm)
            by_jimok.setdefault(t["jimok"], []).append(ppm)
        return {k: statistics.median(v) for k, v in by_jimok.items() if v}


class VWorldGeocoder:
    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
        self._cache: dict[str, Optional[tuple[float, float]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def geocode(self, address: str) -> Optional[tuple[float, float]]:
        """지번 주소 -> (위도, 경도)."""
        if address in self._cache:
            return self._cache[address]
        import requests

        result = None
        try:
            resp = requests.get(
                VWORLD_GEOCODE_URL,
                params={
                    "service": "address", "request": "getcoord", "version": "2.0", "crs": "epsg:4326",
                    "address": address, "format": "json", "type": "parcel", "key": self.api_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json().get("response", {})
            if body.get("status") == "OK":
                pt = body["result"]["point"]
                result = (float(pt["y"]), float(pt["x"]))
        except Exception as e:
            log.warning("지오코딩 실패 %s: %s", address, e)
        self._cache[address] = result
        return result
