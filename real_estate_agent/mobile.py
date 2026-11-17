"""휴대폰에서 쓰기: 같은 와이파이의 휴대폰 브라우저로 이 PC 의 웹 서비스에 접속.

    python main.py mobile        # 와이파이에 열고, 휴대폰 주소와 QR 코드를 보여줌

- 별도 앱 설치 없이 휴대폰 브라우저(크롬·삼성 인터넷·사파리)로 엽니다. 화면은 휴대폰 폭에 맞게 바뀝니다.
- 브라우저 메뉴의 '홈 화면에 추가' 를 누르면 앱처럼 아이콘이 생기고 전체 화면으로 열립니다 (manifest).
- PC 가 켜져 있고 프로그램이 실행 중이어야 하며, 휴대폰과 PC 가 같은 와이파이(공유기)에 있어야 합니다.
- 인터넷 전체가 아니라 사무실·집 와이파이 안에서만 열립니다 (공유기 밖에서는 접속 불가).
"""

from __future__ import annotations

import io
import ipaddress
import socket
from typing import Optional


def _private(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.version == 4 and a.is_private and not a.is_loopback and not a.is_link_local


def lan_addresses() -> list[str]:
    """이 PC 의 와이파이·랜 주소 (예: ['192.168.0.12']). 휴대폰은 이 주소로 접속."""
    found: list[str] = []
    try:  # 바깥으로 나가는 길에 쓰는 주소 (실제로 패킷을 보내지는 않음)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.append(ip)
    except OSError:
        pass
    out = []
    for ip in found:
        if _private(ip) and ip not in out:
            out.append(ip)
    return out  # 맨 앞 = 실제로 공유기로 나가는 주소 (가상 PC·VPN 주소는 뒤로)


def phone_urls(port: int, addresses: Optional[list[str]] = None) -> list[str]:
    return [f"http://{ip}:{port}" for ip in (lan_addresses() if addresses is None else addresses)]


def qr_svg(text: str) -> str:
    """QR 코드 SVG (관리자 화면에 바로 넣음). qrcode 패키지가 없으면 ''."""
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return ""
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf)
    svg = buf.getvalue().decode("utf-8")
    return svg[svg.index("<svg"):] if "<svg" in svg else svg


def qr_text(text: str) -> str:
    """터미널에 찍는 QR 코드 (휴대폰 카메라로 비추면 주소가 열림). qrcode 패키지가 없으면 ''."""
    try:
        import qrcode
    except ImportError:
        return ""
    qr = qrcode.QRCode(border=1)
    qr.add_data(text)
    qr.make(fit=True)
    buf = io.StringIO()
    qr.print_ascii(out=buf, invert=True)
    return buf.getvalue()


def print_phone_help(port: int) -> None:
    urls = phone_urls(port)
    print("\n📱 휴대폰에서 쓰기 (휴대폰과 이 PC 가 같은 와이파이에 있어야 합니다)")
    if not urls:
        print("   ⚠ 이 PC 의 와이파이 주소를 찾지 못했습니다. 와이파이(또는 랜선)에 연결되어 있는지 확인하세요.")
        print("     명령 프롬프트에서 ipconfig 를 입력하면 'IPv4 주소' 에 나오는 192.168.x.x 가 이 PC 주소입니다.")
        return
    print(f"   휴대폰 브라우저 주소창에 입력: {urls[0]}")
    for u in urls[1:]:
        print(f"   (안 되면 이 주소도 시도: {u})")
    art = qr_text(urls[0])
    if art:
        print("   또는 휴대폰 카메라로 아래 QR 코드를 비추세요:")
        try:
            print(art)
        except UnicodeEncodeError:  # 아주 오래된 콘솔 글꼴
            print("   (이 창에서는 QR 코드를 그릴 수 없습니다 - 관리자 화면의 '📱 휴대폰으로 보기' 에도 QR 코드가 있습니다)")
    print("   · 처음 실행할 때 윈도 '방화벽' 창이 뜨면 '개인 네트워크' 에 체크하고 [액세스 허용] 을 누르세요.")
    print("   · 휴대폰 브라우저 메뉴의 '홈 화면에 추가' 를 누르면 앱처럼 아이콘이 생깁니다.")
    print("   · 와이파이 안에서만 열립니다. 공유기의 '포트 포워딩' 으로 인터넷에 열지는 마세요 (관리자 화면 보호).")
