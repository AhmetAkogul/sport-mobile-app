"""Antrenor paneli: olay JSONL dosyasini izler, tarayiciya canli aktarir (0042).

    python -m mono.olay_servisi --kayit out/bildirim.jsonl --port 8765
    # tarayici: http://localhost:8765

Yalniz standart kutuphane (yeni bagimlilik yok). Aktarim Server-Sent Events:
sunucudan panele tek yonlu akis icin yeterli, tarayicida `EventSource` ile
kendiliginden yeniden baglanir. Uretici (`make antrenor --bildirim-kaydi`,
salonda her istasyonun sureci) ile panel birbirinden bagimsizdir: ortak nokta
dosyadir; bir istasyonun cokmesi paneli durdurmaz.

Uclar: `/` panel, `/olaylar` SSE (her olay ve "uyari"), `/durum` kisi ozeti (JSON).
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mono.olay import PanelDurumu, olay_dogrula

PANEL = Path(__file__).with_name("panel.html")


class OlayKaynagi:
    """JSONL dosyasini baslangictan okur, sonra sonuna eklenenleri izler."""

    def __init__(self, yol: Path, aralik_s: float = 0.25):
        self.yol, self.aralik_s = Path(yol), aralik_s
        self.durum = PanelDurumu()
        self.gecmis: list[dict] = []           # olay ya da uyari, sirali
        self._kosul = threading.Condition()
        self._konum = 0
        self.bozuk = 0

    def oku(self) -> int:
        """Yeni satirlari isler; eklenen kayit sayisi."""
        if not self.yol.exists():
            return 0
        with open(self.yol, "rb") as f:
            f.seek(self._konum)
            ham = f.read()
        son = ham.rfind(b"\n")
        if son < 0:
            return 0                           # yarim yazilmis satir: sonra okunur
        self._konum += son + 1
        yeni = []
        for s in ham[:son].decode("utf-8").splitlines():
            if not s.strip():
                continue
            try:
                olay = json.loads(s)
                olay_dogrula(olay)
            except (ValueError, KeyError, AttributeError, TypeError):
                self.bozuk += 1
                continue
            yeni.append(olay)
            uyari = self.durum.ekle(olay)
            if uyari:
                yeni.append(uyari)
        if yeni:
            with self._kosul:
                self.gecmis += yeni
                self._kosul.notify_all()
        return len(yeni)

    def izle(self, dur: threading.Event) -> None:
        while not dur.is_set():
            self.oku()
            dur.wait(self.aralik_s)

    def bekle(self, sira: int, zaman_asimi: float) -> list[dict]:
        with self._kosul:
            if len(self.gecmis) <= sira:
                self._kosul.wait(zaman_asimi)
            return self.gecmis[sira:]


def isleyici(kaynak: OlayKaynagi):
    class Isleyici(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 -- sessiz
            pass

        def _gonder(self, kod, tur, govde: bytes):
            self.send_response(kod)
            self.send_header("Content-Type", tur)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(govde)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._gonder(200, "text/html; charset=utf-8", PANEL.read_bytes())
            elif self.path == "/durum":
                self._gonder(200, "application/json; charset=utf-8", json.dumps(
                    {"kisiler": kaynak.durum.ozet(), "bozuk_satir": kaynak.bozuk},
                    ensure_ascii=False).encode())
            elif self.path == "/olaylar":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sira = max(len(kaynak.gecmis) - 50, 0)   # yeni baglanana son 50 olay
                try:
                    while True:
                        yeni = kaynak.bekle(sira, 15.0)
                        if not yeni:
                            self.wfile.write(b": canli\n\n")      # baglanti canli tutma
                        for o in yeni:
                            self.wfile.write(
                                f"data: {json.dumps(o, ensure_ascii=False)}\n\n".encode())
                        sira += len(yeni)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._gonder(404, "text/plain; charset=utf-8", b"yok")

    return Isleyici


def sunucu(kayit: Path, port: int, host: str = "127.0.0.1"):
    """(sunucu, kaynak, dur); cagiran `serve_forever` ile calistirir."""
    kaynak = OlayKaynagi(kayit)
    kaynak.oku()
    dur = threading.Event()
    threading.Thread(target=kaynak.izle, args=(dur,), daemon=True).start()
    s = ThreadingHTTPServer((host, port), isleyici(kaynak))
    s.daemon_threads = True
    return s, kaynak, dur


def main(argv=None):
    p = argparse.ArgumentParser(description="Antrenor paneli (olay JSONL -> tarayici)")
    p.add_argument("--kayit", type=Path, required=True, help="olay JSONL (--bildirim-kaydi)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 salon agindaki antrenor tabletine acar")
    a = p.parse_args(argv)
    s, _, dur = sunucu(a.kayit, a.port, a.host)
    print(f"Antrenor paneli: http://{a.host}:{a.port}  (izlenen: {a.kayit})", flush=True)
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dur.set()
        s.server_close()


if __name__ == "__main__":
    main()
