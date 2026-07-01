#!/usr/bin/env python3
"""Generate the one-page printable sign that announces the freesoft-interpret
machine-translation service in all 13 supported languages.

Each row is a national flag plus a statement, in that language, that the
service is available. Output: ../translation-service-sign.pdf (US Letter,
one page). Complex scripts (Arabic RTL, Devanagari conjuncts, CJK, Cyrillic)
are shaped by Chromium's HarfBuzz using the system Noto fonts, so a Noto
font set covering those scripts must be installed (fc-list | grep -i noto).

Flags are vendored under ./flags/ (from lipis/flag-icons, MIT; see
flags/README.md and flags/LICENSE), so no network or npm is needed.
Regenerate with:  python3 build.py

Chromium is auto-detected (Playwright cache, then chromium/chrome on PATH);
override with CHROME=/path/to/chrome.
"""
import glob
import html
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = os.path.join(HERE, "flags")
HTML_OUT = os.path.join(HERE, "sign.html")
PDF_OUT = os.path.join(HERE, "..", "translation-service-sign.pdf")

CJK_SC = '"Noto Sans CJK SC"'
CJK_JP = '"Noto Sans CJK JP"'
CJK_KR = '"Noto Sans CJK KR"'
ARAB   = '"Noto Naskh Arabic","Noto Sans Arabic"'
DEVA   = '"Noto Sans Devanagari"'

# (flag country code, endonym, statement, text direction, font-family override)
# Flags for languages spoken across many countries are a judgement call:
# English=US, Spanish=Spain, Portuguese=Portugal, Arabic=Saudi Arabia.
ROWS = [
    ("us", "English",    "We offer a machine translation service to assist you in English.", "ltr", None),
    ("es", "Español",    "Ofrecemos un servicio de traducción automática para ayudarle en español.", "ltr", None),
    ("it", "Italiano",   "Offriamo un servizio di traduzione automatica per assisterti in italiano.", "ltr", None),
    ("fr", "Français",   "Nous proposons un service de traduction automatique pour vous aider en français.", "ltr", None),
    ("de", "Deutsch",    "Wir bieten einen maschinellen Übersetzungsdienst, um Ihnen auf Deutsch zu helfen.", "ltr", None),
    ("pt", "Português",  "Oferecemos um serviço de tradução automática para ajudá-lo em português.", "ltr", None),
    ("nl", "Nederlands", "Wij bieden een automatische vertaaldienst om u in het Nederlands te helpen.", "ltr", None),
    ("ru", "Русский",    "Мы предлагаем услугу машинного перевода, чтобы помочь вам на русском языке.", "ltr", None),
    ("sa", "العربية",    "نوفّر خدمة ترجمة آلية لمساعدتك باللغة العربية.", "rtl", ARAB),
    ("in", "हिन्दी",      "हम हिंदी में आपकी सहायता के लिए मशीन अनुवाद सेवा प्रदान करते हैं।", "ltr", DEVA),
    ("cn", "中文",        "我们提供机器翻译服务，为您提供中文支持。", "ltr", CJK_SC),
    ("jp", "日本語",      "日本語でのご利用をサポートする機械翻訳サービスをご用意しています。", "ltr", CJK_JP),
    ("kr", "한국어",      "한국어 지원을 위한 기계 번역 서비스를 제공합니다.", "ltr", CJK_KR),
]


def flag_svg(code):
    with open(os.path.join(FLAGS, code + ".svg"), encoding="utf-8") as f:
        svg = f.read()
    return svg[svg.find("<svg"):]     # strip any XML/doctype preamble


def row_html(code, endo, text, dr, font):
    fam = f"font-family:{font},'Noto Sans',sans-serif;" if font else ""
    align = "text-align:right;" if dr == "rtl" else ""
    return f"""    <div class="row">
      <div class="flag">{flag_svg(code)}</div>
      <div class="text" dir="{dr}" style="{fam}{align}">
        <div class="endo">{html.escape(endo)}</div>
        <div class="stmt">{html.escape(text)}</div>
      </div>
    </div>"""


def build_html():
    rows = "\n".join(row_html(*r) for r in ROWS)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: Letter portrait; margin: 0.45in 0.6in; }}
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact;
       box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: "Noto Sans", sans-serif; color: #14181f; }}
  header {{ text-align: center; border-bottom: 3px solid #2f6f6f;
           padding-bottom: 10px; margin-bottom: 14px; }}
  header h1 {{ font-size: 28px; margin: 0 0 4px; color: #1f4e4e;
              letter-spacing: .2px; }}
  header .sub {{ font-size: 13px; color: #5a6470; margin: 0; }}
  .row {{ display: flex; align-items: center; gap: 16px;
         padding: 7px 4px; border-bottom: 1px solid #e3e6ea; }}
  .row:last-child {{ border-bottom: none; }}
  .flag {{ flex: 0 0 64px; width: 64px; height: 43px; line-height: 0; }}
  .flag svg {{ width: 64px; height: 43px; display: block;
              border: 1px solid #cfd4da; border-radius: 3px;
              box-shadow: 0 1px 2px rgba(0,0,0,.18); }}
  .text {{ flex: 1 1 auto; }}
  .endo {{ font-size: 17px; font-weight: 700; color: #1f4e4e; line-height: 1.2; }}
  .stmt {{ font-size: 15.5px; color: #20262e; line-height: 1.3; margin-top: 2px; }}
</style></head>
<body>
  <header>
    <h1>&#127760;&nbsp;Machine Translation Service Available</h1>
    <p class="sub">Find your language below &mdash; we can translate to and from each of these.</p>
  </header>
{rows}
</body></html>"""
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(doc)


def chrome_candidates():
    if os.environ.get("CHROME"):
        yield os.environ["CHROME"]
    # Playwright-managed Chromium (not snap-confined) — prefer newest.
    for p in sorted(glob.glob(os.path.expanduser(
            "~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")), reverse=True):
        yield p
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p:
            yield p


def render_pdf():
    if os.path.exists(PDF_OUT):
        os.remove(PDF_OUT)
    flags = ["--headless=new", "--no-sandbox", "--disable-gpu",
             "--disable-dev-shm-usage", "--no-pdf-header-footer"]
    for chrome in chrome_candidates():
        try:
            subprocess.run([chrome, *flags, f"--print-to-pdf={PDF_OUT}",
                            "file://" + HTML_OUT],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=120)
        except Exception:
            continue
        if os.path.exists(PDF_OUT) and os.path.getsize(PDF_OUT) > 10000:
            return chrome
    return None


def main():
    build_html()
    print("wrote", HTML_OUT)
    chrome = render_pdf()
    if not chrome:
        print("No working Chromium found. Set CHROME=/path/to/chrome, or render "
              "manually:\n  chromium --headless=new --no-pdf-header-footer "
              f"--print-to-pdf={PDF_OUT} file://{HTML_OUT}", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {PDF_OUT}  (via {chrome})")
    if shutil.which("pdfinfo"):
        pages = subprocess.run(["pdfinfo", PDF_OUT], capture_output=True, text=True)
        for line in pages.stdout.splitlines():
            if line.startswith("Pages:"):
                n = line.split()[1]
                print(f"pages: {n}")
                if n != "1":
                    print("WARNING: sign is not one page — adjust CSS sizes.", file=sys.stderr)


if __name__ == "__main__":
    main()
