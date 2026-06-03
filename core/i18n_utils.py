"""轻量多语言翻译引擎"""

import json
from pathlib import Path

# ── 翻译缓存 ──
_translations: dict[str, str] = {}
_current_lang: str = "zh"

I18N_DIR = Path(__file__).parent.parent / "i18n"


def _load_translations(lang: str) -> dict[str, str]:
    """加载指定语言的翻译文件"""
    f = I18N_DIR / f"{lang}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def set_language(lang: str):
    """切换语言（zh/en）"""
    global _translations, _current_lang
    _current_lang = lang
    _translations = _load_translations(lang)


def get_language() -> str:
    return _current_lang


def _(text: str) -> str:
    """翻译函数：翻译文本（无参）"""
    if _current_lang == "zh":
        return text
    return _translations.get(text, text)


def _f(text: str, **kwargs) -> str:
    """翻译函数：含格式化参数，先翻译再格式化"""
    t = _(text)
    return t.format(**kwargs)
