"""Supported locales and language configurations."""

import os
from typing import Dict, List
from pathlib import Path

# Default locale
DEFAULT_LOCALE = "en_US"

# Supported locales with their display names
SUPPORTED_LOCALES = {
    "en_US": "English (United States)",
    "en_GB": "English (United Kingdom)", 
    "zh_CN": "中文 (简体)",
    "zh_TW": "中文 (繁體)",
    "ja_JP": "日本語",
    "ko_KR": "한국어",
    "de_DE": "Deutsch",
    "fr_FR": "Français",
    "es_ES": "Español",
    "pt_BR": "Português (Brasil)",
    "ru_RU": "Русский",
    "ar_SA": "العربية",
    "hi_IN": "हिन्दी",
    "th_TH": "ไทย",
    "vi_VN": "Tiếng Việt"
}

# RTL (Right-to-Left) languages
RTL_LOCALES = {"ar_SA", "he_IL", "fa_IR"}

# Language to locale mapping
LANGUAGE_TO_LOCALE = {
    "en": "en_US",
    "zh": "zh_CN", 
    "ja": "ja_JP",
    "ko": "ko_KR",
    "de": "de_DE",
    "fr": "fr_FR",
    "es": "es_ES",
    "pt": "pt_BR",
    "ru": "ru_RU",
    "ar": "ar_SA",
    "hi": "hi_IN",
    "th": "th_TH",
    "vi": "vi_VN"
}

def get_locale_dir() -> Path:
    """Get the directory containing locale files."""
    return Path(__file__).parent / "locales"

def get_available_locales() -> List[str]:
    """Get list of available locales based on existing translation files."""
    locale_dir = get_locale_dir()
    if not locale_dir.exists():
        return [DEFAULT_LOCALE]
    
    available = []
    for locale_code in SUPPORTED_LOCALES.keys():
        locale_file = locale_dir / f"{locale_code}.json"
        if locale_file.exists():
            available.append(locale_code)
    
    # Always include default locale
    if DEFAULT_LOCALE not in available:
        available.append(DEFAULT_LOCALE)
    
    return sorted(available)

def detect_system_locale() -> str:
    """Detect system locale from environment variables."""
    # Try various environment variables
    for env_var in ['LC_ALL', 'LC_MESSAGES', 'LANG', 'LANGUAGE']:
        locale = os.environ.get(env_var)
        if locale:
            # Parse locale string (e.g., "en_US.UTF-8" -> "en_US")
            locale_code = locale.split('.')[0].replace('-', '_')
            
            # Check if we support this locale
            if locale_code in SUPPORTED_LOCALES:
                return locale_code
            
            # Try language part only (e.g., "en" from "en_US")
            lang_code = locale_code.split('_')[0]
            if lang_code in LANGUAGE_TO_LOCALE:
                return LANGUAGE_TO_LOCALE[lang_code]
    
    return DEFAULT_LOCALE

def is_rtl_locale(locale: str) -> bool:
    """Check if locale uses right-to-left text direction."""
    return locale in RTL_LOCALES

def get_locale_info(locale: str) -> Dict[str, str]:
    """Get information about a locale."""
    return {
        "code": locale,
        "name": SUPPORTED_LOCALES.get(locale, locale),
        "rtl": is_rtl_locale(locale),
        "language": locale.split('_')[0],
        "country": locale.split('_')[1] if '_' in locale else ""
    }