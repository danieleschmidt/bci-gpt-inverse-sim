"""Internationalization (i18n) support for BCI-GPT."""

from .translator import Translator, get_translator
from .locales import SUPPORTED_LOCALES, DEFAULT_LOCALE

__all__ = ['Translator', 'get_translator', 'SUPPORTED_LOCALES', 'DEFAULT_LOCALE']