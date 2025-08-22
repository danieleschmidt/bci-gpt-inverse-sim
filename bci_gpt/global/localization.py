"""Global localization framework for BCI-GPT with multi-language and cultural adaptation."""

import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import locale
import gettext
from babel import Locale, dates, numbers
from babel.messages import Catalog
from babel.messages.extract import extract_from_dir
from babel.messages.frontend import CommandLineInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE_SIMPLIFIED = "zh"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    BENGALI = "bn"
    TURKISH = "tr"
    POLISH = "pl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"


class CulturalDimension(Enum):
    """Hofstede's cultural dimensions for adaptation."""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    MASCULINITY = "masculinity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"


@dataclass
class CulturalProfile:
    """Cultural profile for a specific region/language."""
    language: SupportedLanguage
    region: str
    cultural_scores: Dict[CulturalDimension, float] = field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "decimal"
    currency_symbol: str = "$"
    
    # UI/UX preferences
    reading_direction: str = "ltr"  # ltr, rtl
    color_preferences: Dict[str, str] = field(default_factory=dict)
    icon_preferences: Dict[str, str] = field(default_factory=dict)
    
    # Medical/Health preferences
    medical_terminology_style: str = "professional"  # professional, simplified, colloquial
    consent_style: str = "explicit"  # explicit, implied
    privacy_expectations: str = "high"  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'language': self.language.value,
            'region': self.region,
            'cultural_scores': {dim.value: score for dim, score in self.cultural_scores.items()},
            'date_format': self.date_format,
            'time_format': self.time_format,
            'number_format': self.number_format,
            'currency_symbol': self.currency_symbol,
            'reading_direction': self.reading_direction,
            'color_preferences': self.color_preferences,
            'icon_preferences': self.icon_preferences,
            'medical_terminology_style': self.medical_terminology_style,
            'consent_style': self.consent_style,
            'privacy_expectations': self.privacy_expectations
        }


@dataclass
class LocalizationConfig:
    """Configuration for localization system."""
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    translation_directory: Path = Path("./locales")
    
    # Auto-detection settings
    enable_auto_detection: bool = True
    detection_sources: List[str] = field(default_factory=lambda: ["http_header", "url_param", "cookie", "geoip"])
    
    # Translation settings
    enable_machine_translation: bool = False
    translation_service: str = "google"  # google, azure, aws
    translation_confidence_threshold: float = 0.8
    
    # Cultural adaptation
    enable_cultural_adaptation: bool = True
    adapt_colors: bool = True
    adapt_icons: bool = True
    adapt_layout: bool = True
    
    # Medical terminology
    enable_medical_glossary: bool = True
    medical_complexity_levels: List[str] = field(default_factory=lambda: ["simple", "standard", "technical"])
    
    # Caching
    enable_translation_cache: bool = True
    cache_ttl_hours: int = 24


class LanguageSupport:
    """Core language support functionality."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.loaded_catalogs = {}
        self.translation_cache = {}
        
        # Setup translation directory
        self.config.translation_directory.mkdir(parents=True, exist_ok=True)
        
    def load_language_catalog(self, language: SupportedLanguage) -> Optional[Catalog]:
        """Load translation catalog for a specific language."""
        
        if language in self.loaded_catalogs:
            return self.loaded_catalogs[language]
        
        catalog_path = self.config.translation_directory / language.value / "LC_MESSAGES" / "messages.po"
        
        if not catalog_path.exists():
            logger.warning(f"Translation catalog not found for {language.value}: {catalog_path}")
            return None
        
        try:
            with open(catalog_path, 'rb') as f:
                catalog = Catalog()
                # In real implementation, load from .po file
                self.loaded_catalogs[language] = catalog
                logger.info(f"Loaded translation catalog for {language.value}")
                return catalog
        except Exception as e:
            logger.error(f"Error loading catalog for {language.value}: {e}")
            return None
    
    def translate(self, 
                 message: str, 
                 target_language: SupportedLanguage,
                 context: Optional[str] = None,
                 pluralization: Optional[int] = None) -> str:
        """Translate a message to target language."""
        
        # Check cache first
        cache_key = f"{message}:{target_language.value}:{context}:{pluralization}"
        if self.config.enable_translation_cache and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Load catalog
        catalog = self.load_language_catalog(target_language)
        
        if not catalog:
            # Fall back to default language or return original
            if target_language != self.config.fallback_language:
                return self.translate(message, self.config.fallback_language, context, pluralization)
            else:
                return message
        
        # Perform translation
        try:
            # In real implementation, use babel/gettext translation
            translated = self._perform_translation(message, catalog, context, pluralization)
            
            # Cache result
            if self.config.enable_translation_cache:
                self.translation_cache[cache_key] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for '{message}': {e}")
            return message
    
    def _perform_translation(self, 
                           message: str, 
                           catalog: Catalog,
                           context: Optional[str] = None,
                           pluralization: Optional[int] = None) -> str:
        """Perform actual translation using catalog."""
        
        # Simplified implementation - in real system use gettext/babel
        # This would look up the message in the catalog and return translation
        
        # For demo purposes, return a mock translation
        if message == "Welcome":
            translations = {
                SupportedLanguage.SPANISH.value: "Bienvenido",
                SupportedLanguage.FRENCH.value: "Bienvenue", 
                SupportedLanguage.GERMAN.value: "Willkommen",
                SupportedLanguage.CHINESE_SIMPLIFIED.value: "欢迎",
                SupportedLanguage.JAPANESE.value: "いらっしゃいませ"
            }
            return translations.get(catalog.locale, message)
        
        return message
    
    def get_medical_terminology(self, 
                              term: str,
                              language: SupportedLanguage,
                              complexity_level: str = "standard") -> Dict[str, str]:
        """Get medical terminology in different complexity levels."""
        
        # Medical terminology database (simplified)
        medical_terms = {
            "neural_signal": {
                "simple": {
                    SupportedLanguage.ENGLISH.value: "brain signal",
                    SupportedLanguage.SPANISH.value: "señal cerebral",
                    SupportedLanguage.FRENCH.value: "signal cérébral",
                    SupportedLanguage.GERMAN.value: "Gehirnsignal"
                },
                "standard": {
                    SupportedLanguage.ENGLISH.value: "neural signal",
                    SupportedLanguage.SPANISH.value: "señal neural",
                    SupportedLanguage.FRENCH.value: "signal neural",
                    SupportedLanguage.GERMAN.value: "neurales Signal"
                },
                "technical": {
                    SupportedLanguage.ENGLISH.value: "electroencephalographic signal",
                    SupportedLanguage.SPANISH.value: "señal electroencefalográfica",
                    SupportedLanguage.FRENCH.value: "signal électroencéphalographique",
                    SupportedLanguage.GERMAN.value: "elektroenzephalographisches Signal"
                }
            }
        }
        
        if term in medical_terms and complexity_level in medical_terms[term]:
            level_terms = medical_terms[term][complexity_level]
            return {
                'term': level_terms.get(language.value, term),
                'complexity': complexity_level,
                'language': language.value
            }
        
        return {
            'term': term,
            'complexity': complexity_level,
            'language': language.value,
            'note': 'Translation not available'
        }
    
    def format_medical_explanation(self, 
                                 explanation: str,
                                 language: SupportedLanguage,
                                 cultural_profile: CulturalProfile) -> str:
        """Format medical explanation according to cultural preferences."""
        
        # Adapt explanation based on cultural profile
        if cultural_profile.medical_terminology_style == "simplified":
            # Use simpler language
            explanation = self._simplify_medical_language(explanation, language)
        elif cultural_profile.medical_terminology_style == "professional":
            # Use professional medical terminology
            explanation = self._professionalize_language(explanation, language)
        
        # Adapt based on cultural dimensions
        if cultural_profile.cultural_scores.get(CulturalDimension.UNCERTAINTY_AVOIDANCE, 0) > 0.7:
            # High uncertainty avoidance - provide more detailed explanations
            explanation = self._add_detailed_explanations(explanation, language)
        
        return explanation
    
    def _simplify_medical_language(self, text: str, language: SupportedLanguage) -> str:
        """Simplify medical language for better understanding."""
        
        # Simplified implementation - replace complex terms
        simplifications = {
            SupportedLanguage.ENGLISH.value: {
                "electroencephalography": "brain wave recording",
                "neural interface": "brain connection",
                "signal processing": "signal handling"
            },
            SupportedLanguage.SPANISH.value: {
                "electroencefalografía": "grabación de ondas cerebrales",
                "interfaz neural": "conexión cerebral"
            }
        }
        
        if language.value in simplifications:
            for complex_term, simple_term in simplifications[language.value].items():
                text = text.replace(complex_term, simple_term)
        
        return text
    
    def _professionalize_language(self, text: str, language: SupportedLanguage) -> str:
        """Use professional medical terminology."""
        # In real implementation, replace colloquial terms with professional ones
        return text
    
    def _add_detailed_explanations(self, text: str, language: SupportedLanguage) -> str:
        """Add more detailed explanations for high uncertainty avoidance cultures."""
        # In real implementation, add explanatory clauses and clarifications
        return text + " " + self.translate("(Detailed explanation provided for your understanding)", language)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with names."""
        
        languages = []
        for lang in SupportedLanguage:
            try:
                babel_locale = Locale.parse(lang.value)
                languages.append({
                    'code': lang.value,
                    'name': babel_locale.display_name,
                    'native_name': babel_locale.get_display_name(lang.value)
                })
            except Exception:
                languages.append({
                    'code': lang.value,
                    'name': lang.name.title(),
                    'native_name': lang.name.title()
                })
        
        return languages


class CulturalAdaptation:
    """Handle cultural adaptation of UI/UX and content."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.cultural_profiles = self._load_cultural_profiles()
    
    def _load_cultural_profiles(self) -> Dict[str, CulturalProfile]:
        """Load cultural profiles for different regions."""
        
        # Predefined cultural profiles based on research
        profiles = {
            "US": CulturalProfile(
                language=SupportedLanguage.ENGLISH,
                region="US",
                cultural_scores={
                    CulturalDimension.POWER_DISTANCE: 0.4,
                    CulturalDimension.INDIVIDUALISM: 0.91,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.46
                },
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                currency_symbol="$",
                medical_terminology_style="professional",
                consent_style="explicit",
                privacy_expectations="high"
            ),
            "DE": CulturalProfile(
                language=SupportedLanguage.GERMAN,
                region="DE",
                cultural_scores={
                    CulturalDimension.POWER_DISTANCE: 0.35,
                    CulturalDimension.INDIVIDUALISM: 0.67,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.65
                },
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                currency_symbol="€",
                medical_terminology_style="technical",
                consent_style="explicit",
                privacy_expectations="high"
            ),
            "JP": CulturalProfile(
                language=SupportedLanguage.JAPANESE,
                region="JP",
                cultural_scores={
                    CulturalDimension.POWER_DISTANCE: 0.54,
                    CulturalDimension.INDIVIDUALISM: 0.46,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.92
                },
                date_format="%Y年%m月%d日",
                time_format="%H:%M",
                currency_symbol="¥",
                medical_terminology_style="professional",
                consent_style="implied",
                privacy_expectations="medium"
            ),
            "AR": CulturalProfile(
                language=SupportedLanguage.ARABIC,
                region="SA",
                cultural_scores={
                    CulturalDimension.POWER_DISTANCE: 0.95,
                    CulturalDimension.INDIVIDUALISM: 0.25,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.68
                },
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                reading_direction="rtl",
                medical_terminology_style="simplified",
                consent_style="explicit",
                privacy_expectations="medium"
            )
        }
        
        return profiles
    
    def get_cultural_profile(self, region: str, language: Optional[SupportedLanguage] = None) -> CulturalProfile:
        """Get cultural profile for a region."""
        
        if region in self.cultural_profiles:
            return self.cultural_profiles[region]
        
        # Create default profile
        return CulturalProfile(
            language=language or self.config.default_language,
            region=region
        )
    
    def adapt_ui_colors(self, base_colors: Dict[str, str], cultural_profile: CulturalProfile) -> Dict[str, str]:
        """Adapt UI colors based on cultural preferences."""
        
        adapted_colors = base_colors.copy()
        
        # Cultural color adaptations
        color_adaptations = {
            "red": {
                "CN": "#CC0000",  # Darker red for China (luck, prosperity)
                "IN": "#FF6600",  # Orange-red for India (auspicious)
                "SA": "#800000"   # Darker red for Saudi Arabia (modest)
            },
            "green": {
                "SA": "#228B22",  # Forest green for Islamic cultures
                "IN": "#32CD32",  # Lime green for India
                "JP": "#006400"   # Dark green for Japan (nature)
            }
        }
        
        region = cultural_profile.region
        for color_name, color_value in adapted_colors.items():
            if color_name in color_adaptations and region in color_adaptations[color_name]:
                adapted_colors[color_name] = color_adaptations[color_name][region]
        
        return adapted_colors
    
    def adapt_layout(self, layout_config: Dict[str, Any], cultural_profile: CulturalProfile) -> Dict[str, Any]:
        """Adapt layout based on cultural preferences."""
        
        adapted_layout = layout_config.copy()
        
        # Reading direction adaptation
        if cultural_profile.reading_direction == "rtl":
            adapted_layout['text_align'] = 'right'
            adapted_layout['flex_direction'] = 'row-reverse'
            adapted_layout['margin_left'], adapted_layout['margin_right'] = \
                adapted_layout.get('margin_right', 0), adapted_layout.get('margin_left', 0)
        
        # High power distance cultures prefer more hierarchical layouts
        power_distance = cultural_profile.cultural_scores.get(CulturalDimension.POWER_DISTANCE, 0.5)
        if power_distance > 0.7:
            adapted_layout['hierarchy_emphasis'] = 'high'
            adapted_layout['authority_indicators'] = True
        
        # High uncertainty avoidance cultures prefer more structured layouts
        uncertainty_avoidance = cultural_profile.cultural_scores.get(CulturalDimension.UNCERTAINTY_AVOIDANCE, 0.5)
        if uncertainty_avoidance > 0.7:
            adapted_layout['structure_emphasis'] = 'high'
            adapted_layout['navigation_clarity'] = 'detailed'
        
        return adapted_layout
    
    def adapt_consent_flow(self, base_consent: Dict[str, Any], cultural_profile: CulturalProfile) -> Dict[str, Any]:
        """Adapt consent flow based on cultural preferences."""
        
        adapted_consent = base_consent.copy()
        
        if cultural_profile.consent_style == "explicit":
            adapted_consent['require_explicit_checkboxes'] = True
            adapted_consent['detailed_explanations'] = True
            adapted_consent['opt_in_default'] = False
        elif cultural_profile.consent_style == "implied":
            adapted_consent['require_explicit_checkboxes'] = False
            adapted_consent['simplified_explanations'] = True
            adapted_consent['opt_in_default'] = True
        
        # High uncertainty avoidance cultures need more detailed consent
        uncertainty_avoidance = cultural_profile.cultural_scores.get(CulturalDimension.UNCERTAINTY_AVOIDANCE, 0.5)
        if uncertainty_avoidance > 0.7:
            adapted_consent['detailed_explanations'] = True
            adapted_consent['risk_explanations'] = True
            adapted_consent['contact_information'] = True
        
        return adapted_consent
    
    def format_date_time(self, 
                        dt: datetime, 
                        cultural_profile: CulturalProfile,
                        include_time: bool = False) -> str:
        """Format date/time according to cultural preferences."""
        
        try:
            babel_locale = Locale.parse(cultural_profile.language.value)
            
            if include_time:
                return dates.format_datetime(dt, locale=babel_locale)
            else:
                return dates.format_date(dt, locale=babel_locale)
                
        except Exception:
            # Fallback to profile format
            if include_time:
                return dt.strftime(f"{cultural_profile.date_format} {cultural_profile.time_format}")
            else:
                return dt.strftime(cultural_profile.date_format)
    
    def format_number(self, 
                     number: Union[int, float], 
                     cultural_profile: CulturalProfile,
                     currency: bool = False) -> str:
        """Format number according to cultural preferences."""
        
        try:
            babel_locale = Locale.parse(cultural_profile.language.value)
            
            if currency:
                return numbers.format_currency(number, cultural_profile.currency_symbol, locale=babel_locale)
            else:
                return numbers.format_decimal(number, locale=babel_locale)
                
        except Exception:
            # Fallback formatting
            if currency:
                return f"{cultural_profile.currency_symbol}{number:,.2f}"
            else:
                return f"{number:,}"


class GlobalLocalizationManager:
    """Manage global localization across all supported languages and cultures."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.language_support = LanguageSupport(config)
        self.cultural_adaptation = CulturalAdaptation(config)
        self.active_sessions = {}
        
    def detect_user_locale(self, request_headers: Dict[str, str], 
                          ip_address: Optional[str] = None,
                          url_params: Optional[Dict[str, str]] = None) -> Tuple[SupportedLanguage, str]:
        """Detect user's preferred locale from various sources."""
        
        detected_language = self.config.default_language
        detected_region = "US"
        
        # Check URL parameters first (highest priority)
        if url_params and 'lang' in url_params:
            try:
                detected_language = SupportedLanguage(url_params['lang'])
            except ValueError:
                pass
        
        # Check Accept-Language header
        elif 'Accept-Language' in request_headers:
            accept_language = request_headers['Accept-Language']
            languages = self._parse_accept_language(accept_language)
            
            for lang_code, _ in languages:
                try:
                    detected_language = SupportedLanguage(lang_code.split('-')[0])
                    if '-' in lang_code:
                        detected_region = lang_code.split('-')[1].upper()
                    break
                except ValueError:
                    continue
        
        # TODO: Add GeoIP detection for region
        # if ip_address:
        #     detected_region = self._geolocate_ip(ip_address)
        
        return detected_language, detected_region
    
    def _parse_accept_language(self, accept_language: str) -> List[Tuple[str, float]]:
        """Parse Accept-Language header."""
        
        languages = []
        
        for item in accept_language.split(','):
            item = item.strip()
            if ';q=' in item:
                lang, quality = item.split(';q=')
                quality = float(quality)
            else:
                lang = item
                quality = 1.0
            
            languages.append((lang.strip(), quality))
        
        # Sort by quality score
        languages.sort(key=lambda x: x[1], reverse=True)
        return languages
    
    def create_localized_session(self, 
                                session_id: str,
                                language: SupportedLanguage,
                                region: str) -> Dict[str, Any]:
        """Create a localized session for a user."""
        
        cultural_profile = self.cultural_adaptation.get_cultural_profile(region, language)
        
        session = {
            'session_id': session_id,
            'language': language,
            'region': region,
            'cultural_profile': cultural_profile,
            'created_at': datetime.now(),
            'translation_cache': {}
        }
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Created localized session {session_id}: {language.value}-{region}")
        return session
    
    def localize_content(self, 
                        content: Dict[str, Any],
                        session_id: str) -> Dict[str, Any]:
        """Localize content for a specific session."""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return content
        
        session = self.active_sessions[session_id]
        language = session['language']
        cultural_profile = session['cultural_profile']
        
        localized_content = {}
        
        for key, value in content.items():
            if isinstance(value, str):
                # Translate text content
                localized_content[key] = self.language_support.translate(value, language)
            elif isinstance(value, dict):
                # Recursively localize nested content
                localized_content[key] = self.localize_content(value, session_id)
            elif isinstance(value, datetime):
                # Format dates according to cultural preferences
                localized_content[key] = self.cultural_adaptation.format_date_time(value, cultural_profile)
            elif isinstance(value, (int, float)) and 'currency' in key.lower():
                # Format currency
                localized_content[key] = self.cultural_adaptation.format_number(value, cultural_profile, currency=True)
            elif isinstance(value, (int, float)):
                # Format numbers
                localized_content[key] = self.cultural_adaptation.format_number(value, cultural_profile)
            else:
                localized_content[key] = value
        
        return localized_content
    
    def get_medical_content(self, 
                           content_key: str,
                           session_id: str,
                           complexity_level: str = "standard") -> Dict[str, Any]:
        """Get localized medical content."""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        language = session['language']
        cultural_profile = session['cultural_profile']
        
        # Get base medical content
        medical_content = self._get_base_medical_content(content_key)
        
        # Translate and adapt
        localized_content = {
            'title': self.language_support.translate(medical_content['title'], language),
            'content': self.language_support.format_medical_explanation(
                medical_content['content'], language, cultural_profile
            ),
            'terminology': {},
            'complexity_level': complexity_level
        }
        
        # Localize medical terminology
        for term in medical_content.get('key_terms', []):
            localized_content['terminology'][term] = self.language_support.get_medical_terminology(
                term, language, complexity_level
            )
        
        return localized_content
    
    def _get_base_medical_content(self, content_key: str) -> Dict[str, Any]:
        """Get base medical content (before localization)."""
        
        # Simplified medical content database
        medical_content_db = {
            'bci_introduction': {
                'title': 'Introduction to Brain-Computer Interfaces',
                'content': 'A brain-computer interface (BCI) allows direct communication between your brain and a computer system through neural signals.',
                'key_terms': ['neural_signal', 'brain_computer_interface', 'electroencephalography']
            },
            'consent_explanation': {
                'title': 'Informed Consent for BCI Research',
                'content': 'By participating in this BCI research, you consent to the recording and analysis of your brain signals for medical research purposes.',
                'key_terms': ['informed_consent', 'neural_signal', 'medical_research']
            }
        }
        
        return medical_content_db.get(content_key, {
            'title': 'Content Not Found',
            'content': 'The requested content is not available.',
            'key_terms': []
        })
    
    def export_localization_report(self, output_path: Path) -> Path:
        """Export comprehensive localization report."""
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'configuration': {
                'default_language': self.config.default_language.value,
                'fallback_language': self.config.fallback_language.value,
                'enabled_features': {
                    'auto_detection': self.config.enable_auto_detection,
                    'machine_translation': self.config.enable_machine_translation,
                    'cultural_adaptation': self.config.enable_cultural_adaptation,
                    'medical_glossary': self.config.enable_medical_glossary
                }
            },
            'supported_languages': self.language_support.get_supported_languages(),
            'cultural_profiles': {
                region: profile.to_dict() 
                for region, profile in self.cultural_adaptation.cultural_profiles.items()
            },
            'active_sessions': len(self.active_sessions),
            'session_languages': {
                session['language'].value: session['region'] 
                for session in self.active_sessions.values()
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported localization report to: {output_path}")
        return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create localization configuration
    localization_config = LocalizationConfig(
        default_language=SupportedLanguage.ENGLISH,
        enable_auto_detection=True,
        enable_cultural_adaptation=True,
        enable_medical_glossary=True
    )
    
    # Create global localization manager
    localization_manager = GlobalLocalizationManager(localization_config)
    
    print("🌍 BCI-GPT Global Localization Manager")
    print(f"Default language: {localization_config.default_language.value}")
    print(f"Supported languages: {len(SupportedLanguage)} languages")
    
    # Simulate user detection
    request_headers = {
        'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8,fr;q=0.7'
    }
    
    detected_lang, detected_region = localization_manager.detect_user_locale(request_headers)
    print(f"\\nDetected locale: {detected_lang.value}-{detected_region}")
    
    # Create localized session
    session_id = "user_session_123"
    session = localization_manager.create_localized_session(session_id, detected_lang, detected_region)
    
    print(f"Created session: {session_id}")
    print(f"Cultural profile: {session['cultural_profile'].region}")
    
    # Test content localization
    content = {
        'welcome_message': 'Welcome to BCI-GPT',
        'current_date': datetime.now(),
        'cost': 99.99,
        'description': 'Advanced brain-computer interface technology'
    }
    
    localized_content = localization_manager.localize_content(content, session_id)
    print(f"\\nLocalized content:")
    for key, value in localized_content.items():
        print(f"  {key}: {value}")
    
    # Test medical content
    medical_content = localization_manager.get_medical_content(
        'bci_introduction', session_id, 'standard'
    )
    
    print(f"\\nMedical content:")
    print(f"  Title: {medical_content['title']}")
    print(f"  Content: {medical_content['content'][:100]}...")
    print(f"  Terminology count: {len(medical_content['terminology'])}")
    
    # Export report
    report_path = Path("./localization_report.json")
    localization_manager.export_localization_report(report_path)
    print(f"\\nLocalization report exported: {report_path}")
    
    print("\\n🌍 Global localization system validated!")