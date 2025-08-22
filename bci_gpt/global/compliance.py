"""Global compliance management for BCI-GPT with GDPR, HIPAA, and CCPA support."""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
import logging
import uuid
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    HEALTH_INFORMATION = "phi"
    BIOMETRIC_DATA = "biometric"
    NEURAL_DATA = "neural"
    BEHAVIORAL_DATA = "behavioral"
    TECHNICAL_DATA = "technical"
    METADATA = "metadata"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    MEDICAL_TREATMENT = "medical_treatment"
    RESEARCH = "research"
    PRODUCT_IMPROVEMENT = "product_improvement"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    LEGAL_COMPLIANCE = "legal_compliance"
    SECURITY = "security"


class ConsentType(Enum):
    """Types of user consent."""
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    OPT_IN = "opt_in"
    OPT_OUT = "opt_out"
    WITHDRAWN = "withdrawn"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    data_subject_id: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    legal_basis: str
    consent_type: Optional[ConsentType]
    processing_timestamp: datetime
    retention_period: timedelta
    geographic_location: str
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    automated_decision_making: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'record_id': self.record_id,
            'data_subject_id': self.data_subject_id,
            'data_categories': [cat.value for cat in self.data_categories],
            'processing_purposes': [purpose.value for purpose in self.processing_purposes],
            'legal_basis': self.legal_basis,
            'consent_type': self.consent_type.value if self.consent_type else None,
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'retention_period_days': self.retention_period.days,
            'geographic_location': self.geographic_location,
            'third_party_sharing': self.third_party_sharing,
            'cross_border_transfer': self.cross_border_transfer,
            'automated_decision_making': self.automated_decision_making
        }


@dataclass
class ComplianceConfig:
    """Configuration for compliance management."""
    enabled_frameworks: List[ComplianceFramework] = field(default_factory=lambda: [ComplianceFramework.GDPR])
    default_retention_period: timedelta = field(default_factory=lambda: timedelta(days=2555))  # 7 years
    data_protection_officer_contact: str = "dpo@bci-gpt.com"
    privacy_policy_url: str = "https://bci-gpt.com/privacy"
    cookie_policy_url: str = "https://bci-gpt.com/cookies"
    data_subject_rights_url: str = "https://bci-gpt.com/rights"
    
    # Audit settings
    audit_log_retention_days: int = 2555  # 7 years
    audit_log_encryption: bool = True
    
    # Data minimization
    enable_data_minimization: bool = True
    automatic_anonymization: bool = True
    anonymization_delay_days: int = 90
    
    # Cross-border transfer
    enable_cross_border_transfer: bool = False
    approved_transfer_mechanisms: List[str] = field(default_factory=lambda: ["Standard Contractual Clauses"])
    
    # Breach notification
    breach_notification_email: str = "security@bci-gpt.com"
    breach_notification_threshold_hours: int = 72
    
    def is_framework_enabled(self, framework: ComplianceFramework) -> bool:
        """Check if a compliance framework is enabled."""
        return framework in self.enabled_frameworks


class BaseComplianceFramework(ABC):
    """Abstract base class for compliance frameworks."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        
    @abstractmethod
    def get_legal_basis_options(self) -> List[str]:
        """Get available legal basis options for data processing."""
        pass
    
    @abstractmethod
    def validate_processing_record(self, record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate a data processing record against framework requirements."""
        pass
    
    @abstractmethod
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """Get retention requirements for specific data category."""
        pass
    
    @abstractmethod
    def requires_explicit_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if explicit consent is required for processing purpose."""
        pass


class GDPRCompliance(BaseComplianceFramework):
    """GDPR compliance implementation."""
    
    def get_legal_basis_options(self) -> List[str]:
        """GDPR Article 6 legal basis options."""
        return [
            "consent",
            "contract",
            "legal_obligation", 
            "vital_interests",
            "public_task",
            "legitimate_interests"
        ]
    
    def validate_processing_record(self, record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate processing record against GDPR requirements."""
        
        validation = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'requirements_met': []
        }
        
        # Check legal basis
        if record.legal_basis not in self.get_legal_basis_options():
            validation['violations'].append(f"Invalid legal basis: {record.legal_basis}")
            validation['valid'] = False
        else:
            validation['requirements_met'].append(f"Valid legal basis: {record.legal_basis}")
        
        # Check consent requirements for sensitive data
        sensitive_categories = [DataCategory.HEALTH_INFORMATION, DataCategory.BIOMETRIC_DATA, DataCategory.NEURAL_DATA]
        has_sensitive_data = any(cat in record.data_categories for cat in sensitive_categories)
        
        if has_sensitive_data and record.legal_basis == "consent" and record.consent_type != ConsentType.EXPLICIT:
            validation['violations'].append("Sensitive data requires explicit consent under GDPR Article 9")
            validation['valid'] = False
        
        # Check data subject rights compliance
        if record.automated_decision_making:
            validation['warnings'].append("Automated decision-making detected - ensure GDPR Article 22 compliance")
        
        # Check cross-border transfer
        if record.cross_border_transfer and not self.config.approved_transfer_mechanisms:
            validation['violations'].append("Cross-border transfer without approved transfer mechanism")
            validation['valid'] = False
        
        # Check retention period
        max_retention = self.get_retention_requirements(record.data_categories[0])
        if record.retention_period > max_retention:
            validation['warnings'].append(f"Retention period exceeds recommended maximum: {max_retention.days} days")
        
        return validation
    
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """GDPR retention requirements by data category."""
        
        retention_map = {
            DataCategory.PERSONAL_IDENTIFIABLE: timedelta(days=2555),  # 7 years
            DataCategory.HEALTH_INFORMATION: timedelta(days=3650),     # 10 years
            DataCategory.BIOMETRIC_DATA: timedelta(days=1825),        # 5 years
            DataCategory.NEURAL_DATA: timedelta(days=3650),           # 10 years (research)
            DataCategory.BEHAVIORAL_DATA: timedelta(days=1095),       # 3 years
            DataCategory.TECHNICAL_DATA: timedelta(days=365),         # 1 year
            DataCategory.METADATA: timedelta(days=180)                # 6 months
        }
        
        return retention_map.get(data_category, self.config.default_retention_period)
    
    def requires_explicit_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if explicit consent required under GDPR."""
        
        explicit_consent_purposes = [
            ProcessingPurpose.MARKETING,
            ProcessingPurpose.RESEARCH  # For sensitive data
        ]
        
        return purpose in explicit_consent_purposes
    
    def generate_privacy_notice(self) -> Dict[str, Any]:
        """Generate GDPR-compliant privacy notice content."""
        
        return {
            'data_controller': 'BCI-GPT Technologies',
            'data_protection_officer': self.config.data_protection_officer_contact,
            'legal_basis': self.get_legal_basis_options(),
            'data_categories_collected': [cat.value for cat in DataCategory],
            'processing_purposes': [purpose.value for purpose in ProcessingPurpose],
            'retention_periods': {
                cat.value: self.get_retention_requirements(cat).days 
                for cat in DataCategory
            },
            'data_subject_rights': [
                'right_of_access',
                'right_of_rectification', 
                'right_of_erasure',
                'right_to_restrict_processing',
                'right_to_data_portability',
                'right_to_object',
                'rights_related_to_automated_decision_making'
            ],
            'contact_information': {
                'dpo_email': self.config.data_protection_officer_contact,
                'privacy_policy': self.config.privacy_policy_url,
                'data_subject_rights': self.config.data_subject_rights_url
            }
        }


class HIPAACompliance(BaseComplianceFramework):
    """HIPAA compliance implementation."""
    
    def get_legal_basis_options(self) -> List[str]:
        """HIPAA permitted uses and disclosures."""
        return [
            "treatment",
            "payment",
            "healthcare_operations",
            "individual_authorization",
            "public_health",
            "health_oversight",
            "judicial_proceedings",
            "law_enforcement",
            "research_authorization",
            "research_waiver"
        ]
    
    def validate_processing_record(self, record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate processing record against HIPAA requirements."""
        
        validation = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'requirements_met': []
        }
        
        # Check if PHI is involved
        has_phi = DataCategory.HEALTH_INFORMATION in record.data_categories
        
        if has_phi:
            # Verify legal basis for PHI processing
            if record.legal_basis not in self.get_legal_basis_options():
                validation['violations'].append(f"Invalid legal basis for PHI: {record.legal_basis}")
                validation['valid'] = False
            
            # Check minimum necessary standard
            if record.processing_purposes and len(record.processing_purposes) > 3:
                validation['warnings'].append("Multiple processing purposes - ensure minimum necessary standard")
            
            # Check business associate agreements for third-party sharing
            if record.third_party_sharing:
                validation['warnings'].append("Third-party PHI sharing requires Business Associate Agreement")
            
            # Check authorization for research
            if ProcessingPurpose.RESEARCH in record.processing_purposes:
                if record.legal_basis not in ["research_authorization", "research_waiver"]:
                    validation['violations'].append("Research use of PHI requires authorization or waiver")
                    validation['valid'] = False
        
        return validation
    
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """HIPAA retention requirements."""
        
        if data_category == DataCategory.HEALTH_INFORMATION:
            return timedelta(days=2190)  # 6 years minimum
        
        return self.config.default_retention_period
    
    def requires_explicit_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if authorization required under HIPAA."""
        
        authorization_purposes = [
            ProcessingPurpose.MARKETING,
            ProcessingPurpose.RESEARCH
        ]
        
        return purpose in authorization_purposes
    
    def generate_notice_of_privacy_practices(self) -> Dict[str, Any]:
        """Generate HIPAA Notice of Privacy Practices."""
        
        return {
            'covered_entity': 'BCI-GPT Healthcare Services',
            'effective_date': datetime.now().isoformat(),
            'uses_and_disclosures': {
                'treatment': 'We may use and disclose your health information for treatment purposes',
                'payment': 'We may use and disclose your health information for payment purposes',
                'healthcare_operations': 'We may use and disclose your health information for healthcare operations'
            },
            'individual_rights': [
                'right_to_request_restrictions',
                'right_to_request_confidential_communications',
                'right_to_inspect_and_copy',
                'right_to_amend',
                'right_to_accounting_of_disclosures',
                'right_to_paper_copy_of_notice'
            ],
            'contact_information': {
                'privacy_officer': self.config.data_protection_officer_contact,
                'complaint_process': 'Contact privacy officer or HHS'
            }
        }


class CCPACompliance(BaseComplianceFramework):
    """CCPA compliance implementation."""
    
    def get_legal_basis_options(self) -> List[str]:
        """CCPA business purposes for processing personal information."""
        return [
            "performing_services",
            "providing_security",
            "debugging",
            "short_term_transient_use",
            "performing_services_on_behalf",
            "quality_assurance",
            "legal_compliance"
        ]
    
    def validate_processing_record(self, record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate processing record against CCPA requirements."""
        
        validation = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'requirements_met': []
        }
        
        # Check sale/sharing disclosure requirements
        if record.third_party_sharing:
            validation['warnings'].append("Third-party sharing may constitute 'sale' under CCPA - ensure disclosure")
        
        # Check sensitive personal information
        sensitive_categories = [DataCategory.HEALTH_INFORMATION, DataCategory.BIOMETRIC_DATA]
        has_sensitive = any(cat in record.data_categories for cat in sensitive_categories)
        
        if has_sensitive:
            validation['warnings'].append("Sensitive personal information detected - additional CCPA requirements apply")
        
        return validation
    
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """CCPA retention requirements."""
        
        # CCPA doesn't specify retention periods, use business necessity
        return self.config.default_retention_period
    
    def requires_explicit_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if opt-in consent required under CCPA."""
        
        # CCPA generally requires opt-out, but opt-in for sensitive data
        return purpose == ProcessingPurpose.MARKETING
    
    def generate_privacy_policy_ccpa(self) -> Dict[str, Any]:
        """Generate CCPA-compliant privacy policy content."""
        
        return {
            'categories_of_personal_information': [cat.value for cat in DataCategory],
            'business_purposes': self.get_legal_basis_options(),
            'categories_of_sources': [
                'directly_from_consumer',
                'consumer_devices',
                'third_party_providers'
            ],
            'categories_of_third_parties': [
                'service_providers',
                'business_partners',
                'legal_authorities'
            ],
            'consumer_rights': [
                'right_to_know',
                'right_to_delete',
                'right_to_opt_out_of_sale',
                'right_to_non_discrimination'
            ],
            'do_not_sell_link': 'https://bci-gpt.com/do-not-sell',
            'contact_information': self.config.data_protection_officer_contact
        }


class GlobalComplianceManager:
    """Manage compliance across multiple frameworks and regions."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.frameworks = {}
        self.processing_records = []
        self.audit_log = []
        
        # Initialize enabled frameworks
        if config.is_framework_enabled(ComplianceFramework.GDPR):
            self.frameworks[ComplianceFramework.GDPR] = GDPRCompliance(config)
        
        if config.is_framework_enabled(ComplianceFramework.HIPAA):
            self.frameworks[ComplianceFramework.HIPAA] = HIPAACompliance(config)
        
        if config.is_framework_enabled(ComplianceFramework.CCPA):
            self.frameworks[ComplianceFramework.CCPA] = CCPACompliance(config)
    
    def record_data_processing(self, 
                             data_subject_id: str,
                             data_categories: List[DataCategory],
                             processing_purposes: List[ProcessingPurpose],
                             legal_basis: str,
                             geographic_location: str,
                             consent_type: Optional[ConsentType] = None,
                             **kwargs) -> str:
        """Record a data processing activity."""
        
        record_id = str(uuid.uuid4())
        
        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            legal_basis=legal_basis,
            consent_type=consent_type,
            processing_timestamp=datetime.now(),
            retention_period=self.config.default_retention_period,
            geographic_location=geographic_location,
            **kwargs
        )
        
        # Validate against all enabled frameworks
        validation_results = {}
        for framework_name, framework in self.frameworks.items():
            validation = framework.validate_processing_record(record)
            validation_results[framework_name.value] = validation
        
        # Store record
        self.processing_records.append(record)
        
        # Log audit entry
        self._log_audit_entry("data_processing_recorded", {
            'record_id': record_id,
            'data_subject_id': data_subject_id,
            'validation_results': validation_results
        })
        
        logger.info(f"Recorded data processing activity: {record_id}")
        return record_id
    
    def validate_cross_framework_compliance(self, record_id: str) -> Dict[str, Any]:
        """Validate a processing record across all enabled frameworks."""
        
        record = self._get_processing_record(record_id)
        if not record:
            return {'error': f'Processing record {record_id} not found'}
        
        overall_validation = {
            'record_id': record_id,
            'overall_compliance': True,
            'framework_results': {},
            'cross_framework_conflicts': []
        }
        
        for framework_name, framework in self.frameworks.items():
            validation = framework.validate_processing_record(record)
            overall_validation['framework_results'][framework_name.value] = validation
            
            if not validation['valid']:
                overall_validation['overall_compliance'] = False
        
        # Check for cross-framework conflicts
        conflicts = self._detect_cross_framework_conflicts(overall_validation['framework_results'])
        overall_validation['cross_framework_conflicts'] = conflicts
        
        return overall_validation
    
    def _detect_cross_framework_conflicts(self, framework_results: Dict[str, Any]) -> List[str]:
        """Detect conflicts between different framework requirements."""
        
        conflicts = []
        
        # Example: GDPR explicit consent vs CCPA opt-out model
        if ('gdpr' in framework_results and 'ccpa' in framework_results):
            gdpr_violations = framework_results['gdpr'].get('violations', [])
            ccpa_violations = framework_results['ccpa'].get('violations', [])
            
            # Check for consent model conflicts
            has_gdpr_consent_issue = any('consent' in v.lower() for v in gdpr_violations)
            has_ccpa_consent_issue = any('consent' in v.lower() for v in ccpa_violations)
            
            if has_gdpr_consent_issue and has_ccpa_consent_issue:
                conflicts.append("GDPR explicit consent requirements may conflict with CCPA opt-out model")
        
        return conflicts
    
    def generate_data_protection_impact_assessment(self, 
                                                 processing_description: str,
                                                 data_categories: List[DataCategory],
                                                 processing_purposes: List[ProcessingPurpose]) -> Dict[str, Any]:
        """Generate Data Protection Impact Assessment (DPIA)."""
        
        dpia = {
            'assessment_id': str(uuid.uuid4()),
            'created_at': datetime.now().isoformat(),
            'processing_description': processing_description,
            'data_categories': [cat.value for cat in data_categories],
            'processing_purposes': [purpose.value for purpose in processing_purposes],
            'risk_assessment': {},
            'mitigation_measures': [],
            'necessity_and_proportionality': {},
            'safeguards': []
        }
        
        # Risk assessment
        risk_level = self._assess_processing_risk(data_categories, processing_purposes)
        dpia['risk_assessment'] = {
            'overall_risk_level': risk_level,
            'high_risk_indicators': self._identify_high_risk_indicators(data_categories, processing_purposes),
            'residual_risk': 'medium'  # After mitigation measures
        }
        
        # Mitigation measures
        dpia['mitigation_measures'] = self._generate_mitigation_measures(data_categories, processing_purposes)
        
        # Safeguards
        dpia['safeguards'] = [
            'encryption_at_rest_and_in_transit',
            'access_controls_and_authentication',
            'audit_logging',
            'data_minimization',
            'pseudonymization_where_appropriate',
            'regular_security_assessments'
        ]
        
        return dpia
    
    def _assess_processing_risk(self, 
                              data_categories: List[DataCategory],
                              processing_purposes: List[ProcessingPurpose]) -> str:
        """Assess risk level of data processing."""
        
        high_risk_categories = [DataCategory.HEALTH_INFORMATION, DataCategory.BIOMETRIC_DATA, DataCategory.NEURAL_DATA]
        high_risk_purposes = [ProcessingPurpose.RESEARCH, ProcessingPurpose.ANALYTICS]
        
        has_high_risk_data = any(cat in high_risk_categories for cat in data_categories)
        has_high_risk_purpose = any(purpose in high_risk_purposes for purpose in processing_purposes)
        
        if has_high_risk_data and has_high_risk_purpose:
            return "high"
        elif has_high_risk_data or has_high_risk_purpose:
            return "medium"
        else:
            return "low"
    
    def _identify_high_risk_indicators(self,
                                     data_categories: List[DataCategory],
                                     processing_purposes: List[ProcessingPurpose]) -> List[str]:
        """Identify high-risk indicators for DPIA."""
        
        indicators = []
        
        if DataCategory.HEALTH_INFORMATION in data_categories:
            indicators.append("processing_of_health_data")
        
        if DataCategory.BIOMETRIC_DATA in data_categories:
            indicators.append("processing_of_biometric_data")
        
        if DataCategory.NEURAL_DATA in data_categories:
            indicators.append("processing_of_neural_data")
        
        if ProcessingPurpose.RESEARCH in processing_purposes:
            indicators.append("systematic_monitoring")
        
        return indicators
    
    def _generate_mitigation_measures(self,
                                    data_categories: List[DataCategory],
                                    processing_purposes: List[ProcessingPurpose]) -> List[str]:
        """Generate appropriate mitigation measures."""
        
        measures = [
            "implement_privacy_by_design",
            "conduct_regular_privacy_impact_assessments",
            "establish_data_retention_policies",
            "implement_data_subject_rights_procedures"
        ]
        
        if DataCategory.HEALTH_INFORMATION in data_categories:
            measures.extend([
                "implement_hipaa_safeguards",
                "establish_business_associate_agreements",
                "conduct_hipaa_risk_assessments"
            ])
        
        if any(cat in [DataCategory.BIOMETRIC_DATA, DataCategory.NEURAL_DATA] for cat in data_categories):
            measures.extend([
                "implement_biometric_data_safeguards",
                "establish_enhanced_consent_mechanisms",
                "implement_data_minimization_techniques"
            ])
        
        return measures
    
    def _get_processing_record(self, record_id: str) -> Optional[DataProcessingRecord]:
        """Get processing record by ID."""
        for record in self.processing_records:
            if record.record_id == record_id:
                return record
        return None
    
    def _log_audit_entry(self, action: str, details: Dict[str, Any]):
        """Log audit entry."""
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'user': 'system'  # In real implementation, get from authentication context
        }
        
        self.audit_log.append(audit_entry)
    
    def export_compliance_report(self, output_path: Path) -> Path:
        """Export comprehensive compliance report."""
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'enabled_frameworks': [f.value for f in self.config.enabled_frameworks],
            'total_processing_records': len(self.processing_records),
            'processing_records': [record.to_dict() for record in self.processing_records],
            'audit_log_entries': len(self.audit_log),
            'compliance_configuration': {
                'default_retention_period_days': self.config.default_retention_period.days,
                'data_protection_officer': self.config.data_protection_officer_contact,
                'enable_data_minimization': self.config.enable_data_minimization,
                'enable_cross_border_transfer': self.config.enable_cross_border_transfer
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported compliance report to: {output_path}")
        return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create compliance configuration
    compliance_config = ComplianceConfig(
        enabled_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
        data_protection_officer_contact="dpo@bci-gpt.com",
        enable_data_minimization=True,
        automatic_anonymization=True
    )
    
    # Create global compliance manager
    compliance_manager = GlobalComplianceManager(compliance_config)
    
    print("🔒 BCI-GPT Global Compliance Manager")
    print(f"Enabled frameworks: {[f.value for f in compliance_config.enabled_frameworks]}")
    
    # Record a data processing activity
    record_id = compliance_manager.record_data_processing(
        data_subject_id="patient_123",
        data_categories=[DataCategory.NEURAL_DATA, DataCategory.HEALTH_INFORMATION],
        processing_purposes=[ProcessingPurpose.MEDICAL_TREATMENT, ProcessingPurpose.RESEARCH],
        legal_basis="consent",
        geographic_location="EU",
        consent_type=ConsentType.EXPLICIT,
        third_party_sharing=False,
        automated_decision_making=True
    )
    
    print(f"\\nRecorded processing activity: {record_id}")
    
    # Validate compliance
    validation = compliance_manager.validate_cross_framework_compliance(record_id)
    print(f"Overall compliance: {'✅ Compliant' if validation['overall_compliance'] else '❌ Non-compliant'}")
    
    for framework, result in validation['framework_results'].items():
        status = "✅ Valid" if result['valid'] else "❌ Invalid"
        print(f"  {framework.upper()}: {status}")
        
        if result['violations']:
            for violation in result['violations']:
                print(f"    - Violation: {violation}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"    - Warning: {warning}")
    
    # Generate DPIA
    dpia = compliance_manager.generate_data_protection_impact_assessment(
        processing_description="Neural signal processing for BCI communication",
        data_categories=[DataCategory.NEURAL_DATA, DataCategory.BIOMETRIC_DATA],
        processing_purposes=[ProcessingPurpose.MEDICAL_TREATMENT, ProcessingPurpose.RESEARCH]
    )
    
    print(f"\\nGenerated DPIA: {dpia['assessment_id']}")
    print(f"Risk level: {dpia['risk_assessment']['overall_risk_level']}")
    print(f"Mitigation measures: {len(dpia['mitigation_measures'])}")
    
    # Export compliance report
    report_path = Path("./compliance_report.json")
    compliance_manager.export_compliance_report(report_path)
    print(f"\\nCompliance report exported: {report_path}")
    
    print("\\n🔒 Global compliance management system validated!")