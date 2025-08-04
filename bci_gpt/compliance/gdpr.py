"""GDPR compliance implementation for EEG data processing."""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class LegalBasis(Enum):
    """GDPR legal basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataCategory(Enum):
    """Categories of personal data."""
    BIOMETRIC = "biometric"  # EEG data
    HEALTH = "health"        # Health-related information
    IDENTITY = "identity"    # Personal identifiers
    DEMOGRAPHIC = "demographic"  # Age, gender, etc.

@dataclass
class DataSubject:
    """Data subject information."""
    subject_id: str
    consent_given: bool
    consent_date: Optional[datetime] = None
    consent_withdrawn: bool = False
    withdrawal_date: Optional[datetime] = None
    data_categories: List[DataCategory] = None
    legal_basis: LegalBasis = LegalBasis.CONSENT
    retention_period: Optional[int] = None  # days
    
    def __post_init__(self):
        if self.data_categories is None:
            self.data_categories = []

@dataclass 
class ProcessingActivity:
    """Processing activity record."""
    activity_id: str
    purpose: str
    data_categories: List[DataCategory]
    legal_basis: LegalBasis
    recipients: List[str]
    retention_period: int  # days
    security_measures: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None

class GDPRCompliance:
    """GDPR compliance manager for BCI-GPT."""
    
    def __init__(self, data_dir: Path):
        """Initialize GDPR compliance manager.
        
        Args:
            data_dir: Directory for storing compliance data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Compliance data storage
        self.subjects_file = self.data_dir / "data_subjects.json"
        self.activities_file = self.data_dir / "processing_activities.json"
        self.audit_log_file = self.data_dir / "audit_log.json"
        
        # Load existing data
        self.data_subjects: Dict[str, DataSubject] = self._load_data_subjects()
        self.processing_activities: Dict[str, ProcessingActivity] = self._load_processing_activities()
        
        # Default retention periods (days)
        self.default_retention = {
            DataCategory.BIOMETRIC: 2555,  # 7 years for research data
            DataCategory.HEALTH: 2555,     # 7 years for health data
            DataCategory.IDENTITY: 1095,   # 3 years for identifiers
            DataCategory.DEMOGRAPHIC: 1095  # 3 years for demographics
        }
        
    def _load_data_subjects(self) -> Dict[str, DataSubject]:
        """Load data subjects from storage."""
        if not self.subjects_file.exists():
            return {}
        
        try:
            with open(self.subjects_file, 'r') as f:
                data = json.load(f)
            
            subjects = {}
            for subject_id, subject_data in data.items():
                # Convert string dates back to datetime
                if subject_data.get('consent_date'):
                    subject_data['consent_date'] = datetime.fromisoformat(subject_data['consent_date'])
                if subject_data.get('withdrawal_date'):
                    subject_data['withdrawal_date'] = datetime.fromisoformat(subject_data['withdrawal_date'])
                
                # Convert string enums back to enums
                if subject_data.get('data_categories'):
                    subject_data['data_categories'] = [
                        DataCategory(cat) for cat in subject_data['data_categories']
                    ]
                if subject_data.get('legal_basis'):
                    subject_data['legal_basis'] = LegalBasis(subject_data['legal_basis'])
                
                subjects[subject_id] = DataSubject(**subject_data)
            
            return subjects
            
        except Exception as e:
            logger.error(f"Failed to load data subjects: {e}")
            return {}
    
    def _load_processing_activities(self) -> Dict[str, ProcessingActivity]:
        """Load processing activities from storage."""
        if not self.activities_file.exists():
            return {}
        
        try:
            with open(self.activities_file, 'r') as f:
                data = json.load(f)
            
            activities = {}
            for activity_id, activity_data in data.items():
                # Convert string dates back to datetime
                activity_data['created_at'] = datetime.fromisoformat(activity_data['created_at'])
                if activity_data.get('updated_at'):
                    activity_data['updated_at'] = datetime.fromisoformat(activity_data['updated_at'])
                
                # Convert string enums back to enums
                activity_data['data_categories'] = [
                    DataCategory(cat) for cat in activity_data['data_categories']
                ]
                activity_data['legal_basis'] = LegalBasis(activity_data['legal_basis'])
                
                activities[activity_id] = ProcessingActivity(**activity_data)
            
            return activities
            
        except Exception as e:
            logger.error(f"Failed to load processing activities: {e}")
            return {}
    
    def _save_data_subjects(self):
        """Save data subjects to storage."""
        try:
            data = {}
            for subject_id, subject in self.data_subjects.items():
                subject_dict = asdict(subject)
                
                # Convert datetime objects to ISO strings
                if subject_dict.get('consent_date'):
                    subject_dict['consent_date'] = subject.consent_date.isoformat()
                if subject_dict.get('withdrawal_date'):
                    subject_dict['withdrawal_date'] = subject.withdrawal_date.isoformat()
                
                # Convert enums to strings
                subject_dict['data_categories'] = [cat.value for cat in subject.data_categories]
                subject_dict['legal_basis'] = subject.legal_basis.value
                
                data[subject_id] = subject_dict
            
            with open(self.subjects_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save data subjects: {e}")
    
    def _save_processing_activities(self):
        """Save processing activities to storage."""
        try:
            data = {}
            for activity_id, activity in self.processing_activities.items():
                activity_dict = asdict(activity)
                
                # Convert datetime objects to ISO strings
                activity_dict['created_at'] = activity.created_at.isoformat()
                if activity_dict.get('updated_at'):
                    activity_dict['updated_at'] = activity.updated_at.isoformat()
                
                # Convert enums to strings
                activity_dict['data_categories'] = [cat.value for cat in activity.data_categories]
                activity_dict['legal_basis'] = activity.legal_basis.value
                
                data[activity_id] = activity_dict
            
            with open(self.activities_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save processing activities: {e}")
    
    def register_data_subject(self,
                            subject_id: str,
                            consent_given: bool,
                            data_categories: List[DataCategory],
                            legal_basis: LegalBasis = LegalBasis.CONSENT,
                            retention_period: Optional[int] = None) -> bool:
        """Register a new data subject.
        
        Args:
            subject_id: Unique identifier for data subject
            consent_given: Whether consent was given
            data_categories: Categories of data to be processed
            legal_basis: Legal basis for processing
            retention_period: Custom retention period in days
            
        Returns:
            True if registration successful
        """
        try:
            # Calculate retention period if not provided
            if retention_period is None:
                retention_period = max(
                    self.default_retention.get(cat, 1095) for cat in data_categories
                )
            
            subject = DataSubject(
                subject_id=subject_id,
                consent_given=consent_given,
                consent_date=datetime.now() if consent_given else None,
                data_categories=data_categories,
                legal_basis=legal_basis,
                retention_period=retention_period
            )
            
            self.data_subjects[subject_id] = subject
            self._save_data_subjects()
            
            self._log_activity(
                "data_subject_registered",
                {"subject_id": subject_id, "consent": consent_given}
            )
            
            logger.info(f"Registered data subject: {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data subject {subject_id}: {e}")
            return False
    
    def withdraw_consent(self, subject_id: str) -> bool:
        """Withdraw consent for a data subject.
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            True if withdrawal successful
        """
        if subject_id not in self.data_subjects:
            logger.warning(f"Data subject not found: {subject_id}")
            return False
        
        try:
            subject = self.data_subjects[subject_id]
            subject.consent_withdrawn = True
            subject.withdrawal_date = datetime.now()
            
            self._save_data_subjects()
            
            self._log_activity(
                "consent_withdrawn",
                {"subject_id": subject_id, "withdrawal_date": subject.withdrawal_date.isoformat()}
            )
            
            logger.info(f"Consent withdrawn for subject: {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent for {subject_id}: {e}")
            return False
    
    def can_process_data(self, subject_id: str) -> bool:
        """Check if data processing is allowed for a subject.
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            True if processing is allowed
        """
        if subject_id not in self.data_subjects:
            return False
        
        subject = self.data_subjects[subject_id]
        
        # Check if consent is withdrawn
        if subject.consent_withdrawn:
            return False
        
        # Check if data retention period has expired
        if subject.retention_period and subject.consent_date:
            expiry_date = subject.consent_date + timedelta(days=subject.retention_period)
            if datetime.now() > expiry_date:
                return False
        
        # Check legal basis
        if subject.legal_basis == LegalBasis.CONSENT:
            return subject.consent_given and not subject.consent_withdrawn
        
        # For other legal bases, assume processing is allowed
        return True
    
    def register_processing_activity(self,
                                   activity_id: str,
                                   purpose: str,
                                   data_categories: List[DataCategory],
                                   legal_basis: LegalBasis,
                                   recipients: List[str],
                                   retention_period: int,
                                   security_measures: List[str]) -> bool:
        """Register a processing activity.
        
        Args:
            activity_id: Unique identifier for activity
            purpose: Purpose of processing
            data_categories: Categories of data processed
            legal_basis: Legal basis for processing
            recipients: List of data recipients
            retention_period: Retention period in days
            security_measures: List of security measures
            
        Returns:
            True if registration successful
        """
        try:
            activity = ProcessingActivity(
                activity_id=activity_id,
                purpose=purpose,
                data_categories=data_categories,
                legal_basis=legal_basis,
                recipients=recipients,
                retention_period=retention_period,
                security_measures=security_measures,
                created_at=datetime.now()
            )
            
            self.processing_activities[activity_id] = activity
            self._save_processing_activities()
            
            self._log_activity(
                "processing_activity_registered",
                {"activity_id": activity_id, "purpose": purpose}
            )
            
            logger.info(f"Registered processing activity: {activity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register processing activity {activity_id}: {e}")
            return False
    
    def get_data_subject_info(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a data subject (for data subject access requests).
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            Dictionary with subject information or None
        """
        if subject_id not in self.data_subjects:
            return None
        
        subject = self.data_subjects[subject_id]
        
        info = {
            "subject_id": subject.subject_id,
            "consent_given": subject.consent_given,
            "consent_date": subject.consent_date.isoformat() if subject.consent_date else None,
            "consent_withdrawn": subject.consent_withdrawn,
            "withdrawal_date": subject.withdrawal_date.isoformat() if subject.withdrawal_date else None,
            "data_categories": [cat.value for cat in subject.data_categories],
            "legal_basis": subject.legal_basis.value,
            "retention_period": subject.retention_period,
            "can_process": self.can_process_data(subject_id)
        }
        
        # Add processing activities involving this subject
        info["processing_activities"] = [
            activity.activity_id for activity in self.processing_activities.values()
            if any(cat in subject.data_categories for cat in activity.data_categories)
        ]
        
        return info
    
    def delete_data_subject(self, subject_id: str) -> bool:
        """Delete all data for a subject (right to erasure).
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            True if deletion successful
        """
        if subject_id not in self.data_subjects:
            logger.warning(f"Data subject not found: {subject_id}")
            return False
        
        try:
            # Remove from data subjects
            del self.data_subjects[subject_id]
            self._save_data_subjects()
            
            self._log_activity(
                "data_subject_deleted",
                {"subject_id": subject_id, "deletion_date": datetime.now().isoformat()}
            )
            
            logger.info(f"Deleted data subject: {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data subject {subject_id}: {e}")
            return False
    
    def get_expired_data_subjects(self) -> List[str]:
        """Get list of data subjects whose retention period has expired.
        
        Returns:
            List of subject IDs with expired data
        """
        expired = []
        current_time = datetime.now()
        
        for subject_id, subject in self.data_subjects.items():
            if subject.retention_period and subject.consent_date:
                expiry_date = subject.consent_date + timedelta(days=subject.retention_period)
                if current_time > expiry_date:
                    expired.append(subject_id)
        
        return expired
    
    def cleanup_expired_data(self) -> int:
        """Remove data for subjects whose retention period has expired.
        
        Returns:
            Number of subjects cleaned up
        """
        expired_subjects = self.get_expired_data_subjects()
        cleaned_count = 0
        
        for subject_id in expired_subjects:
            if self.delete_data_subject(subject_id):
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired data subjects")
        
        return cleaned_count
    
    def _log_activity(self, activity_type: str, details: Dict[str, Any]):
        """Log compliance activity.
        
        Args:
            activity_type: Type of activity
            details: Activity details
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "activity_type": activity_type,
                "details": details
            }
            
            # Load existing log
            audit_log = []
            if self.audit_log_file.exists():
                with open(self.audit_log_file, 'r') as f:
                    audit_log = json.load(f)
            
            # Add new entry
            audit_log.append(log_entry)
            
            # Keep only last 10000 entries
            if len(audit_log) > 10000:
                audit_log = audit_log[-10000:]
            
            # Save log
            with open(self.audit_log_file, 'w') as f:
                json.dump(audit_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report.
        
        Returns:
            Compliance report dictionary
        """
        total_subjects = len(self.data_subjects)
        consented_subjects = sum(1 for s in self.data_subjects.values() if s.consent_given)
        withdrawn_subjects = sum(1 for s in self.data_subjects.values() if s.consent_withdrawn)
        expired_subjects = len(self.get_expired_data_subjects())
        
        # Data categories breakdown
        category_counts = {}
        for subject in self.data_subjects.values():
            for category in subject.data_categories:
                category_counts[category.value] = category_counts.get(category.value, 0) + 1
        
        # Legal basis breakdown
        legal_basis_counts = {}
        for subject in self.data_subjects.values():
            basis = subject.legal_basis.value
            legal_basis_counts[basis] = legal_basis_counts.get(basis, 0) + 1
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_subjects": {
                "total": total_subjects,
                "consented": consented_subjects,
                "withdrawn": withdrawn_subjects,
                "expired": expired_subjects
            },
            "data_categories": category_counts,
            "legal_basis": legal_basis_counts,
            "processing_activities": {
                "total": len(self.processing_activities),
                "activities": list(self.processing_activities.keys())
            }
        }
        
        return report