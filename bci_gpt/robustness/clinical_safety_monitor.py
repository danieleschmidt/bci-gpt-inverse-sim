#!/usr/bin/env python3
"""
Clinical Safety Monitoring System for BCI-GPT
Generation 2: Medical-grade safety monitoring with emergency protocols
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    EMERGENCY = "emergency"

class SafetyEvent(Enum):
    """Types of safety events."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FATIGUE_DETECTED = "fatigue_detected"
    SIGNAL_LOSS = "signal_loss"
    HIGH_IMPEDANCE = "high_impedance"
    ARTIFACT_SURGE = "artifact_surge"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_MALFUNCTION = "system_malfunction"

@dataclass
class SafetyAlert:
    """Safety alert with detailed context."""
    event_type: SafetyEvent
    severity: SafetyLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    actions_taken: List[str] = field(default_factory=list)

class ClinicalSafetyMonitor:
    """Comprehensive clinical safety monitoring system."""
    
    def __init__(self, 
                 max_session_duration: int = 3600,  # 1 hour
                 fatigue_threshold: float = 0.8,
                 emergency_contact: str = "safety@clinic.com"):
        
        self.max_session_duration = max_session_duration
        self.fatigue_threshold = fatigue_threshold
        self.emergency_contact = emergency_contact
        
        # Session tracking
        self.current_session = None
        self.session_history = []
        
        # Safety monitoring
        self.safety_alerts = []
        self.active_monitors = {}
        self.emergency_callbacks = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_queue = queue.Queue()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Clinical Safety Monitor initialized")
    
    def start_session(self, 
                     patient_id: str,
                     session_type: str = "bci_communication",
                     operator_id: str = None,
                     medical_supervision: bool = True) -> Dict[str, Any]:
        """Start monitored clinical session."""
        
        if self.current_session:
            self.logger.warning("Ending previous session before starting new one")
            self.end_session("new_session_requested")
        
        self.current_session = {
            "session_id": f"session_{int(time.time())}",
            "patient_id": patient_id,
            "session_type": session_type,
            "operator_id": operator_id,
            "medical_supervision": medical_supervision,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": 0,
            "safety_events": [],
            "quality_metrics": {},
            "emergency_stops": 0
        }
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Log session start
        self.log_safety_event(
            SafetyEvent.SESSION_START,
            SafetyLevel.NORMAL,
            f"Session started for patient {patient_id}",
            {"session_info": self.current_session}
        )
        
        self.logger.info(f"Clinical session started: {self.current_session['session_id']}")
        return self.current_session
    
    def end_session(self, reason: str = "normal_completion") -> Dict[str, Any]:
        """End current clinical session."""
        
        if not self.current_session:
            self.logger.warning("No active session to end")
            return {}
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Update session info
        self.current_session["end_time"] = datetime.now()
        self.current_session["duration"] = (
            self.current_session["end_time"] - self.current_session["start_time"]
        ).total_seconds()
        
        # Generate session report
        session_report = self.generate_session_report()
        
        # Archive session
        self.session_history.append(self.current_session.copy())
        
        # Log session end
        self.log_safety_event(
            SafetyEvent.SESSION_END,
            SafetyLevel.NORMAL,
            f"Session ended: {reason}",
            {
                "session_duration": self.current_session["duration"],
                "reason": reason,
                "report": session_report
            }
        )
        
        ended_session = self.current_session
        self.current_session = None
        
        self.logger.info(f"Clinical session ended: {ended_session['session_id']}")
        return ended_session
    
    def check_fatigue_status(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor patient fatigue from EEG signals."""
        
        fatigue_indicators = {
            "alpha_power_increase": False,
            "beta_power_decrease": False,
            "blink_rate_increase": False,
            "performance_degradation": False,
            "fatigue_score": 0.0
        }
        
        # Simple fatigue detection (would use advanced algorithms in production)
        if "data" in eeg_data:
            signal = eeg_data["data"]
            
            # Simulate fatigue detection
            # Real implementation would analyze:
            # - Alpha wave power increase (8-12 Hz)
            # - Beta wave power decrease (13-30 Hz) 
            # - Microsaccades and blinks
            # - Task performance metrics
            
            if isinstance(signal, list) and len(signal) > 0:
                # Mock fatigue calculation
                signal_variance = sum((x - sum(signal)/len(signal))**2 for x in signal) / len(signal)
                normalized_variance = min(1.0, signal_variance / 100.0)
                fatigue_indicators["fatigue_score"] = normalized_variance
                
                if normalized_variance > self.fatigue_threshold:
                    fatigue_indicators["performance_degradation"] = True
                    
                    self.log_safety_event(
                        SafetyEvent.FATIGUE_DETECTED,
                        SafetyLevel.WARNING,
                        f"Fatigue detected (score: {normalized_variance:.2f})",
                        {"fatigue_indicators": fatigue_indicators}
                    )
        
        return fatigue_indicators
    
    def check_signal_quality(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor EEG signal quality for safety issues."""
        
        quality_status = {
            "signal_present": True,
            "impedance_ok": True,
            "artifact_level": "low",
            "quality_score": 1.0,
            "safety_concern": False
        }
        
        # Check signal presence
        if "data" not in eeg_data or not eeg_data["data"]:
            quality_status["signal_present"] = False
            quality_status["safety_concern"] = True
            
            self.log_safety_event(
                SafetyEvent.SIGNAL_LOSS,
                SafetyLevel.EMERGENCY,
                "EEG signal lost - emergency stop required",
                {"eeg_data_keys": list(eeg_data.keys())}
            )
        
        # Check impedances
        if "impedances" in eeg_data:
            impedances = eeg_data["impedances"]
            high_impedance_threshold = 50000  # ohms
            
            high_impedance_channels = [
                i for i, imp in enumerate(impedances) 
                if imp > high_impedance_threshold
            ]
            
            if high_impedance_channels:
                quality_status["impedance_ok"] = False
                if len(high_impedance_channels) > len(impedances) // 2:
                    quality_status["safety_concern"] = True
                    
                    self.log_safety_event(
                        SafetyEvent.HIGH_IMPEDANCE,
                        SafetyLevel.WARNING,
                        f"High impedance detected in {len(high_impedance_channels)} channels",
                        {"affected_channels": high_impedance_channels}
                    )
        
        return quality_status
    
    def emergency_stop(self, reason: str, operator_id: str = None) -> Dict[str, Any]:
        """Execute emergency stop procedures."""
        
        emergency_response = {
            "timestamp": datetime.now(),
            "reason": reason,
            "operator_id": operator_id,
            "actions_taken": [],
            "session_terminated": False
        }
        
        self.logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        
        # Immediate actions
        emergency_response["actions_taken"].append("system_halt_initiated")
        
        # Stop current session
        if self.current_session:
            self.current_session["emergency_stops"] += 1
            emergency_response["session_terminated"] = True
            emergency_response["actions_taken"].append("session_terminated")
        
        # Execute emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(reason, emergency_response)
                emergency_response["actions_taken"].append("emergency_callback_executed")
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {e}")
        
        # Log emergency event
        self.log_safety_event(
            SafetyEvent.EMERGENCY_STOP,
            SafetyLevel.EMERGENCY,
            f"Emergency stop executed: {reason}",
            emergency_response
        )
        
        # Notify emergency contacts
        self.notify_emergency_contacts(reason, emergency_response)
        emergency_response["actions_taken"].append("emergency_contacts_notified")
        
        # End session
        if self.current_session:
            self.end_session(f"emergency_stop: {reason}")
        
        return emergency_response
    
    def register_emergency_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for emergency situations."""
        self.emergency_callbacks.append(callback)
        self.logger.info("Emergency callback registered")
    
    def notify_emergency_contacts(self, reason: str, context: Dict[str, Any]):
        """Notify emergency contacts (mock implementation)."""
        notification = {
            "timestamp": datetime.now().isoformat(),
            "emergency_reason": reason,
            "contact": self.emergency_contact,
            "context": context
        }
        
        # In production, would send actual notifications
        self.logger.critical(f"EMERGENCY NOTIFICATION: {json.dumps(notification, indent=2)}")
    
    def log_safety_event(self, 
                        event_type: SafetyEvent,
                        severity: SafetyLevel,
                        message: str,
                        context: Dict[str, Any] = None):
        """Log safety event with proper severity handling."""
        
        alert = SafetyAlert(
            event_type=event_type,
            severity=severity,
            message=message,
            context=context or {}
        )
        
        self.safety_alerts.append(alert)
        
        if self.current_session:
            self.current_session["safety_events"].append({
                "event_type": event_type.value,
                "severity": severity.value,
                "message": message,
                "timestamp": alert.timestamp.isoformat()
            })
        
        # Queue for monitoring thread
        self.alert_queue.put(alert)
        
        # Log with appropriate level
        if severity == SafetyLevel.EMERGENCY:
            self.logger.critical(f"🚨 {event_type.value.upper()}: {message}")
        elif severity == SafetyLevel.WARNING:
            self.logger.warning(f"⚠️  {event_type.value.upper()}: {message}")
        else:
            self.logger.info(f"ℹ️  {event_type.value.upper()}: {message}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        self.logger.info("Safety monitoring loop started")
        
        while self.monitoring_active and self.current_session:
            try:
                # Check session duration
                session_duration = (datetime.now() - self.current_session["start_time"]).total_seconds()
                
                if session_duration > self.max_session_duration:
                    self.log_safety_event(
                        SafetyEvent.EMERGENCY_STOP,
                        SafetyLevel.EMERGENCY,
                        f"Session duration {session_duration:.0f}s exceeds maximum {self.max_session_duration}s"
                    )
                    self.emergency_stop("session_duration_exceeded")
                    break
                
                # Process queued alerts
                try:
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                except queue.Empty:
                    pass
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Back off on errors
        
        self.logger.info("Safety monitoring loop ended")
    
    def _process_alert(self, alert: SafetyAlert):
        """Process safety alert with appropriate responses."""
        
        if alert.severity == SafetyLevel.EMERGENCY:
            # Immediate emergency response
            if alert.event_type in [SafetyEvent.SIGNAL_LOSS, SafetyEvent.SYSTEM_MALFUNCTION]:
                self.emergency_stop(f"Critical safety event: {alert.event_type.value}")
        
        elif alert.severity == SafetyLevel.WARNING:
            # Warning-level responses
            if alert.event_type == SafetyEvent.FATIGUE_DETECTED:
                # Suggest break or reduce session intensity
                self.logger.warning("Recommending session break due to fatigue")
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session safety report."""
        
        if not self.current_session:
            return {}
        
        session = self.current_session
        
        report = {
            "session_summary": {
                "session_id": session["session_id"],
                "patient_id": session["patient_id"],
                "duration": session["duration"],
                "session_type": session["session_type"],
                "medical_supervision": session["medical_supervision"]
            },
            "safety_metrics": {
                "total_safety_events": len(session["safety_events"]),
                "emergency_stops": session["emergency_stops"],
                "fatigue_episodes": len([
                    e for e in session["safety_events"] 
                    if e["event_type"] == SafetyEvent.FATIGUE_DETECTED.value
                ]),
                "signal_quality_issues": len([
                    e for e in session["safety_events"]
                    if e["event_type"] in [SafetyEvent.SIGNAL_LOSS.value, SafetyEvent.HIGH_IMPEDANCE.value]
                ])
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if report["safety_metrics"]["fatigue_episodes"] > 2:
            report["recommendations"].append("Consider shorter session durations")
        
        if report["safety_metrics"]["signal_quality_issues"] > 3:
            report["recommendations"].append("Check electrode placement and impedances")
        
        if session["duration"] > self.max_session_duration * 0.8:
            report["recommendations"].append("Session approached maximum duration")
        
        return report
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety monitoring status."""
        
        status = {
            "monitoring_active": self.monitoring_active,
            "current_session": self.current_session is not None,
            "total_sessions": len(self.session_history),
            "total_safety_alerts": len(self.safety_alerts),
            "recent_alerts": []
        }
        
        # Get recent alerts (last 10)
        recent_alerts = sorted(self.safety_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
        status["recent_alerts"] = [
            {
                "event_type": alert.event_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in recent_alerts
        ]
        
        if self.current_session:
            session_duration = (datetime.now() - self.current_session["start_time"]).total_seconds()
            status["current_session_info"] = {
                "session_id": self.current_session["session_id"],
                "duration": session_duration,
                "patient_id": self.current_session["patient_id"],
                "safety_events_count": len(self.current_session["safety_events"])
            }
        
        return status

# Example usage and testing
if __name__ == "__main__":
    print("🏥 Testing Clinical Safety Monitoring System...")
    
    # Initialize safety monitor
    safety_monitor = ClinicalSafetyMonitor(
        max_session_duration=30,  # 30 seconds for testing
        fatigue_threshold=0.5
    )
    
    # Register emergency callback
    def emergency_callback(reason: str, context: Dict[str, Any]):
        print(f"🚨 EMERGENCY CALLBACK: {reason}")
    
    safety_monitor.register_emergency_callback(emergency_callback)
    
    # Start session
    session = safety_monitor.start_session(
        patient_id="test_patient_001",
        session_type="communication_training",
        operator_id="clinician_jane",
        medical_supervision=True
    )
    
    print(f"✅ Session started: {session['session_id']}")
    
    # Simulate monitoring
    import time
    
    # Test fatigue detection
    mock_eeg_data = {
        "data": [100 * i for i in range(50)],  # High variance signal
        "sampling_rate": 1000,
        "impedances": [5000, 45000, 6000]  # One high impedance
    }
    
    fatigue_status = safety_monitor.check_fatigue_status(mock_eeg_data)
    print(f"✅ Fatigue check: {fatigue_status['fatigue_score']:.2f}")
    
    signal_quality = safety_monitor.check_signal_quality(mock_eeg_data)
    print(f"✅ Signal quality: {signal_quality['quality_score']:.2f}")
    
    # Wait to test session duration monitoring
    print("⏳ Waiting for session duration limit...")
    time.sleep(5)
    
    # Check status
    status = safety_monitor.get_safety_status()
    print(f"✅ Safety status: {status['total_safety_alerts']} alerts")
    
    # Let monitoring detect timeout
    time.sleep(30)
    
    # Manual end if not automatically ended
    if safety_monitor.current_session:
        final_session = safety_monitor.end_session("test_completed")
        print(f"✅ Session ended: {final_session['duration']:.1f}s")
    
    print("\n🛡️  Clinical Safety System Ready!")
