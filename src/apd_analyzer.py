
"""
APD Analyzer Module
Handles APD violation analysis and classification
"""

import cv2
import numpy as np

class APDAnalyzer:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize APD Analyzer
        
        Args:
            confidence_threshold: Threshold for APD detection confidence
        """
        self.confidence_threshold = confidence_threshold
        
        print("✅ APD Analyzer initialized")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
    
    def analyze_apd_status(self, person_bbox, apd_detections):
        """
        Analyze APD violations for a person (focus on violations only)
        
        Args:
            person_bbox: Bounding box of the person [x1, y1, x2, y2]
            apd_detections: List of APD item detections (if any)
            
        Returns:
            Dictionary with violation analysis results
        """
        px1, py1, px2, py2 = person_bbox
        
        # For violation detection, we assume no APD by default
        has_helmet = False
        has_vest = False
        helmet_confidence = 0.0
        vest_confidence = 0.0
        
        # Only check for APD if we have APD detections (when using custom model)
        if apd_detections:
            for detection in apd_detections:
                dx1, dy1, dx2, dy2 = detection['bbox']
                
                # Check if APD item overlaps with person
                if self._is_overlapping(person_bbox, detection['bbox']):
                    if detection['class'] == 'helmet' and detection['confidence'] > self.confidence_threshold:
                        has_helmet = True
                        helmet_confidence = detection['confidence']
                    elif detection['class'] == 'vest' and detection['confidence'] > self.confidence_threshold:
                        has_vest = True
                        vest_confidence = detection['confidence']
        
        # Focus on VIOLATION types (not compliant status)
        violation_type = self._determine_violation_type(has_helmet, has_vest)
        
        return {
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'helmet_confidence': helmet_confidence,
            'vest_confidence': vest_confidence,
            'violation_type': violation_type,
            'is_violation': violation_type != 'compliant',
            'violation_severity': self._get_violation_severity(violation_type)
        }
    
    def analyze_frame(self, person_detections, apd_detections):
        """
        Analyze APD violations for all persons in a frame (focus on violations)
        
        Args:
            person_detections: List of person detections
            apd_detections: List of APD item detections
            
        Returns:
            List of violation analysis results for each person
        """
        results = []
        
        for person in person_detections:
            analysis = self.analyze_apd_status(person['bbox'], apd_detections)
            
            # Add person information to analysis
            result = {
                'person_bbox': person['bbox'],
                'person_confidence': person['confidence'],
                **analysis
            }
            
            results.append(result)
        
        return results
    
    def get_violation_summary(self, analyses):
        """
        Get summary of violations from analyses (focus on violations only)
        
        Args:
            analyses: List of APD analysis results
            
        Returns:
            Dictionary with violation summary
        """
        total_persons = len(analyses)
        violations = [a for a in analyses if a['is_violation']]
        total_violations = len(violations)
        
        helmet_violations = sum(1 for a in analyses if a['violation_type'] == 'no_helmet')
        vest_violations = sum(1 for a in analyses if a['violation_type'] == 'no_vest')
        both_violations = sum(1 for a in analyses if a['violation_type'] == 'both_violations')
        
        # Calculate violation rate (instead of compliance rate)
        violation_rate = (total_violations / total_persons * 100) if total_persons > 0 else 0
        
        return {
            'total_persons': total_persons,
            'total_violations': total_violations,
            'helmet_violations': helmet_violations,
            'vest_violations': vest_violations,
            'both_violations': both_violations,
            'violation_rate': violation_rate,
            'high_severity_violations': sum(1 for a in violations if a.get('violation_severity') == 'high'),
            'medium_severity_violations': sum(1 for a in violations if a.get('violation_severity') == 'medium')
        }
    
    def _is_overlapping(self, bbox1, bbox2):
        """
        Check if two bounding boxes overlap
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Boolean indicating if boxes overlap
        """
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        return x1_max < x2_min and y1_max < y2_min
    
    def _determine_violation_type(self, has_helmet, has_vest):
        """
        Determine violation type based on APD status (focus on violations)
        
        Args:
            has_helmet: Boolean indicating if person has helmet
            has_vest: Boolean indicating if person has vest
            
        Returns:
            Violation type string
        """
        if has_helmet and has_vest:
            return 'compliant'  # No violation
        elif not has_helmet and not has_vest:
            return 'both_violations'  # Most severe violation
        elif not has_helmet:
            return 'no_helmet'  # Helmet violation
        elif not has_vest:
            return 'no_vest'  # Vest violation
        else:
            return 'unknown'
    
    def _get_violation_severity(self, violation_type):
        """
        Get violation severity level
        
        Args:
            violation_type: Type of violation
            
        Returns:
            Severity level string
        """
        severity_map = {
            'both_violations': 'high',
            'no_helmet': 'medium',
            'no_vest': 'medium',
            'compliant': 'none',
            'unknown': 'low'
        }
        return severity_map.get(violation_type, 'low')
    
    def draw_analysis_overlay(self, frame, analyses):
        """
        Draw violation analysis overlay on frame (focus on violations)
        
        Args:
            frame: Input frame
            analyses: List of APD analysis results
            
        Returns:
            Frame with violation overlay drawn
        """
        overlay = frame.copy()
        
        for analysis in analyses:
            bbox = analysis['person_bbox']
            x1, y1, x2, y2 = bbox
            
            # Choose color based on violation (red for violations, green for compliant)
            if analysis['is_violation']:
                if analysis['violation_severity'] == 'high':
                    color = (0, 0, 255)  # Red for high severity
                elif analysis['violation_severity'] == 'medium':
                    color = (0, 165, 255)  # Orange for medium severity
                else:
                    color = (255, 255, 0)  # Yellow for low severity
                label = f"VIOLATION: {analysis['violation_type'].upper()}"
            else:
                color = (0, 255, 0)  # Green for compliant
                label = "COMPLIANT"
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(overlay, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(overlay, label, (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw violation severity indicator
            if analysis['is_violation']:
                severity_text = f"Severity: {analysis['violation_severity'].upper()}"
                cv2.putText(overlay, severity_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def set_confidence_threshold(self, threshold):
        """
        Set confidence threshold for APD detection
        
        Args:
            threshold: Confidence threshold (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 APD confidence threshold set to {self.confidence_threshold}")
    
    def get_analyzer_info(self):
        """
        Get information about the analyzer (focus on violations)
        
        Returns:
            Dictionary with analyzer information
        """
        return {
            'confidence_threshold': self.confidence_threshold,
            'violation_types': ['compliant', 'no_helmet', 'no_vest', 'both_violations'],
            'violation_severity_levels': ['none', 'low', 'medium', 'high'],
            'focus': 'APD Violation Detection',
            'supported_apd_items': ['helmet', 'vest']
        }
