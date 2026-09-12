import re
import html
from typing import Dict, Any, List

class TextPreprocessor:
    """
    Cleans incoming customer tweets, detects and redacts Sensitive Personal Identifiable
    Information (PII) such as Delta 6-character PNRs and 13-digit e-ticket numbers,
    and flags critical operational and emotional signals.
    """

    # Delta PNRs are 6-character uppercase alphanumeric strings (e.g., H7K9P2, ABC123)
    PNR_REGEX = re.compile(r'\b[A-Z0-9]{6}\b')
    # Airline e-ticket numbers (Delta ticket numbers start with '006' and are 13 digits)
    TICKET_REGEX = re.compile(r'\b006\d{10}\b|\b\d{13}\b')
    # Credit Card pattern (4 groups of 4 digits or 16 continuous digits)
    CREDIT_CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
    # Phone number regex
    PHONE_REGEX = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    # Email regex
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

    # Urgency & Risk Lexicons
    CRITICAL_RISK_KEYWORDS = [
        "medical", "insulin", "heart medication", "wheelchair", "elderly", "unaccompanied minor",
        "stranded", "stuck on tarmac", "hours on tarmac", "tarmac", "lawsuit", "dot complaint",
        "faail", "screaming", "crying", "passed out", "missed wedding", "missed funeral"
    ]
    
    IMMINENT_FLIGHT_KEYWORDS = [
        "gate closing", "boarding now", "take off in", "departing in", "minutes until",
        "doors closing", "stuck in line", "tsa line", "flight is leaving"
    ]

    SARCASM_MARKERS = [
        "magical", "wonderful job", "great job delta", "outstanding service", "brilliant service",
        "slowest airline", "thanks for leaving me", "thanks for nothing", "love spending the night",
        "clap", "slow clap", "golf clap"
    ]

    def clean_text(self, text: str) -> str:
        """Decodes HTML entities and normalizes whitespace."""
        if not text:
            return ""
        text = html.unescape(text)
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def sanitize_pii(self, text: str) -> Dict[str, Any]:
        """
        Detects and masks sensitive customer data.
        Returns cleaned text, masked text, detected entities, and boolean flag.
        """
        cleaned = self.clean_text(text)
        entities_found = []
        masked_text = cleaned

        # 1. Credit Cards
        cc_matches = self.CREDIT_CARD_REGEX.findall(masked_text)
        if cc_matches:
            entities_found.append("CREDIT_CARD")
            masked_text = self.CREDIT_CARD_REGEX.sub("[REDACTED_CREDIT_CARD]", masked_text)

        # 2. Delta Ticket Numbers (006...)
        ticket_matches = self.TICKET_REGEX.findall(masked_text)
        if ticket_matches:
            entities_found.append("TICKET_NUMBER")
            masked_text = self.TICKET_REGEX.sub("[REDACTED_TICKET_NUMBER]", masked_text)

        # 3. Emails
        email_matches = self.EMAIL_REGEX.findall(masked_text)
        if email_matches:
            entities_found.append("EMAIL")
            masked_text = self.EMAIL_REGEX.sub("[REDACTED_EMAIL]", masked_text)

        # 4. Phones
        phone_matches = self.PHONE_REGEX.findall(masked_text)
        if phone_matches:
            entities_found.append("PHONE_NUMBER")
            masked_text = self.PHONE_REGEX.sub("[REDACTED_PHONE]", masked_text)

        # 5. PNR / Confirmation Codes (Avoid matching common 6-letter English words)
        stopwords_6char = {"PLEASE", "THANKS", "FLIGHT", "GROUND", "RETURN", "AIRBUS", "BOEING", "SEATTLE", "BOSTON", "ALWAYS", "UPDATE", "FAILED", "WINDOW", "BEFORE"}
        potential_pnrs = self.PNR_REGEX.findall(masked_text)
        for cand in potential_pnrs:
            if cand not in stopwords_6char and any(c.isdigit() for c in cand):
                entities_found.append("PNR_CONFIRMATION_CODE")
                masked_text = re.sub(r'\b' + re.escape(cand) + r'\b', "[REDACTED_PNR]", masked_text)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "sanitized_text": masked_text,
            "pii_detected": len(entities_found) > 0,
            "entities": list(set(entities_found))
        }

    def extract_signals(self, text: str) -> Dict[str, Any]:
        """
        Extracts urgency, high-liability risks, and sarcasm indicators.
        """
        text_lower = text.lower()
        critical_flags = [kw for kw in self.CRITICAL_RISK_KEYWORDS if kw in text_lower]
        imminent_flags = [kw for kw in self.IMMINENT_FLIGHT_KEYWORDS if kw in text_lower]
        sarcasm_flags = [kw for kw in self.SARCASM_MARKERS if kw in text_lower]

        has_flight_num = bool(re.search(r'\b(?:DL|DL\s?|Flight\s?)\d{1,4}\b', text, re.IGNORECASE))

        return {
            "has_critical_risk": len(critical_flags) > 0,
            "critical_factors": critical_flags,
            "is_imminent": len(imminent_flags) > 0,
            "imminent_factors": imminent_flags,
            "has_sarcasm": len(sarcasm_flags) > 0,
            "sarcasm_factors": sarcasm_flags,
            "has_flight_number": has_flight_num
        }
