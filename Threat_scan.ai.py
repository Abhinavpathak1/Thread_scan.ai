"""ThreatScan AI: standalone fraud and phishing risk scanner.

Run `python Threat_scan.ai.py --train` once to train the models, then launch
with `python -m streamlit run Threat_scan.ai.py` using that same Python.
"""
import argparse
import csv
import html
import json
import math
import os
import re
import string
import sys
import time
import textwrap
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import sklearn
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
MODELS_DIR = str(Path(__file__).resolve().parent / "models")

# Embedded training dataset
DATASET = [
    # ══════════════════════════════════════════════════════════
    # LABEL 0 — LEGITIMATE
    # ══════════════════════════════════════════════════════════

    # Casual / personal
    ("Hey, are we still on for coffee at 3pm tomorrow?", 0),
    ("Thanks for dinner last night, it was lovely! Let's do it again soon.", 0),
    ("Can you pick up milk and bread on the way home?", 0),
    ("Happy birthday! Hope you have an amazing day filled with joy.", 0),
    ("Don't forget we have the family reunion on Sunday at noon.", 0),
    ("I just finished reading that book you recommended — absolutely brilliant.", 0),
    ("Running 10 minutes late, sorry! Start without me.", 0),
    ("The kids had a great time at the party. Thanks for organizing it.", 0),
    ("Thinking of you during this difficult time. Let me know if you need anything.", 0),
    ("Just checking in — how did the interview go yesterday?", 0),

    # Work / professional
    ("Reminder: the quarterly review is scheduled for Thursday at 2pm in conference room B.", 0),
    ("Please find attached the invoice for services rendered in October.", 0),
    ("Hi team, the standup meeting has been moved to 2pm today due to a conflict.", 0),
    ("Great work on the presentation today. The client was very impressed.", 0),
    ("The project deadline has been extended to the 20th. Please update your timelines.", 0),
    ("Could you review the attached document before our meeting tomorrow?", 0),
    ("Your leave request for December 24–26 has been approved.", 0),
    ("The quarterly report is ready for review in the shared drive.", 0),
    ("Please complete the mandatory cybersecurity training by end of month.", 0),
    ("I've uploaded the latest design mockups to the project folder.", 0),
    ("The budget proposal has been submitted to finance for review.", 0),
    ("We'll need your feedback on the draft contract by Friday at the latest.", 0),
    ("Our new office address will be effective from the 1st of next month.", 0),
    ("The all-hands meeting has been rescheduled to next Wednesday.", 0),

    # Transactional / notifications
    ("Your order #78432 has shipped and will arrive by Thursday.", 0),
    ("Your library books are due back on the 15th of this month.", 0),
    ("Your dentist appointment is confirmed for Friday at 10am with Dr. Smith.", 0),
    ("Your password was successfully changed. If this wasn't you, contact support.", 0),
    ("Your gym membership auto-renews on the 1st. No action required.", 0),
    ("Your flight BA2291 to New York departs at 08:15 from Terminal 5.", 0),
    ("Your subscription to Adobe Creative Cloud renews on November 1.", 0),
    ("Your tax documents for last year are ready to download from your account.", 0),
    ("We've updated our privacy policy. You can review the changes in your settings.", 0),
    ("Your parcel was delivered and left in a safe place as instructed.", 0),
    ("Your bank statement for September is now available in your online account.", 0),
    ("Reminder: your car MOT is due next month. Book online at our website.", 0),
    ("Your Amazon Prime membership was renewed for another year.", 0),
    ("Your Spotify subscription payment was processed successfully.", 0),
    ("We confirm your hotel reservation for 2 nights, check-in December 14.", 0),
    ("Your electricity bill for October is £87.43, due by the 20th.", 0),
    ("Your insurance renewal is due in 30 days. We'll send a reminder closer to the date.", 0),

    # Customer support (real)
    ("Thank you for contacting us. A support agent will respond within 24 hours.", 0),
    ("We're sorry to hear about your experience. Please reply with your order number.", 0),
    ("Your refund of £23.50 has been processed and will appear in 3–5 business days.", 0),
    ("We've resolved the issue you reported. Please let us know if it recurs.", 0),

    # News / informational
    ("The local council meeting will be held at the town hall on Monday at 7pm.", 0),
    ("Road closure on Main Street this weekend due to water main repairs.", 0),
    ("School will be closed on Friday for teacher training day.", 0),
    ("The community fundraiser raised over £5,000 for the local food bank.", 0),

    # Emails with links (legit)
    ("Please reset your password using this link: https://accounts.google.com/reset/password", 0),
    ("View your invoice online: https://stripe.com/invoices/in_1234567890", 0),
    ("Track your order here: https://www.amazon.co.uk/orders/track?id=123", 0),
    ("Join the meeting: https://zoom.us/j/123456789?pwd=abc123", 0),
    ("Download the report: https://docs.google.com/document/d/1a2b3c", 0),

    # ══════════════════════════════════════════════════════════
    # LABEL 1 — FRAUD / PHISHING / SCAM / THREAT
    # ══════════════════════════════════════════════════════════

    # Phishing — account takeover
    ("URGENT: Your bank account has been suspended! Verify immediately at http://secure-bank-login.xyz", 1),
    ("Your PayPal account is limited. Click here to restore access: http://paypal-secure.tk/verify", 1),
    ("Your Apple ID has been locked due to suspicious activity. Verify at: http://apple-id-verify.tk", 1),
    ("Dear customer, your Netflix subscription has expired. Update payment: http://netflix-billing.ml", 1),
    ("Your Amazon account will be closed! Confirm your details at http://amazon-verification.gq/login", 1),
    ("Microsoft account alert: Unusual sign-in detected. Secure your account: http://msft-secure.xyz", 1),
    ("Your Chase bank account has been temporarily frozen. Unlock at: http://chase-verify.top/login", 1),
    ("ALERT: Your Gmail password was changed. If not you, click here immediately: http://google-secure.tk", 1),
    ("Your Facebook account has been flagged. Restore access: http://fb-account-verify.xyz/restore", 1),
    ("IMPORTANT: Your Instagram was accessed from a new device. Verify: http://insta-secure.ml/check", 1),
    ("Your Coinbase account is at risk. Secure it NOW: http://coinbase-alert.tk/secure", 1),
    ("Wells Fargo: We detected suspicious activity. Please verify at http://wellsfargo-secure.gq", 1),
    ("Your Venmo account requires immediate verification: http://venmo-verify.xyz/confirm", 1),
    ("HSBC Security Alert: Please verify your identity at http://hsbc-online-secure.top/id-check", 1),
    ("Your Apple Pay has been disabled. Reactivate at http://applepay-secure.xyz/activate", 1),

    # Lottery / prize scams
    ("Congratulations! You've won $1,000,000 in our international lottery. Claim your prize now!", 1),
    ("You have been selected as our lucky winner! You've won a FREE iPhone 15! Claim: http://free-prize.xyz", 1),
    ("WINNER! You are our lucky customer #10000. Reply with your bank details to receive $5,000.", 1),
    ("Congratulations! Your email won the UK national lottery. Send your details to claim £850,000.", 1),
    ("You've won an exclusive reward! Claim your $500 gift card before it expires: http://reward-claim.tk", 1),
    ("FINAL DRAW: You are one of 5 winners of our $2 million sweepstakes! Respond immediately.", 1),
    ("Your survey reward of $750 is ready. Complete verification at http://survey-reward.xyz/claim", 1),
    ("Our records show you have unclaimed prize money of $4,500. Claim here: http://claim-funds.ml", 1),

    # IRS / government scams
    ("IRS FINAL NOTICE: You owe $3,200. Pay now or face ARREST. Call 1-800-555-IRS immediately.", 1),
    ("IRS ALERT: An arrest warrant has been issued. Pay $2,500 in gift cards to avoid jail.", 1),
    ("Your Social Security Number has been suspended. Call 1-888-555-9999 to reactivate immediately.", 1),
    ("SOCIAL SECURITY ADMINISTRATION: Your SSN has been involved in criminal activity. Call NOW.", 1),
    ("Department of Justice: You are under federal investigation. Contact agent immediately.", 1),
    ("IRS: Your tax refund of $4,200 is pending. Provide your bank routing number to receive funds.", 1),
    ("Medicare Alert: Your benefits will be cancelled unless you confirm your details today.", 1),
    ("Government grant of $9,800 available for you. No repayment! Apply: http://gov-grant.xyz", 1),

    # Nigerian prince / advance fee fraud
    ("DEAR FRIEND, I am a prince from Nigeria and need your help to transfer $45 million USD.", 1),
    ("I am the late General's widow. I have $28 million to transfer and need a trusted partner.", 1),
    ("Greetings, I am a diplomat with a consignment box worth $15.5 million needing your assistance.", 1),
    ("I have been diagnosed with cancer and wish to donate my $12 million estate to a trustworthy person.", 1),
    ("CONFIDENTIAL: I am an attorney for a deceased client who shares your surname. $6.5M is unclaimed.", 1),
    ("GOD BLESS YOU. I am contacting you regarding an inheritance of $4.2 million that needs a beneficiary.", 1),

    # Tech support scams
    ("Microsoft Security Alert: Your computer has been infected with 3 viruses! Call 1-800-555-MSFT NOW.", 1),
    ("WARNING: 5 critical errors found on your PC. Download our free antivirus: http://protect-now.xyz", 1),
    ("APPLE SECURITY: Your iPhone has been hacked. Call our specialist team: 1-888-APPLE-FIX", 1),
    ("Your computer is sending INFECTED data to the internet! Call support immediately: 1-800-555-1234", 1),
    ("Google Alert: Your Chrome browser has been compromised. Fix it now: http://chrome-fix.tk", 1),
    ("Norton Security: Your license has expired. Your PC is at risk. Renew at http://norton-renew.xyz", 1),

    # Financial / investment scams
    ("Crypto investment opportunity! Turn $500 into $50,000 in 30 days. 100% guaranteed returns!", 1),
    ("LIMITED TIME: Invest in our Bitcoin fund and earn 40% monthly returns. Minimum deposit $200.", 1),
    ("Work from home and earn $5,000 per week! No experience needed. Apply now for instant approval.", 1),
    ("Make $3,000 daily from home with our proven trading system. Free trial — sign up now!", 1),
    ("EXCLUSIVE: Early access to next 1000x crypto token. Buy now before it launches publicly!", 1),
    ("Forex trading signal: guaranteed 85% win rate. Join our VIP group for just $99/month.", 1),
    ("Your credit score can be ERASED legally! Remove all bad items for $299. Call now!", 1),
    ("Pre-approved for a $50,000 personal loan with NO credit check! Zero interest for 12 months!", 1),
    ("Debt relief program: eliminate 80% of your debt legally. Free consultation today!", 1),

    # Gift card / wire transfer scams
    ("Send $500 in iTunes gift cards and email me the codes. This is URGENT and confidential.", 1),
    ("I need you to purchase $1,000 in Google Play gift cards. I'll explain later. Very urgent.", 1),
    ("Please wire $2,000 to this account immediately. I'm stuck abroad and need help. — [Boss name]", 1),
    ("CEO REQUEST: Purchase $800 in Amazon gift cards for client gifts. Keep this confidential.", 1),
    ("Emergency: A family member has been arrested abroad. Send $3,000 bail via Western Union NOW.", 1),
    ("I am stranded in London after being robbed. Please wire £1,500 urgently. I'll repay you.", 1),

    # Delivery / parcel scams
    ("FINAL NOTICE: Unpaid toll of $2.34. Pay now to avoid $55 fine: http://pay-toll.xyz/now", 1),
    ("Your parcel is held at customs. Pay £2.99 clearance fee: http://royal-mail-customs.tk", 1),
    ("DHL: Your package could not be delivered. Pay £1.45 redelivery fee: http://dhl-redelivery.xyz", 1),
    ("FedEx Alert: Your shipment requires additional customs clearance. Pay at http://fedex-customs.ml", 1),
    ("USPS: Your parcel is on hold. Confirm your address and pay $3 fee: http://usps-track.xyz", 1),

    # Urgency / account suspension
    ("LAST CHANCE: Your account expires in 24 hours! Click immediately or lose all data: http://renew-now.gq", 1),
    ("URGENT: Your email account will be deleted in 12 hours unless you verify: http://email-verify.tk", 1),
    ("ALERT: Suspicious login from Russia detected on your account. Verify NOW or account deleted.", 1),
    ("Your account has been scheduled for permanent deletion. Cancel this action: http://cancel-delete.xyz", 1),
    ("WARNING: You have exceeded your storage limit. Your emails will be deleted. Upgrade: http://email-upgrade.ml", 1),
    ("Your subscription will auto-renew for $299 unless cancelled. Click: http://cancel-subscription.xyz", 1),

    # Romantic / relationship scams
    ("I saw your profile and felt an immediate connection. I am a US Army officer stationed abroad.", 1),
    ("Hello beautiful, I am a successful surgeon working with MSF. I would love to know you better.", 1),
    ("I know this is sudden but I have developed strong feelings for you. I need $3,000 for medical emergency.", 1),

    # COVID / health scams
    ("You qualify for a $1,400 COVID relief stimulus. Claim at http://covid-relief-fund.xyz now.", 1),
    ("FREE COVID-19 test kit — claim yours today. Provide your Medicare number: http://covid-test-free.tk", 1),
    ("VACCINE SIDE EFFECT COMPENSATION: You may be owed $5,000. File your claim: http://vax-claim.xyz", 1),

    # SMS smishing
    ("Your bank card has been used for a £450 transaction. If not you call 0800-555-FRAUD immediately.", 1),
    ("HMRC: You are owed a tax refund. Claim £289 at http://hmrc-refund.xyz before it expires.", 1),
    ("Your mobile account has been compromised. Verify at http://o2-secure.tk or service suspended.", 1),

    # Mixed urgency + financial
    ("FREE money from the government! You qualify for $8,500. Provide your SSN to claim today.", 1),
    ("URGENT: You have 48 hours to claim your inheritance of $780,000 before it is forfeited.", 1),
    ("Exclusive: You have been pre-selected for a $25,000 home improvement grant. Apply by midnight.", 1),
    ("Your student loan can be FORGIVEN today! Submit your details: http://loan-forgive.xyz/apply", 1),

    # ══════════════════════════════════════════════════════════
    # LABEL 0 — BORDERLINE LEGITIMATE (should score low)
    # ══════════════════════════════════════════════════════════
    ("Limited time offer: 30% off your next purchase. Use code SAVE30 at checkout.", 0),
    ("Don't miss our Black Friday sale — up to 50% off all products this weekend only.", 0),
    ("Your account login was attempted from a new device. Was this you? Click to confirm.", 0),
    ("Reminder: your annual subscription renews in 7 days. No action needed unless you wish to cancel.", 0),
    ("We noticed you haven't logged in recently. Click here to keep your account active.", 0),
    ("Your credit card ending in 4321 was charged $87.50 at Tesco. Not you? Contact us.", 0),
    ("Please verify your email address to complete your registration: https://signup.example.com/verify", 0),
    ("Your trial period ends in 3 days. Upgrade to keep access: https://app.example.com/upgrade", 0),
    ("You have a new message waiting in your inbox. Log in to read it.", 0),
    ("Join our referral program and earn $10 for every friend who signs up.", 0),

    # ══════════════════════════════════════════════════════════
    # LABEL 1 — BORDERLINE FRAUD (subtle)
    # ══════════════════════════════════════════════════════════
    ("We noticed unusual activity on your account. Please verify your recent transactions immediately.", 1),
    ("You may be eligible for a $500 refund. Confirm your personal details to receive it.", 1),
    ("Claim your exclusive membership reward of 50,000 points before they expire tonight.", 1),
    ("IMPORTANT: Your credit score has dropped 80 points. Fix it now: http://credit-fix.xyz", 1),
    ("Your debit card has been used at an ATM in another country. Confirm or block: http://card-secure.tk", 1),
    ("Congratulations! As a valued customer you've been selected for a special cash reward of $250.", 1),
    ("Act now — your loan application has been pre-approved. No credit check. Limited slots available.", 1),
    ("Verify your PayPal account to receive your pending payment of $1,247.50.", 1),
    ("Your identity has been used to open a new credit account. Dispute this immediately.", 1),
    ("WARNING: We found your email and password on the dark web. Secure your account now.", 1),
]

TEXTS = [item[0] for item in DATASET]
LABELS = [item[1] for item in DATASET]


def ensure_dataset_file(path: Path | None = None) -> Path:
    """Export the embedded training data if its CSV copy is missing."""
    dataset_path = path or Path(__file__).resolve().parent / "dataset.csv"
    if dataset_path.exists():
        return dataset_path

    with dataset_path.open("x", newline="", encoding="utf-8") as dataset_file:
        writer = csv.writer(dataset_file)
        writer.writerow(("text", "label", "class"))
        writer.writerows(
            (text, label, "fraud" if label else "legitimate")
            for text, label in DATASET
        )
    return dataset_path


# Embedded feature engineering and URL analysis
URGENCY_WORDS = [
    "urgent", "immediately", "expires", "limited time", "act now", "hours left",
    "deadline", "final notice", "last chance", "expire", "now or", "before it",
    "respond immediately", "within 24", "within 48", "today only", "don't wait",
    "time sensitive", "asap", "right now", "instant access", "ending soon",
    "24 hours", "48 hours", "midnight", "expire tonight", "final warning",
    "last reminder", "one last chance",
]

FINANCIAL_WORDS = [
    "bank account", "credit card", "debit card", "ssn", "social security",
    "wire transfer", "bitcoin", "crypto", "cryptocurrency", "paypal", "venmo",
    "cashapp", "zelle", "western union", "moneygram", "routing number",
    "account number", "gift card", "itunes", "google play", "amazon gift",
    "prize", "winner", "lottery", "million", "thousand dollars", "free money",
    "cash reward", "unclaimed", "grant", "inheritance", "refund", "tax refund",
    "irs", "claim your", "reward points", "investment", "returns", "profit",
    "earn money", "make money", "income", "passive income", "financial freedom",
    "no credit check", "pre-approved", "guaranteed returns",
]

THREAT_WORDS = [
    "arrested", "arrest warrant", "lawsuit", "legal action", "suspended",
    "account closed", "account locked", "account compromised", "hacked",
    "virus detected", "malware", "infected", "security breach", "blocked",
    "frozen", "terminated", "deleted", "cancelled", "legal proceedings",
    "court summons", "police", "fbi", "federal", "criminal", "investigation",
    "felony", "charges filed",
]

SOCIAL_ENG_WORDS = [
    "dear friend", "beloved", "greetings", "i am a prince", "nigerian",
    "inheritance", "deceased", "confidential", "secret agent", "only you",
    "trusted partner", "god bless", "percentage", "million usd", "million dollars",
    "help me transfer", "dying", "cancer", "widow", "diplomat", "consignment",
    "unclaimed funds", "foreign fund",
]

IMPERSONATION_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook", "netflix",
    "bank of america", "chase", "wells fargo", "citibank", "irs", "fbi", "cia",
    "social security administration", "fedex", "ups", "dhl", "usps", "customs",
    "hmrc", "royal mail", "vodafone", "o2", "hsbc", "barclays", "lloyds",
    "coinbase", "binance", "spotify", "instagram", "whatsapp",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".club", ".work",
    ".click", ".link", ".download", ".zip", ".review", ".country",
    ".science", ".vip", ".win", ".online", ".site", ".space", ".faith",
    ".stream", ".date", ".racing",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "short.link",
    "buff.ly", "dlvr.it", "ift.tt", "rebrand.ly", "bl.ink", "cutt.ly",
    "is.gd", "rb.gy", "shorturl.at",
]

TRUST_INDICATORS = [
    "https://www.", "your account", "as requested", "thank you for",
    "confirmation", "receipt", "tracking number", "order number",
    "reference number", "invoice", "statement",
]

MANIPULATION_PHRASES = [
    "act now", "don't miss", "you've been selected", "exclusively for you",
    "only you can", "this offer expires", "final opportunity", "last chance",
    "before it's too late", "while supplies last", "limited availability",
    "respond immediately", "do not ignore", "failure to respond", "or else",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def extract_urls(text: str) -> list:
    return re.findall(r'https?://[^\s<>"\',;]+', text)

def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def domain_risk_score(domain: str) -> float:
    score = 0.0
    if any(tld in domain for tld in SUSPICIOUS_TLDS):
        score += 0.40
    if any(s in domain for s in URL_SHORTENERS):
        score += 0.30
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
        score += 0.40
    if domain.count('-') >= 3:
        score += 0.20
    if domain.count('.') >= 4:
        score += 0.15
    for brand in IMPERSONATION_BRANDS:
        brand_clean = brand.replace(' ', '')
        if brand_clean in domain and not domain.endswith(f".{brand_clean}.com"):
            score += 0.35
    ent = char_entropy(domain)
    if ent > 3.6:
        score += 0.20
    return min(1.0, score)

# ──────────────────────────────────────────────────────────────
# SKLEARN TRANSFORMER
# ──────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Keyword counts
    "urgency_hits", "financial_hits", "threat_hits",
    "social_eng_hits", "impersonation_hits", "manipulation_hits",
    "trust_hits",
    # Structural
    "caps_ratio", "exclamation_count", "question_count",
    "url_count", "word_count", "char_count", "avg_word_len",
    "sentence_count", "avg_sentence_len",
    # Style
    "double_space_count", "dollar_signs", "number_blobs",
    "all_caps_words", "ellipsis_count", "comma_count",
    # URL
    "has_url", "max_url_domain_risk", "avg_url_domain_risk",
    "has_http_only", "has_shortener", "has_ip_domain",
    "has_suspicious_tld", "url_entropy_avg", "longest_url_len",
    "has_brand_in_domain",
    # Derived ratios
    "urgency_density", "financial_density", "threat_density",
    "caps_word_ratio",
]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 36 handcrafted features from raw text.
    Compatible with sklearn Pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(t) for t in X], dtype=np.float32)

    def _extract(self, text: str) -> list:
        t_lower = text.lower()
        words = text.split()
        n_words = max(len(words), 1)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)
        urls = extract_urls(text)

        # ── Keyword hits ──
        urgency_hits     = sum(1 for kw in URGENCY_WORDS if kw in t_lower)
        financial_hits   = sum(1 for kw in FINANCIAL_WORDS if kw in t_lower)
        threat_hits      = sum(1 for kw in THREAT_WORDS if kw in t_lower)
        social_eng_hits  = sum(1 for kw in SOCIAL_ENG_WORDS if kw in t_lower)
        impersonation    = sum(1 for b in IMPERSONATION_BRANDS if b in t_lower)
        manipulation     = sum(1 for p in MANIPULATION_PHRASES if p in t_lower)
        trust_hits       = sum(1 for t in TRUST_INDICATORS if t in t_lower)

        # ── Structural ──
        caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        excl          = text.count("!")
        quest         = text.count("?")
        avg_word_len  = float(np.mean([len(w) for w in words])) if words else 0.0
        avg_sent_len  = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0

        # ── Style ──
        double_space  = text.count("  ")
        dollar_signs  = text.count("$")
        number_blobs  = len(re.findall(r'\b\d{3,}\b', text))
        all_caps_w    = sum(1 for w in words if w.isupper() and len(w) > 2)
        ellipsis      = text.count("...")
        commas        = text.count(",")

        # ── URL features ──
        has_url = int(len(urls) > 0)
        domain_risks = []
        url_entropies = []
        has_http_only = 0
        has_shortener = 0
        has_ip = 0
        has_sus_tld = 0
        has_brand_domain = 0
        longest_url = 0

        for u in urls:
            parsed = urlparse(u)
            domain = parsed.netloc.lower()
            domain_risks.append(domain_risk_score(domain))
            url_entropies.append(char_entropy(domain))
            if u.startswith("http://"):
                has_http_only = 1
            if any(s in domain for s in URL_SHORTENERS):
                has_shortener = 1
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                has_ip = 1
            if any(tld in domain for tld in SUSPICIOUS_TLDS):
                has_sus_tld = 1
            for brand in IMPERSONATION_BRANDS:
                if brand.replace(' ', '') in domain:
                    has_brand_domain = 1
            longest_url = max(longest_url, len(u))

        max_domain_risk = max(domain_risks) if domain_risks else 0.0
        avg_domain_risk = float(np.mean(domain_risks)) if domain_risks else 0.0
        avg_url_ent     = float(np.mean(url_entropies)) if url_entropies else 0.0

        # ── Derived ratios ──
        urgency_density  = urgency_hits / n_words
        financial_density = financial_hits / n_words
        threat_density   = threat_hits / n_words
        caps_word_ratio  = all_caps_w / n_words

        return [
            urgency_hits, financial_hits, threat_hits,
            social_eng_hits, impersonation, manipulation, trust_hits,
            caps_ratio, excl, quest,
            len(urls), n_words, len(text), avg_word_len,
            n_sentences, avg_sent_len,
            double_space, dollar_signs, number_blobs,
            all_caps_w, ellipsis, commas,
            has_url, max_domain_risk, avg_domain_risk,
            has_http_only, has_shortener, has_ip,
            has_sus_tld, avg_url_ent, longest_url,
            has_brand_domain,
            urgency_density, financial_density, threat_density,
            caps_word_ratio,
        ]


def analyze_urls(text: str) -> list:
    """Return per-URL risk details for the report."""
    results = []
    for url in extract_urls(text):
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        flags = []
        score = 0

        if any(tld in domain for tld in SUSPICIOUS_TLDS):
            flags.append("Suspicious TLD"); score += 35
        if url.startswith("http://"):
            flags.append("No HTTPS encryption"); score += 20
        if any(s in domain for s in URL_SHORTENERS):
            flags.append("URL shortener (hides destination)"); score += 25
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            flags.append("IP address used as domain"); score += 40
        if domain.count('-') >= 3:
            flags.append("Excessive hyphens in domain"); score += 15
        if domain.count('.') >= 4:
            flags.append("Deep subdomain nesting"); score += 15
        if len(url) > 100:
            flags.append("Unusually long URL"); score += 10
        ent = char_entropy(domain)
        if ent > 3.6:
            flags.append(f"High domain entropy ({ent:.2f}) — looks randomly generated"); score += 20
        for brand in IMPERSONATION_BRANDS:
            if brand.replace(' ', '') in domain:
                flags.append(f"Brand name '{brand}' used in domain (impersonation risk)"); score += 35

        results.append({
            "url":    url,
            "domain": domain,
            "score":  min(100, score),
            "flags":  flags if flags else ["No suspicious indicators found"],
            "safe":   score < 25,
        })
    return results



def train_models():
    dataset_path = ensure_dataset_file()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("  THREAT·SCAN·AI — Model Training Pipeline")
    print("=" * 65)
    print(f"  Dataset file: {dataset_path}")
    print(f"  Samples: {len(TEXTS)}  |  Fraud: {sum(LABELS)}  |  Legit: {len(LABELS)-sum(LABELS)}")
    print()

    # ──────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # ──────────────────────────────────────────────────────────────

    print("[1/8] Preparing data & extracting features...")

    fe = FeatureExtractor()
    X_hand   = fe.transform(TEXTS)              # handcrafted 36-dim features
    X_text   = np.array(TEXTS)                  # raw text for TF-IDF models
    y        = np.array(LABELS)

    # Train / test split (stratified)
    (X_text_tr, X_text_te,
     X_hand_tr, X_hand_te,
     y_tr, y_te) = train_test_split(
        X_text, X_hand, y,
        test_size=0.22, random_state=42, stratify=y
    )

    print(f"  Train: {len(y_tr)} samples  |  Test: {len(y_te)} samples")

    # ──────────────────────────────────────────────────────────────
    # METRICS HELPER
    # ──────────────────────────────────────────────────────────────

    def evaluate(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred.astype(float)
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float("nan")

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        print(f"\n  ┌─ {name}")
        print(f"  │  Accuracy:  {acc*100:.1f}%")
        print(f"  │  F1 Score:  {f1*100:.1f}%")
        print(f"  │  Precision: {prec*100:.1f}%")
        print(f"  │  Recall:    {rec*100:.1f}%")
        print(f"  │  ROC-AUC:   {auc:.4f}")
        print(f"  │  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        return {"name": name, "acc": acc, "f1": f1, "precision": prec,
                "recall": rec, "auc": auc}


    # ──────────────────────────────────────────────────────────────
    # MODEL 1 — TF-IDF + Logistic Regression
    # ──────────────────────────────────────────────────────────────

    print("\n[2/8] Training Model 1: TF-IDF + Logistic Regression...")
    t0 = time.time()

    m1 = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=2.0, max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    m1.fit(X_text_tr, y_tr)
    m1_metrics = evaluate(m1, X_text_te, y_te, "TF-IDF + Logistic Regression")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")
    joblib.dump(m1, f"{MODELS_DIR}/m1_tfidf_lr.pkl")


    # ──────────────────────────────────────────────────────────────
    # MODEL 2 — Char N-Gram + LinearSVC (calibrated)
    # ──────────────────────────────────────────────────────────────

    print("\n[3/8] Training Model 2: Char N-Gram TF-IDF + LinearSVC...")
    t0 = time.time()

    svc_base = Pipeline([
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=4000,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
    ])

    m2 = CalibratedClassifierCV(svc_base, cv=3, method="sigmoid")
    m2.fit(X_text_tr, y_tr)
    m2_metrics = evaluate(m2, X_text_te, y_te, "Char N-Gram + SVC (calibrated)")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")
    joblib.dump(m2, f"{MODELS_DIR}/m2_chargram_svc.pkl")


    # ──────────────────────────────────────────────────────────────
    # MODEL 3 — Random Forest on handcrafted features
    # ──────────────────────────────────────────────────────────────

    print("\n[4/8] Training Model 3: Random Forest (300 trees)...")
    t0 = time.time()

    m3 = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    m3.fit(X_hand_tr, y_tr)
    m3_metrics = evaluate(m3, X_hand_te, y_te, "Random Forest (300 trees)")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")

    # Feature importance
    importances = m3.feature_importances_
    top_idx = np.argsort(importances)[::-1][:8]
    print("  │  Top features:")
    for i in top_idx:
        print(f"  │    {FEATURE_NAMES[i]:30s} {importances[i]:.4f}")
    joblib.dump(m3, f"{MODELS_DIR}/m3_random_forest.pkl")


    # ──────────────────────────────────────────────────────────────
    # MODEL 4 — Gradient Boosting on handcrafted features
    # ──────────────────────────────────────────────────────────────

    print("\n[5/8] Training Model 4: Gradient Boosting (200 estimators)...")
    t0 = time.time()

    m4 = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        min_samples_split=2,
        random_state=42,
    )
    m4.fit(X_hand_tr, y_tr)
    m4_metrics = evaluate(m4, X_hand_te, y_te, "Gradient Boosting (200 est.)")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")
    joblib.dump(m4, f"{MODELS_DIR}/m4_gradient_boost.pkl")


    # ──────────────────────────────────────────────────────────────
    # MODEL 5 — MLP Neural Network (Deep Learning)
    # ──────────────────────────────────────────────────────────────

    print("\n[6/8] Training Model 5: MLP Neural Network (Deep Learning)...")
    t0 = time.time()

    # Scale features for neural net
    scaler = StandardScaler()
    X_hand_tr_sc = scaler.fit_transform(X_hand_tr)
    X_hand_te_sc = scaler.transform(X_hand_te)

    m5 = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        batch_size=16,
    )
    m5.fit(X_hand_tr_sc, y_tr)
    m5_metrics = evaluate(m5, X_hand_te_sc, y_te, "MLP Neural Network (3-layer)")
    print(f"  │  Layers: Input(36) → 128 → 64 → 32 → Output(2)")
    print(f"  │  Activation: ReLU | Solver: Adam | Epochs: {m5.n_iter_}")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")
    joblib.dump(m5, f"{MODELS_DIR}/m5_mlp_neural_net.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")


    # ──────────────────────────────────────────────────────────────
    # MODEL 6 — Naive Bayes (TF-IDF word counts)
    # ──────────────────────────────────────────────────────────────

    print("\n[7/8] Training Model 6: Multinomial Naive Bayes (TF-IDF)...")
    t0 = time.time()

    m6 = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1,
        )),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    # MinMaxScaler to keep values non-negative for MultinomialNB
    # Already ensured by TF-IDF (values >= 0)
    m6.fit(X_text_tr, y_tr)
    m6_metrics = evaluate(m6, X_text_te, y_te, "Multinomial Naive Bayes")
    print(f"  └─ Trained in {time.time()-t0:.2f}s")
    joblib.dump(m6, f"{MODELS_DIR}/m6_naive_bayes.pkl")


    # ──────────────────────────────────────────────────────────────
    # CROSS-VALIDATION on best models
    # ──────────────────────────────────────────────────────────────

    print("\n[8/8] Cross-Validation (5-fold stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model, X, name in [
        (m1, X_text, "TF-IDF LR"),
        (m3, X_hand, "Random Forest"),
        (m4, X_hand, "Grad Boost"),
        (m5, scaler.transform(X_hand), "MLP Neural Net"),
    ]:
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        print(f"  {name:22s}  CV F1: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")


    # ──────────────────────────────────────────────────────────────
    # SAVE FEATURE EXTRACTOR + METADATA
    # ──────────────────────────────────────────────────────────────

    joblib.dump(fe, f"{MODELS_DIR}/feature_extractor.pkl")

    metadata = {
        "sklearn_version": sklearn.__version__,
        "n_train":   int(len(y_tr)),
        "n_test":    int(len(y_te)),
        "n_total":   int(len(y)),
        "n_fraud":   int(sum(y)),
        "n_legit":   int(len(y) - sum(y)),
        "feature_names": FEATURE_NAMES,
        "models": {
            "m1": {"file": "m1_tfidf_lr.pkl",       "type": "text",  "desc": "TF-IDF + Logistic Regression"},
            "m2": {"file": "m2_chargram_svc.pkl",    "type": "text",  "desc": "Char N-Gram + SVC (calibrated)"},
            "m3": {"file": "m3_random_forest.pkl",   "type": "hand",  "desc": "Random Forest (300 trees)"},
            "m4": {"file": "m4_gradient_boost.pkl",  "type": "hand",  "desc": "Gradient Boosting (200 est.)"},
            "m5": {"file": "m5_mlp_neural_net.pkl",  "type": "hand_scaled", "desc": "MLP Neural Net (128→64→32)"},
            "m6": {"file": "m6_naive_bayes.pkl",     "type": "text",  "desc": "Multinomial Naive Bayes"},
        },
        "ensemble_weights": {"m1": 0.25, "m2": 0.20, "m3": 0.18,
                             "m4": 0.17, "m5": 0.12, "m6": 0.08},
        "metrics": {
            "m1": m1_metrics, "m2": m2_metrics, "m3": m3_metrics,
            "m4": m4_metrics, "m5": m5_metrics, "m6": m6_metrics,
        }
    }

    with open(f"{MODELS_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────
    # TRAINING REPORT
    # ──────────────────────────────────────────────────────────────

    all_metrics = [m1_metrics, m2_metrics, m3_metrics, m4_metrics, m5_metrics, m6_metrics]
    avg_acc = np.mean([m["acc"] for m in all_metrics]) * 100
    avg_f1  = np.mean([m["f1"]  for m in all_metrics]) * 100

    report = f"""
    ═══════════════════════════════════════════════════════════════
      THREAT·SCAN·AI — TRAINING REPORT
    ═══════════════════════════════════════════════════════════════

    Dataset
      Total samples : {len(y)}
      Fraud / Scam  : {sum(y)}
      Legitimate    : {len(y) - sum(y)}
      Train split   : {len(y_tr)} ({len(y_tr)/len(y)*100:.0f}%)
      Test split    : {len(y_te)} ({len(y_te)/len(y)*100:.0f}%)

    Model Performance (held-out test set)
      {'Model':<35} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}
      {'─'*65}
    """
    for m in all_metrics:
        report += (f"  {m['name']:<35} {m['acc']*100:>5.1f}% {m['f1']*100:>5.1f}% "
                   f"{m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% {m['auc']:>6.3f}\n")
    report += f"""
      Average Accuracy : {avg_acc:.1f}%
      Average F1 Score : {avg_f1:.1f}%

    Models Saved to: models/
      m1_tfidf_lr.pkl          TF-IDF + Logistic Regression
      m2_chargram_svc.pkl      Char N-Gram + SVC (calibrated)
      m3_random_forest.pkl     Random Forest (300 trees)
      m4_gradient_boost.pkl    Gradient Boosting (200 estimators)
      m5_mlp_neural_net.pkl    MLP Neural Network (128→64→32)
      m6_naive_bayes.pkl       Multinomial Naive Bayes
      feature_extractor.pkl    Handcrafted feature transformer
      scaler.pkl               StandardScaler for neural network
      metadata.json            Model config & performance metrics

    ═══════════════════════════════════════════════════════════════
    """
    report = textwrap.dedent(report).strip() + "\n"
    print(report)

    with open(f"{MODELS_DIR}/training_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("✓ All models saved successfully.")
    print("  Run:  python -m streamlit run Threat_scan.ai.py")


def load_models():
    """Load all trained models from disk."""
    required = [
        "m1_tfidf_lr.pkl", "m2_chargram_svc.pkl",
        "m3_random_forest.pkl", "m4_gradient_boost.pkl",
        "m5_mlp_neural_net.pkl", "m6_naive_bayes.pkl",
        "feature_extractor.pkl", "scaler.pkl", "metadata.json",
    ]
    missing = [f for f in required if not os.path.exists(f"{MODELS_DIR}/{f}")]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Please run:  python Threat_scan.ai.py --train  first."
        )

    with open(f"{MODELS_DIR}/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    trained_version = meta.get("sklearn_version")
    if trained_version != sklearn.__version__:
        version_detail = trained_version or "unknown"
        raise RuntimeError(
            f"Trained models use scikit-learn {version_detail}, but "
            f"{sklearn.__version__} is installed in {sys.executable}.\n"
            f'Install the project dependencies with: "{sys.executable}" -m pip install -r requirements.txt\n'
            f'Then launch the app with the same interpreter: "{sys.executable}" -m streamlit run Threat_scan.ai.py'
        )

    return {
        "m1":  joblib.load(f"{MODELS_DIR}/m1_tfidf_lr.pkl"),
        "m2":  joblib.load(f"{MODELS_DIR}/m2_chargram_svc.pkl"),
        "m3":  joblib.load(f"{MODELS_DIR}/m3_random_forest.pkl"),
        "m4":  joblib.load(f"{MODELS_DIR}/m4_gradient_boost.pkl"),
        "m5":  joblib.load(f"{MODELS_DIR}/m5_mlp_neural_net.pkl"),
        "m6":  joblib.load(f"{MODELS_DIR}/m6_naive_bayes.pkl"),
        "fe":  joblib.load(f"{MODELS_DIR}/feature_extractor.pkl"),
        "sc":  joblib.load(f"{MODELS_DIR}/scaler.pkl"),
        "meta": meta,
    }


# ──────────────────────────────────────────────────────────────
# RULE-BASED NLP (runs in parallel with ML models)
# ──────────────────────────────────────────────────────────────

def rule_based_scores(text: str) -> dict:
    t = text.lower()
    words = text.split()
    n = max(len(words), 1)

    urgency_hits  = [kw for kw in URGENCY_WORDS if kw in t]
    fin_hits      = [kw for kw in FINANCIAL_WORDS if kw in t]
    threat_hits   = [kw for kw in THREAT_WORDS if kw in t]
    social_hits   = [kw for kw in SOCIAL_ENG_WORDS if kw in t]
    imp_hits      = [b  for b  in IMPERSONATION_BRANDS if b in t]
    manip_hits    = [p  for p  in MANIPULATION_PHRASES if p in t]

    urgency_score  = min(100, len(urgency_hits)  * 16 + text.count("!") * 4)
    financial_score= min(100, len(fin_hits)      * 12)
    threat_score   = min(100, len(threat_hits)   * 18)
    social_score   = min(100, len(social_hits)   * 22)
    imp_score      = min(100, len(imp_hits)       * 14)
    manip_score    = min(100, len(manip_hits)     * 14)

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    ling_anomaly = min(100, int(
        caps_ratio * 60 +
        text.count("!") / n * 150 +
        (20 if len([w for w in words if w.isupper() and len(w) > 2]) > 2 else 0)
    ))

    manipulation = min(100, int(
        urgency_score  * 0.35 +
        threat_score   * 0.30 +
        social_score   * 0.20 +
        manip_score    * 0.15
    ))

    # Simple sentiment
    neg = ["urgent", "warning", "alert", "danger", "critical", "threat",
           "arrest", "suspend", "freeze", "close", "delete", "expire", "penalty"]
    pos = ["safe", "secure", "verified", "confirmed", "thank", "welcome",
           "approved", "delivered", "success", "scheduled"]
    neg_c = sum(1 for w in neg if w in t)
    pos_c = sum(1 for w in pos if w in t)
    sentiment = (pos_c - neg_c) / max(neg_c + pos_c, 1)
    sentiment = max(-1.0, min(1.0, sentiment))

    all_flags = urgency_hits + fin_hits[:3] + threat_hits + social_hits[:2]
    flagged_kw = list(dict.fromkeys(all_flags))[:10]

    return {
        "urgency_score":   urgency_score,
        "financial_score": financial_score,
        "threat_score":    threat_score,
        "social_score":    social_score,
        "impersonation_score": imp_score,
        "linguistic_anomaly":  ling_anomaly,
        "manipulation_score":  manipulation,
        "sentiment":       round(sentiment, 3),
        "flagged_keywords": flagged_kw,
    }


# ──────────────────────────────────────────────────────────────
# FULL ANALYSIS
# ──────────────────────────────────────────────────────────────

def analyze(text: str, models: dict) -> dict:
    """
    Run full hybrid analysis:
      • 6 ML/DL models (trained)
      • Rule-based NLP engine
      • URL feature extractor
      • Weighted ensemble voting
    """
    fe  = models["fe"]
    sc  = models["sc"]
    meta = models["meta"]
    w   = meta["ensemble_weights"]

    # Feature matrices
    X_hand = fe.transform([text])
    X_sc   = sc.transform(X_hand)

    # ── Get probabilities from each model ──
    def prob(model, X):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0][1])
        return float(model.decision_function(X)[0])

    p1 = prob(models["m1"], [text])
    p2 = prob(models["m2"], [text])
    p3 = prob(models["m3"], X_hand)
    p4 = prob(models["m4"], X_hand)
    p5 = prob(models["m5"], X_sc)
    p6 = prob(models["m6"], [text])

    # Clip to [0, 1]
    p1, p2, p3, p4, p5, p6 = [
        max(0.0, min(1.0, p)) for p in (p1, p2, p3, p4, p5, p6)
    ]

    # Rule-based probability
    rules = rule_based_scores(text)
    p_rule = min(1.0, (
        rules["urgency_score"]    * 0.22 +
        rules["financial_score"]  * 0.20 +
        rules["threat_score"]     * 0.18 +
        rules["social_score"]     * 0.15 +
        rules["impersonation_score"] * 0.15 +
        rules["linguistic_anomaly"]  * 0.10
    ) / 100)

    # URL risk
    url_details = analyze_urls(text)
    url_risk = max((u["score"] for u in url_details), default=0) / 100

    # ── Ensemble: weighted average ──
    ensemble = (
        p1 * w["m1"] +
        p2 * w["m2"] +
        p3 * w["m3"] +
        p4 * w["m4"] +
        p5 * w["m5"] +
        p6 * w["m6"]
    )
    # Boost ensemble if URL is highly suspicious
    ensemble = min(1.0, ensemble + url_risk * 0.12 + p_rule * 0.08)

    fraud_prob = int(round(ensemble * 100))
    is_fraud   = ensemble >= 0.42

    # ── Risk score ──
    risk_score = min(100, int(
        ensemble * 72 +
        url_risk * 15 +
        rules["urgency_score"]   * 0.08 +
        rules["linguistic_anomaly"] * 0.05
    ))

    # ── Verdict ──
    if   risk_score >= 80: verdict = "CRITICAL DANGER"
    elif risk_score >= 60: verdict = "HIGH RISK"
    elif risk_score >= 38: verdict = "MODERATE RISK"
    elif risk_score >= 18: verdict = "LOW RISK"
    else:                  verdict = "SAFE"

    # ── Threat categories ──
    threat_cats = [
        {"name": "Phishing",             "score": min(100, int(p1*55 + url_risk*45)),   "detected": p1 > 0.45 or url_risk > 0.45},
        {"name": "Social Engineering",   "score": rules["social_score"],                "detected": rules["social_score"] > 28},
        {"name": "Financial Scam",       "score": rules["financial_score"],             "detected": rules["financial_score"] > 28},
        {"name": "Urgency Manipulation", "score": rules["urgency_score"],               "detected": rules["urgency_score"] > 22},
        {"name": "Impersonation",        "score": rules["impersonation_score"],         "detected": rules["impersonation_score"] > 18},
        {"name": "Malware / Ransomware", "score": min(100, int(url_risk*80 + p2*20)),   "detected": url_risk > 0.50},
        {"name": "Identity Theft",       "score": min(100, int(p3*50 + rules["financial_score"]*0.5)), "detected": p3 > 0.55},
        {"name": "Linguistic Anomaly",   "score": rules["linguistic_anomaly"],          "detected": rules["linguistic_anomaly"] > 28},
    ]

    # ── ML signal breakdown ──
    model_info = meta["models"]
    ml_signals = [
        {"name": model_info["m1"]["desc"], "prob": p1, "pct": int(p1*100),
         "triggered": p1 > 0.45,
         "detail": "Word & phrase pattern fraud classification (TF-IDF 5000 features, trigrams)"},
        {"name": model_info["m2"]["desc"], "prob": p2, "pct": int(p2*100),
         "triggered": p2 > 0.45,
         "detail": "Character-level (2–5 gram) pattern detection — catches obfuscated scam text"},
        {"name": model_info["m3"]["desc"], "prob": p3, "pct": int(p3*100),
         "triggered": p3 > 0.45,
         "detail": f"300-tree ensemble on {len(fe._extract(''))} handcrafted features"},
        {"name": model_info["m4"]["desc"], "prob": p4, "pct": int(p4*100),
         "triggered": p4 > 0.45,
         "detail": "Sequential boosting on handcrafted features — excels at subtle patterns"},
        {"name": model_info["m5"]["desc"], "prob": p5, "pct": int(p5*100),
         "triggered": p5 > 0.45,
         "detail": "3-layer neural network (Input→128→64→32→Output), ReLU, Adam optimizer"},
        {"name": model_info["m6"]["desc"], "prob": p6, "pct": int(p6*100),
         "triggered": p6 > 0.45,
         "detail": "Probabilistic bigram model — fast baseline using Bayes theorem"},
        {"name": "Rule-Based NLP Engine", "prob": p_rule, "pct": int(p_rule*100),
         "triggered": p_rule > 0.35,
         "detail": "Expert lexicon matching across urgency, financial, threat, and social-engineering patterns"},
    ]

    # ── Deep learning insights ──
    dl_insights = {
        "urgency_level":    rules["urgency_score"],
        "manipulation":     rules["manipulation_score"],
        "linguistic_anomaly": rules["linguistic_anomaly"],
        "financial_exposure": rules["financial_score"],
        "social_engineering": rules["social_score"],
        "sentiment":        rules["sentiment"],
        "mlp_confidence":   int(p5 * 100),
        "ensemble_certainty": int(abs(ensemble - 0.5) * 200),  # distance from decision boundary
    }

    # ── Recommendations ──
    recs = []
    if verdict in ("CRITICAL DANGER", "HIGH RISK"):
        recs.append("Do NOT click any links or call any phone numbers in this message.")
        recs.append("Report this to your IT/security team or relevant authorities immediately.")
        recs.append("Block the sender and delete this message without replying.")
    if url_risk > 0.4:
        recs.append("Avoid all URLs in this message — they show signs of malicious hosting (suspicious domain, no HTTPS, or IP-based).")
    if rules["financial_score"] > 25:
        recs.append("Never share bank details, SSN, gift card codes, or payment information via unsolicited messages.")
    if rules["impersonation_score"] > 15:
        recs.append("If this claims to be from a known brand, visit their official website directly — do not use links in this message.")
    if verdict == "MODERATE RISK":
        recs.append("Treat this message with caution. Verify the sender through an independent channel before taking any action.")
    if verdict in ("SAFE", "LOW RISK"):
        recs.append("This message appears legitimate. Always stay vigilant for unexpected requests for personal information.")
    recs = (recs or ["No specific threats detected. Continue practicing safe messaging habits."])[:5]

    # ── Explanation ──
    top_model = max(ml_signals[:6], key=lambda x: x["pct"])
    if is_fraud:
        explanation = (
            f"This content was classified as potentially FRAUDULENT by {sum(1 for s in ml_signals[:6] if s['triggered'])} "
            f"out of 6 ML/DL models, with a weighted ensemble fraud probability of {fraud_prob}%. "
            f"The strongest signal was {top_model['name']} ({top_model['pct']}%). "
            f"Key triggers: {', '.join(rules['flagged_keywords'][:4]) or 'pattern-based signals'}."
        )
    else:
        models_safe = sum(1 for s in ml_signals[:6] if not s["triggered"])
        explanation = (
            f"This content appears {verdict.lower()} — {models_safe}/6 models classified it as legitimate. "
            f"Ensemble fraud probability: {fraud_prob}%. "
            f"{'No significant threat indicators found.' if risk_score < 20 else 'Some mild signals present but below fraud threshold.'}"
        )

    return {
        "risk_score":       risk_score,
        "verdict":          verdict,
        "fraud_probability": fraud_prob,
        "is_fraud":         is_fraud,
        "explanation":      explanation,
        "flagged_keywords": rules["flagged_keywords"],
        "threat_categories": threat_cats,
        "ml_signals":       ml_signals,
        "dl_insights":      dl_insights,
        "url_details":      url_details,
        "url_risk":         int(url_risk * 100),
        "recommendations":  recs,
        "raw_probs": {
            "m1_lr":       round(p1, 4),
            "m2_svc":      round(p2, 4),
            "m3_rf":       round(p3, 4),
            "m4_gb":       round(p4, 4),
            "m5_mlp":      round(p5, 4),
            "m6_nb":       round(p6, 4),
            "rules":       round(p_rule, 4),
            "ensemble":    round(ensemble, 4),
        },
    }


def run_app():
    st.set_page_config(
        page_title="THREAT·SCAN·AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ──────────────────────────────────────────────────────────────
    # CSS
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body, .stApp {
        background-color: #020407 !important;
        color: #b8ccd8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stApp::before {
        content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
        background-image:
            linear-gradient(rgba(0,229,255,.025) 1px,transparent 1px),
            linear-gradient(90deg,rgba(0,229,255,.025) 1px,transparent 1px);
        background-size:38px 38px;
    }
    #MainMenu,footer,header { display:none !important; }
    .block-container { padding:1.5rem 2rem 4rem !important; max-width:1180px !important; }

    .hero {
        position:relative; overflow:hidden; text-align:center; margin:1rem 0 1.7rem;
        padding:2rem 1.4rem 1.5rem; border-radius:16px;
        border:1px solid rgba(0,229,255,.18);
        background:radial-gradient(ellipse at 50% 0%,rgba(0,229,255,.12),transparent 62%),
          linear-gradient(145deg,rgba(7,18,28,.98),rgba(3,7,12,.96));
        box-shadow:0 18px 60px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.035);
    }
    .hero-kicker { font-family:'Share Tech Mono',monospace; font-size:10px;
        letter-spacing:3px; color:#70e8ef; text-transform:uppercase; }
    .hero-sub { margin:.65rem 0 1.1rem; color:rgba(220,240,248,.62);
        font-size:12px; letter-spacing:1px; }
    .model-badges { display:flex; justify-content:center; gap:8px;
        flex-wrap:wrap; max-width:900px; margin:0 auto; }
    .model-badge { padding:6px 11px; border-radius:999px;
        border:1px solid rgba(118,255,3,.2); background:rgba(118,255,3,.055);
        color:rgba(180,255,118,.8); font:9px 'Share Tech Mono',monospace;
        letter-spacing:.8px; }
    .hero-meta { display:flex; justify-content:center; gap:9px; flex-wrap:wrap;
        margin-top:16px; color:rgba(255,255,255,.42);
        font:9px 'Share Tech Mono',monospace; letter-spacing:1.5px; }
    .hero-meta span { padding:5px 9px; border-radius:4px;
        background:rgba(255,255,255,.035); }

    /* Title */
    .title-main { font-family:'Rajdhani',sans-serif !important; font-weight:700 !important;
        font-size:clamp(38px,7vw,64px) !important; letter-spacing:6px; line-height:1.05;
        background:linear-gradient(135deg,#00e5ff,#0097a7,rgba(255,255,255,.45));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    /* Panel */
    .panel { background:rgba(6,10,16,.97); border:1px solid rgba(0,229,255,.1);
        border-radius:10px; padding:20px; margin-bottom:14px;
        box-shadow:0 10px 32px rgba(0,0,0,.12); }
    .ptitle { font-family:'Share Tech Mono',monospace; font-size:9px;
        letter-spacing:4px; color:rgba(255,255,255,.25); margin-bottom:14px; }

    /* Verdict */
    .verdict { display:inline-block; padding:8px 28px; font-family:'Share Tech Mono',monospace;
        font-size:14px; font-weight:bold; letter-spacing:4px; border-radius:2px; margin-bottom:14px; }
    .v-safe     { background:rgba(0,229,255,.07); border:1px solid rgba(0,229,255,.5); color:#00e5ff; }
    .v-low      { background:rgba(118,255,3,.07); border:1px solid rgba(118,255,3,.5); color:#76ff03; }
    .v-moderate { background:rgba(255,214,0,.07); border:1px solid rgba(255,214,0,.5); color:#ffd600; }
    .v-high     { background:rgba(255,109,0,.07); border:1px solid rgba(255,109,0,.5); color:#ff6d00; }
    .v-critical { background:rgba(255,23,68,.07);  border:1px solid rgba(255,23,68,.5);  color:#ff1744;
        animation:cpulse 1.4s infinite alternate; }

    /* Metric card */
    .mc  { background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06);
        border-radius:9px; padding:16px 10px; text-align:center;
        transition:transform .2s ease,border-color .2s ease,background .2s ease; }
    .mc:hover { transform:translateY(-2px); border-color:rgba(0,229,255,.22);
        background:rgba(0,229,255,.035); }
    .mv  { font-family:'Rajdhani',sans-serif; font-size:42px; font-weight:700; line-height:1; }
    .ml  { font-family:'Share Tech Mono',monospace; font-size:9px;
        letter-spacing:3px; color:rgba(255,255,255,.26); margin-top:5px; }

    /* Progress bar */
    .pb { margin-bottom:9px; }
    .pbh { display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace;
        font-size:10px; margin-bottom:3px; }
    .pbt { height:4px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden; }
    .pbf { height:100%; border-radius:2px; transition:width .8s ease; }

    /* Threat grid */
    .tg  { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
    .ti  { padding:9px 13px; border-radius:2px; display:flex;
        justify-content:space-between; align-items:center;
        font-family:'Share Tech Mono',monospace; font-size:10px; }
    .ti1 { background:rgba(255,23,68,.07); border:1px solid rgba(255,23,68,.28); color:#ff6b8a; }
    .ti0 { background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06); color:rgba(255,255,255,.2); }

    /* Signal card */
    .sc  { padding:14px; border-radius:2px; margin-bottom:9px; }
    .sc1 { background:rgba(255,109,0,.05); border:1px solid rgba(255,109,0,.25); }
    .sc0 { background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06); }

    /* URL block */
    .ub  { padding:14px; background:rgba(255,255,255,.02);
        border:1px solid rgba(255,255,255,.07); border-radius:2px; margin-bottom:12px; }
    .uc  { padding:12px; border-radius:2px; text-align:center; }
    .uc-d { background:rgba(255,23,68,.07); border:1px solid rgba(255,23,68,.25); }
    .uc-s { background:rgba(118,255,3,.06); border:1px solid rgba(118,255,3,.2); }
    .uc-n { background:rgba(0,229,255,.05); border:1px solid rgba(0,229,255,.14); }

    /* Keyword chip */
    .chip { display:inline-block; padding:2px 11px; border-radius:2px; margin:3px;
        background:rgba(255,23,68,.09); border:1px solid rgba(255,23,68,.28);
        color:#ff7597; font-family:'Share Tech Mono',monospace; font-size:10px; }

    /* Rec item */
    .ri  { display:flex; gap:13px; align-items:flex-start; padding:13px; border-radius:2px;
        margin-bottom:9px; background:rgba(0,229,255,.04); border:1px solid rgba(0,229,255,.13); }
    .rn  { min-width:27px; height:27px; border-radius:50%; background:rgba(0,229,255,.13);
        border:1px solid rgba(0,229,255,.38); display:flex; align-items:center; justify-content:center;
        color:#00e5ff; font-family:'Share Tech Mono',monospace; font-size:11px; flex-shrink:0; }
    .rt  { font-size:13px; color:rgba(255,255,255,.65); line-height:1.6; }

    /* Log */
    .logbox { background:#000c; border:1px solid rgba(0,229,255,.1); border-radius:2px;
        padding:13px; font-family:'Share Tech Mono',monospace; font-size:11px;
        max-height:160px; overflow-y:auto; }
    .ll { margin-bottom:2px; }

    /* Buttons */
    .stButton>button { background:rgba(0,229,255,.07) !important;
        border:1px solid rgba(0,229,255,.44) !important; border-radius:2px !important;
        color:#00e5ff !important; font-family:'Share Tech Mono',monospace !important;
        letter-spacing:3px !important; font-size:12px !important; transition:all .3s !important; }
    .stButton>button:hover { background:rgba(0,229,255,.16) !important;
        box-shadow:0 0 26px rgba(0,229,255,.22) !important; }
    .stButton>button:focus-visible { outline:2px solid #76ff03 !important;
        outline-offset:3px !important; }

    /* Textarea */
    .stTextArea textarea { background:rgba(0,0,0,.7) !important;
        border:1px solid rgba(0,229,255,.18) !important; border-radius:2px !important;
        color:#c9d8e8 !important; font-family:'JetBrains Mono',monospace !important;
        font-size:13px !important; }
    .stTextArea textarea:focus { border-color:rgba(0,229,255,.5) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:transparent !important;
        border-bottom:1px solid rgba(255,255,255,.07) !important; gap:3px; }
    .stTabs [data-baseweb="tab"] { background:transparent !important;
        border:1px solid transparent !important; border-radius:2px 2px 0 0 !important;
        color:rgba(255,255,255,.27) !important; font-family:'Share Tech Mono',monospace !important;
        font-size:10px !important; letter-spacing:2px !important; }
    .stTabs [aria-selected="true"] { background:rgba(0,229,255,.08) !important;
        border-color:rgba(0,229,255,.38) !important; color:#00e5ff !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top:18px !important; }

    @media (max-width:700px) {
        .block-container { padding:.75rem .8rem 2.5rem !important; }
        .hero { padding:1.5rem .8rem 1.1rem; }
        .title-main { letter-spacing:3px; }
        .hero-kicker { letter-spacing:1.5px; }
        .tg { grid-template-columns:1fr; }
        .stTabs [data-baseweb="tab"] { font-size:9px !important; letter-spacing:.5px !important; }
        .panel { padding:15px; }
    }

    .div { border:none; border-top:1px solid rgba(255,255,255,.06); margin:13px 0; }
    .footer { text-align:center; margin-top:34px; padding-top:16px;
        border-top:1px solid rgba(255,255,255,.04);
        font-family:'Share Tech Mono',monospace; font-size:9px;
        letter-spacing:3px; color:rgba(255,255,255,.09); }
    @keyframes cpulse { from{box-shadow:0 0 10px rgba(255,23,68,.2);}
        to{box-shadow:0 0 30px rgba(255,23,68,.5);} }
    </style>
    """, unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────
    def rc(s):
        if s >= 80: return "#ff1744"
        if s >= 60: return "#ff6d00"
        if s >= 40: return "#ffd600"
        if s >= 20: return "#76ff03"
        return "#00e5ff"

    def pbar(lbl, val, color):
        return f"""<div class="pb">
      <div class="pbh"><span style="color:{color};">{lbl}</span><span style="color:{color};">{val}%</span></div>
      <div class="pbt"><div class="pbf" style="width:{val}%;background:{color};box-shadow:0 0 7px {color};"></div></div>
    </div>"""

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    def log_html(lines):
        h = '<div class="logbox">'
        for t, msg, col in lines[-14:]:
            h += f'<div class="ll"><span style="color:rgba(255,255,255,.18);">[{t}]</span> <span style="color:{col};">{msg}</span></div>'
        h += "</div>"
        return h


    # ──────────────────────────────────────────────────────────────
    # SESSION STATE
    # ──────────────────────────────────────────────────────────────
    if "logs" not in st.session_state:
        st.session_state.logs = [
            (ts(), "SYSTEM BOOT — Loading trained ML models from disk...", "#00e5ff66"),
            (ts(), "Models: TF-IDF LR | Char SVC | Random Forest | Grad Boost | MLP Neural Net | Naive Bayes", "#76ff0366"),
            (ts(), "Feature extractor loaded. Rule-based NLP engine ready.", "#76ff0366"),
            (ts(), "Awaiting input for threat analysis.", "#ffffff33"),
        ]
    if "result"  not in st.session_state: st.session_state.result = None
    if "scans"   not in st.session_state: st.session_state.scans  = 0

    def log(msg, col="#00e5ff77"):
        st.session_state.logs.append((ts(), msg, col))


    # ──────────────────────────────────────────────────────────────
    # LOAD MODELS (cached)
    # ──────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def get_models():
        return load_models()

    try:
        models = get_models()
        model_loaded = True
    except FileNotFoundError as e:
        model_loaded = False
        model_error = str(e)
    except RuntimeError as e:
        model_loaded = False
        model_error = str(e)


    # ══════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">LOCAL-FIRST MESSAGE DEFENSE · CYBERSECURITY PROJECT</div>
      <div class="title-main" style="margin-top:12px;">THREAT·SCAN·AI</div>
      <div class="hero-sub">
        Fraud, phishing and scam analysis powered by a six-model hybrid ensemble
      </div>
      <div class="model-badges">
    """ + "".join(
            f'<span class="model-badge">● {m}</span>'
            for m in [
            "TF-IDF + LOG.REG", "CHAR N-GRAM + SVC", "RANDOM FOREST",
            "GRADIENT BOOST", "MLP NEURAL NET", "NAIVE BAYES"
            ]
        ) + """
      </div>
      <div class="hero-meta"><span>100% LOCAL PROCESSING</span><span>NO API REQUIRED</span>
        <span>36 HANDCRAFTED FEATURES</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # Model not trained guard
    # ──────────────────────────────────────────────────────────────
    if not model_loaded:
        st.error(f"⚠ {model_error}")
        st.code("python Threat_scan.ai.py --train", language="bash")
        st.stop()

    meta = models["meta"]


    # ══════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:10px;
        letter-spacing:3px;color:rgba(0,229,255,.5);margin-bottom:12px;">◈ MODEL INFO</div>""",
        unsafe_allow_html=True)

        n_tr = meta.get("n_train", "–")
        n_te = meta.get("n_test",  "–")
        for lbl, val in [("Training Samples", n_tr), ("Test Samples", n_te),
                         ("Total Features", "36 handcrafted + TF-IDF"), ("Ensemble Models", "6")]:
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
              letter-spacing:2px;color:rgba(255,255,255,.2);">{lbl}</div>
              <div style="font-family:'Rajdhani',sans-serif;font-size:20px;
              font-weight:700;color:#00e5ff;">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border:1px solid rgba(255,255,255,.05);margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
        letter-spacing:2px;color:rgba(255,255,255,.2);margin-bottom:8px;">SCANS THIS SESSION</div>""",
        unsafe_allow_html=True)
        st.markdown(f"""<div style="font-family:'Rajdhani',sans-serif;font-size:36px;
        font-weight:700;color:#00e5ff;">{st.session_state.scans:03d}</div>""",
        unsafe_allow_html=True)

        st.markdown("<hr style='border:1px solid rgba(255,255,255,.05);margin:12px 0;'>", unsafe_allow_html=True)

        SAMPLES = {
            "— Select sample —": "",
            "🟢 Safe — Casual chat": "Hey! Are we still meeting for lunch tomorrow at 12:30? Let me know if the plan changes.",
            "🟢 Safe — Work email": "Hi team, the project deadline has been extended to the 20th. Please update your timelines accordingly.",
            "🟡 Moderate — Suspicious verify": "Dear customer, we've noticed unusual activity on your account. Please verify your details at http://secure-paypal-check.top/login",
            "🔴 High — Lottery scam": "CONGRATULATIONS! You have WON $1,000,000 in our international lottery! CLAIM NOW before it EXPIRES in 24 HOURS! Click: http://bit.ly/claim-prize-2024 Reply with your bank account number and SSN!",
            "⚫ Critical — IRS arrest": "IRS FINAL NOTICE: An arrest warrant has been issued in your name for tax fraud. You owe $3,800. Pay IMMEDIATELY via iTunes gift cards or wire transfer to avoid arrest. Call 1-800-555-FAKE. Provide your SSN and routing number now.",
            "🔵 Phishing — Apple ID": "Your Apple ID has been locked due to suspicious activity. Verify NOW at http://apple-id-secure-login.xyz/verify or your account will be permanently deleted within 24 hours. Enter your password and payment info.",
            "💜 Advance fee — Nigerian prince": "DEAR FRIEND, I am Prince Abdullah from Nigeria and need your help to transfer $45 million USD out of the country. You will receive 30% commission. God bless you. Reply with your bank account details urgently.",
        }

        def apply_sample():
            st.session_state.scan_text = SAMPLES[st.session_state.sample_choice]

        choice = st.selectbox(
            "Quick test samples:",
            list(SAMPLES.keys()),
            key="sample_choice",
            on_change=apply_sample,
        )


    # ══════════════════════════════════════════════════════════════
    #  INPUT PANEL
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="panel"><div class="ptitle">◈ INPUT TARGET FOR ANALYSIS</div>', unsafe_allow_html=True)

    col_ta, col_st = st.columns([3, 1])
    default_txt = SAMPLES.get(choice, "")

    with col_ta:
        user_input = st.text_area("Message or URL to scan", value=default_txt, height=178,
            placeholder="Paste any message, link, email, or SMS here...\nAll processing is 100% local — no data leaves your machine.",
            label_visibility="collapsed", key="scan_text")

    with col_st:
        wc = len(user_input.split()) if user_input.strip() else 0
        cc = len(user_input)
        uc = len(re.findall(r'https?://[^\s]+', user_input))
        st.markdown(f"""
        <div style="padding:14px;background:rgba(255,255,255,.02);
        border:1px solid rgba(255,255,255,.06);border-radius:2px;height:178px;
        display:flex;flex-direction:column;justify-content:space-around;">
        {''.join(f"""<div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
          letter-spacing:2px;color:rgba(255,255,255,.2);">{lbl}</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:30px;
          font-weight:700;color:{col};">{val}</div>
        </div>""" for lbl, val, col in [
            ("CHARACTERS", cc, "#00e5ff"),
            ("WORDS", wc, "#00e5ff"),
            ("URLs", uc, "#ff6d00" if uc else "#00e5ff"),
        ])}
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 5])
    with b1: scan_btn = st.button("⚡  INITIATE SCAN", use_container_width=True)
    with b2:
        if st.button("✕  CLEAR", use_container_width=True):
            st.session_state.result = None
            st.session_state.scan_text = ""
            st.rerun()

    # Log
    st.markdown("""<div style="margin-top:14px;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:4px;
    color:rgba(255,255,255,.18);margin-bottom:7px;">◈ SYSTEM LOG</div>""", unsafe_allow_html=True)
    log_ph = st.empty()
    log_ph.markdown(log_html(st.session_state.logs), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════
    #  SCAN
    # ══════════════════════════════════════════════════════════════
    if scan_btn:
        if not user_input.strip():
            st.warning("⚠ Please enter a message or URL to analyze.")
        else:
            prog = st.progress(0)
            steps = [
                (12, "Extracting 36 handcrafted features..."),
                (24, "Running TF-IDF Logistic Regression..."),
                (36, "Running Char N-Gram + SVC..."),
                (50, "Running Random Forest (300 trees)..."),
                (64, "Running Gradient Boosting (200 est.)..."),
                (76, "Running MLP Neural Network (128→64→32)..."),
                (86, "Running Naive Bayes classifier..."),
                (93, "Executing rule-based NLP engine..."),
                (98, "Computing weighted ensemble..."),
            ]
            for p, msg in steps:
                prog.progress(p)
                log(msg)
                log_ph.markdown(log_html(st.session_state.logs), unsafe_allow_html=True)
                time.sleep(0.15)

            result = analyze(user_input, models)
            prog.progress(100)
            st.session_state.result = result
            st.session_state.scans += 1

            v  = result["verdict"]
            rs = result["risk_score"]
            fp = result["fraud_probability"]
            col_ = "#ff174488" if rs >= 60 else ("#ffd60088" if rs >= 38 else "#76ff0388")
            log(f"SCAN COMPLETE ✓ — Risk: {rs}/100  |  {v}  |  Fraud prob: {fp}%", col_)
            log_ph.markdown(log_html(st.session_state.logs), unsafe_allow_html=True)
            prog.empty()


    # ══════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════
    if st.session_state.result:
        r  = st.session_state.result
        rs = r["risk_score"]
        fp = r["fraud_probability"]
        v  = r["verdict"]
        ifd= r["is_fraud"]
        rcolor = rc(rs)

        vmap = {"SAFE":"v-safe","LOW RISK":"v-low","MODERATE RISK":"v-moderate",
                "HIGH RISK":"v-high","CRITICAL DANGER":"v-critical"}
        vcls = vmap.get(v, "v-safe")

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;
        letter-spacing:4px;color:rgba(0,229,255,.4);margin-bottom:14px;">
        ◈ THREAT ANALYSIS REPORT</div>""", unsafe_allow_html=True)

        # Banner
        chips = "".join(
            f'<span class="chip">⚑ {html.escape(k)}</span>'
            for k in r["flagged_keywords"]
        )
        explanation = html.escape(r["explanation"])
        st.markdown(f"""
        <div style="background:rgba(6,10,16,.97);border:1px solid {rcolor}33;border-radius:4px;
        padding:22px;margin-bottom:14px;box-shadow:0 0 36px {rcolor}0f;">
          <div class="verdict {vcls}">◈ {v}</div>
          <p style="font-size:13px;color:rgba(255,255,255,.6);line-height:1.75;max-width:720px;margin:0 0 12px 0;">
            {explanation}</p>
          <div>{chips}</div>
        </div>""", unsafe_allow_html=True)

        # Metric row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col_, val, lbl, sub, color in [
            (mc1, str(rs),                     "RISK SCORE",     "/100",        rcolor),
            (mc2, f"{fp}%",                    "FRAUD PROB.",    "ENSEMBLE",    "#ff6d00" if fp>=50 else "#76ff03"),
            (mc3, "FRAUD" if ifd else "LEGIT", "ML VERDICT",     "6 MODELS",    "#ff1744" if ifd else "#76ff03"),
            (mc4, str(r["url_risk"])+"%",      "URL RISK",       "MAX SCORE",   rc(r["url_risk"])),
            (mc5, str(len(r["url_details"])),  "URLs FOUND",     "IN INPUT",    "#ff6d00" if r["url_details"] else "#00e5ff"),
        ]:
            col_.markdown(f"""<div class="mc">
              <div class="mv" style="color:{color};font-size:36px;">{val}</div>
              <div class="ml">{lbl}</div><div class="ml">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────────
        t1,t2,t3,t4,t5,t6 = st.tabs([
            "📊  OVERVIEW", "🤖  ML MODELS", "🧠  ENSEMBLE",
            "🔗  URL ANALYSIS", "📈  DEEP LEARNING", "✅  RECOMMENDATIONS"
        ])

        # TAB 1 — Overview
        with t1:
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="panel"><div class="ptitle">THREAT CATEGORIES</div>', unsafe_allow_html=True)
                grid = '<div class="tg">'
                for cat in r["threat_categories"]:
                    det = cat["detected"]; sc2 = cat["score"]; c2 = rc(sc2)
                    ico = "⚠" if det else "○"; cs = "ti1" if det else "ti0"
                    grid += f'<div class="ti {cs}"><span>{ico} {cat["name"]}</span><span style="color:{c2};font-weight:bold;">{sc2}%</span></div>'
                grid += "</div>"
                st.markdown(grid + "</div>", unsafe_allow_html=True)

            with cb:
                st.markdown('<div class="panel"><div class="ptitle">SIGNAL STRENGTH</div>', unsafe_allow_html=True)
                dl = r["dl_insights"]
                bars = ""
                for lbl, val in [
                    ("Overall Risk Score",  rs),
                    ("Urgency Level",       dl["urgency_level"]),
                    ("Manipulation Score",  dl["manipulation"]),
                    ("Financial Exposure",  dl["financial_exposure"]),
                    ("Social Engineering",  dl["social_engineering"]),
                    ("Linguistic Anomaly",  dl["linguistic_anomaly"]),
                ]:
                    bars += pbar(lbl, val, rc(val))
                sc_col = "#ff6d00" if dl["sentiment"] < -0.3 else ("#76ff03" if dl["sentiment"] > 0.2 else "#ffd600")
                bars += f"""<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.05);
                display:flex;justify-content:space-between;">
                  <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                  color:rgba(255,255,255,.26);">NLP SENTIMENT SCORE</span>
                  <span style="font-family:'Share Tech Mono',monospace;font-size:12px;
                  font-weight:bold;color:{sc_col};">{dl['sentiment']:.3f}</span>
                </div>"""
                st.markdown(bars + "</div>", unsafe_allow_html=True)

        # TAB 2 — ML Models
        with t2:
            ca, cb = st.columns(2)
            for i, sig in enumerate(r["ml_signals"]):
                with (ca if i % 2 == 0 else cb):
                    trig = sig["triggered"]; pct2 = sig["pct"]; c3 = rc(pct2)
                    cls  = "sc1" if trig else "sc0"
                    ico  = "▶" if trig else "▷"
                    st.markdown(f"""<div class="sc {cls}">
                      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-family:'Share Tech Mono',monospace;font-size:11px;
                        color:{c3 if trig else 'rgba(255,255,255,.33)'};">{ico} {sig['name']}</span>
                        <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                        color:{c3 if trig else 'rgba(255,255,255,.2)'};">{pct2}%</span>
                      </div>
                      <div style="font-size:10px;color:rgba(255,255,255,.33);line-height:1.55;margin-bottom:8px;">
                        {sig['detail']}</div>
                      {pbar('TRIGGERED' if trig else 'CLEAN', pct2, c3)}
                    </div>""", unsafe_allow_html=True)

        # TAB 3 — Ensemble
        with t3:
            st.markdown('<div class="panel"><div class="ptitle">WEIGHTED ENSEMBLE — ALL MODEL PROBABILITIES</div>', unsafe_allow_html=True)
            rp = r["raw_probs"]
            cols = st.columns(4)
            model_display = [
                ("m1_lr",   "TF-IDF\nLOG.REG",    "Model 1"),
                ("m2_svc",  "CHAR\nN-GRAM SVC",   "Model 2"),
                ("m3_rf",   "RANDOM\nFOREST",      "Model 3"),
                ("m4_gb",   "GRAD\nBOOST",         "Model 4"),
                ("m5_mlp",  "MLP\nNEURAL NET",     "Model 5"),
                ("m6_nb",   "NAIVE\nBAYES",        "Model 6"),
                ("rules",   "RULE\nNLP",           "Heuristic"),
                ("ensemble","ENSEMBLE\nFINAL",     "⭐ Result"),
            ]
            for i, (key, label, sub) in enumerate(model_display):
                pct3 = int(rp.get(key, 0) * 100)
                c4 = rc(pct3)
                is_ens = key == "ensemble"
                cols[i % 4].markdown(f"""
                <div style="padding:14px;text-align:center;border-radius:2px;margin-bottom:10px;
                background:rgba(255,255,255,.02);
                border:{'2px' if is_ens else '1px'} solid {c4}{'66' if is_ens else '22'};">
                  <div style="font-family:'Rajdhani',sans-serif;font-size:{36 if is_ens else 28}px;
                  font-weight:700;color:{c4};">{pct3}%</div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:1px;
                  color:rgba(255,255,255,.25);white-space:pre-line;margin-top:4px;">{label}</div>
                </div>""", unsafe_allow_html=True)

            # Weights table
            st.markdown('<div class="ptitle" style="margin-top:16px;">ENSEMBLE WEIGHTS</div>', unsafe_allow_html=True)
            weights = models["meta"]["ensemble_weights"]
            wc2, wx, wy = st.columns(3)
            for i, (k, w) in enumerate(weights.items()):
                target_col = [wc2, wx, wy][i % 3]
                pct4 = int(w * 100)
                target_col.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:7px 10px;
                background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);
                border-radius:2px;margin-bottom:6px;">
                  <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                  color:rgba(255,255,255,.4);">{k.upper()}</span>
                  <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                  color:#00e5ff;">{pct4}%</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 4 — URL Analysis
        with t4:
            if r["url_details"]:
                for url_info in r["url_details"]:
                    safe_url = html.escape(url_info["url"], quote=True)
                    flag_html = "".join(
                        f'<div style="padding:5px 10px;margin:3px 0;background:rgba(255,23,68,.07);'
                        f'border-left:2px solid #ff174466;font-family:\'Share Tech Mono\',monospace;'
                        f'font-size:10px;color:#ff7597;">⚑ {f}</div>'
                        for f in (html.escape(flag) for flag in url_info["flags"])
                    )
                    score_c = rc(url_info["score"])
                    st.markdown(f"""
                    <div class="ub">
                      <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                      letter-spacing:2px;color:rgba(255,255,255,.2);margin-bottom:8px;">URL DETECTED</div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                      color:#ff7597;word-break:break-all;margin-bottom:12px;">🔗 {safe_url}</div>
                      <div style="display:flex;gap:14px;align-items:center;margin-bottom:12px;">
                        <div style="font-family:'Rajdhani',sans-serif;font-size:42px;font-weight:700;
                        color:{score_c};">{url_info['score']}<span style="font-size:16px;">/100</span></div>
                        <div>
                          <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                          letter-spacing:2px;color:rgba(255,255,255,.25);">RISK SCORE</div>
                          <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                          color:{score_c};margin-top:4px;">{'⚠ SUSPICIOUS' if not url_info['safe'] else '✓ APPEARS SAFE'}</div>
                        </div>
                      </div>
                      {flag_html}
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding:32px;text-align:center;font-family:'Share Tech Mono',monospace;
                font-size:12px;color:rgba(0,229,255,.35);border:1px solid rgba(255,255,255,.05);
                border-radius:2px;">○ NO URLs DETECTED IN INPUT</div>""", unsafe_allow_html=True)

        # TAB 5 — Deep Learning
        with t5:
            st.markdown('<div class="panel"><div class="ptitle">NEURAL NETWORK + FEATURE ANALYSIS</div>', unsafe_allow_html=True)
            dl2 = r["dl_insights"]
            da, db = st.columns(2)
            dl_items = [
                ("Urgency Level",      dl2["urgency_level"],     "Pressure & time-manipulation tactics detected"),
                ("Manipulation Score", dl2["manipulation"],       "Psychological coercion and fear-based framing"),
                ("Linguistic Anomaly", dl2["linguistic_anomaly"], "Grammar irregularities, ALL CAPS, excessive punctuation"),
                ("Financial Exposure", dl2["financial_exposure"], "Financial keywords, account/SSN/payment solicitation"),
                ("Social Engineering", dl2["social_engineering"], "Relationship exploitation and false trust signals"),
                ("MLP Confidence",     dl2["mlp_confidence"],    "Neural network (128→64→32) fraud probability output"),
            ]
            for i, (lbl, val, desc) in enumerate(dl_items):
                c5 = rc(val)
                with (da if i % 2 == 0 else db):
                    st.markdown(f"""
                    <div style="padding:15px;background:rgba(255,255,255,.02);
                    border:1px solid {c5}1a;border-radius:2px;margin-bottom:12px;">
                      <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                      letter-spacing:1px;color:{c5};margin-bottom:4px;">{lbl}</div>
                      <div style="font-family:'Rajdhani',sans-serif;font-size:36px;
                      font-weight:700;color:{c5};line-height:1;">{val}<span style="font-size:15px;">/100</span></div>
                      <div style="font-size:10px;color:rgba(255,255,255,.3);margin-top:6px;line-height:1.5;">{desc}</div>
                      {pbar('', val, c5)}
                    </div>""", unsafe_allow_html=True)

            # Sentiment + certainty
            sent = dl2["sentiment"]
            cert = dl2["ensemble_certainty"]
            sc2  = "#ff6d00" if sent < -0.3 else ("#76ff03" if sent > 0.2 else "#ffd600")
            sl   = "NEGATIVE / THREATENING" if sent < -0.3 else ("POSITIVE / NEUTRAL" if sent > 0.2 else "AMBIGUOUS")
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;">
              <div style="padding:16px;background:rgba(255,255,255,.02);
              border:1px solid rgba(255,255,255,.07);border-radius:2px;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:3px;color:rgba(255,255,255,.25);margin-bottom:8px;">NLP SENTIMENT SCORE</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:46px;
                font-weight:700;color:{sc2};">{sent:.3f}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{sc2};margin-top:4px;">{sl}</div>
              </div>
              <div style="padding:16px;background:rgba(255,255,255,.02);
              border:1px solid rgba(255,255,255,.07);border-radius:2px;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
                letter-spacing:3px;color:rgba(255,255,255,.25);margin-bottom:8px;">ENSEMBLE CERTAINTY</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:46px;
                font-weight:700;color:{rc(cert)};">{cert}%</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                color:rgba(255,255,255,.3);margin-top:4px;">Distance from decision boundary</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 6 — Recommendations
        with t6:
            for i, rec in enumerate(r["recommendations"], 1):
                st.markdown(f"""<div class="ri">
                  <div class="rn">{i}</div>
                  <div class="rt">{rec}</div>
                </div>""", unsafe_allow_html=True)

            if ifd or rs >= 60:
                st.markdown(f"""<div style="margin-top:14px;padding:18px;
                background:rgba(255,23,68,.06);border:1px solid rgba(255,23,68,.3);
                border-radius:2px;animation:cpulse 2s infinite alternate;">
                  <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                  letter-spacing:3px;color:#ff1744;margin-bottom:7px;">⚠  SECURITY ALERT</div>
                  <div style="font-size:13px;color:rgba(255,255,255,.6);line-height:1.7;">
                  This content was flagged as <strong style="color:#ff7597;">HIGH RISK</strong> by
                  the ensemble of 6 trained ML/DL models. Do not interact with this message,
                  click any links, or share personal information.
                  </div>
                </div>""", unsafe_allow_html=True)


    # Footer
    st.markdown("""
    <div class="footer">
      THREAT·SCAN·AI — 6-MODEL HYBRID ENSEMBLE — TF-IDF · CHAR N-GRAM · RANDOM FOREST · GRADIENT BOOST · MLP NEURAL NET · NAIVE BAYES<br>
      100% LOCAL — NO API — NO DATA LEAVES YOUR MACHINE
    </div>""", unsafe_allow_html=True)


def main():
    parser = argparse.ArgumentParser(description="ThreatScan AI fraud and phishing scanner")
    parser.add_argument("--train", action="store_true", help="train and save the six models")
    args, _ = parser.parse_known_args()
    if args.train:
        train_models()
    else:
        run_app()


if __name__ == "__main__":
    main()
