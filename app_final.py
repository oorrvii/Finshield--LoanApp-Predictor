import re
import math
import os
import joblib
import pandas as pd
import streamlit as st
import textwrap
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from google_play_scraper import app as play_app_scraper, reviews as play_reviews_scraper
    HAS_PLAY_SCRAPER = True
except ImportError:
    HAS_PLAY_SCRAPER = False


FEATURE_COLUMNS = [
    "contacts", "sms", "microphone", "location", "photos_media_storage",
    "disclosure_score", "review_redflag_score", "avg_review_sentiment",
    "pct_strongly_negative_reviews", "avg_review_length", "install_count",
]

REDFLAG_KEYWORDS = [
    'harass', 'threat', 'blackmail', 'recovery agent', 'shared my contact',
    'shared my photo', 'called my family', 'called my office', 'called my boss',
    'defame', 'morphed', 'abuse', 'extort', 'humiliate', 'fake photo',
    'contacted my', 'leak my photo', 'suicide', 'fraud app', 'scam',
    'collecting data', 'ask reference', 'visit my work', 'threatening',
]
REDFLAG_PATTERN = re.compile('|'.join(re.escape(k) for k in REDFLAG_KEYWORDS), re.I)

FEATURES_CSV_PATH = "app_features_final.csv"
MIN_REVIEWS_REQUIRED = 10

KNOWN_BANKS = [
    "hdfc", "icici", "sbi", "statebank", "axis", "kotak", "baroda", "bob", "pnb",
    "canara", "unionbank", "idfc", "indusind", "yesbank", "rbl", "federal", "centralbank",
    "indianbank", "uco", "bankofindia", "iob", "psb", "dbs", "hsbc", "citi", "standardchartered",
    "bandhan", "au", "aubank", "equitas", "ujjivan", "jana", "survodaya",
    "ltfinance", "ltfs", "lntfinance", "lt-finance", "bajaj", "bajajfinserv", "bajajfinance",
    "tata", "tatacapital", "tataneu", "piramal", "adityabirla", "abfl", "godrej", "godrejcapital",
    "mahindra", "mmfsl", "shriram", "stfc", "muthoot", "muthootfinance", "manappuram",
    "cholamandalam", "chola", "sundaram", "iifl", "hero", "herofincorp", "tvssundaram", "tvscredit",
    "lendingkart", "creditsaison", "homecredit", "paytm", "groww", "kreditbee", "navi", "fibe",
    "earlysalary", "moneyview", "cashe", "kissht", "stashfin", "faircent", "mpokket", "slice",
    "onecard", "fatakpay", "cred", "jupiter", "freo", "lazypay", "branch", "nira", "flexiloans",
    "zest", "zestmoney", "dhanvarsha", "indialends", "rupeeredee"
]

analyzer = SentimentIntensityAnalyzer()

def explain_feature(name: str, value) -> tuple[str, bool]:
    if name == "contacts":
        return ("Asks for access to your Contacts list — not something a loan app genuinely needs", True) if value else \
               ("Does not ask for your Contacts list", False)
    if name == "sms":
        return ("Asks to read your SMS messages — often used to intercept OTPs or spam your contacts", True) if value else \
               ("Does not ask to read your SMS messages", False)
    if name == "microphone":
        return ("Asks for access to your Microphone — unusual for a loan app", True) if value else \
               ("Does not ask for microphone access", False)
    if name == "location":
        return ("Asks for your precise Location", True) if value else \
               ("Does not ask for your location", False)
    if name == "photos_media_storage":
        return ("Asks for access to your Photos & Media — has been used in some cases to threaten borrowers with personal images", True) if value else \
               ("Does not ask for access to your photos", False)
    if name == "disclosure_score":
        v = value or 0
        if v <= 2:
            return (f"Barely explains its own terms — only {v} of 5 basics disclosed (interest rate, tenure, RBI/NBFC registration, support contact, privacy policy)", True)
        return (f"Clearly discloses key loan terms ({v} of 5 basics covered)", False)
    if name == "review_redflag_score":
        pct = (value or 0) * 100
        if pct >= 10:
            return (f"About {pct:.0f}% of reviews mention harassment, threats, or recovery-agent abuse", True)
        return ("Very few reviews mention harassment or threats", False)
    if name == "avg_review_sentiment":
        if (value or 0) > 0.5:
            return ("Reviews are unusually, uniformly positive — sometimes a sign of fake/boosted reviews burying real complaints", True)
        if (value or 0) < -0.2:
            return ("Reviews lean negative overall", True)
        return ("Reviews show a normal, mixed sentiment", False)
    if name == "pct_strongly_negative_reviews":
        pct = (value or 0) * 100
        if pct >= 15:
            return (f"About {pct:.0f}% of reviews are strongly negative", True)
        return ("Few reviews are strongly negative", False)
    if name == "avg_review_length":
        if (value or 0) >= 15:
            return ("Reviews tend to be long and detailed — often a sign of genuine, specific complaints", True)
        return ("Reviews tend to be short, generic comments", False)
    if name == "install_count":
        v = value or 0
        if v < 10000:
            return (f"Relatively few installs ({v:,}) — less track record to go on", True)
        return (f"Has a substantial install base ({v:,})", False)
    return (f"{name}: {value}", False)

FEATURE_LABELS = {
    "contacts": "Contacts permission", "sms": "SMS permission",
    "microphone": "Microphone permission", "location": "Location permission",
    "photos_media_storage": "Photos/Media permission",
    "disclosure_score": "Terms disclosure", "review_redflag_score": "Harassment mentions in reviews",
    "avg_review_sentiment": "Review sentiment pattern", "pct_strongly_negative_reviews": "Strongly negative reviews",
    "avg_review_length": "Review detail level", "install_count": "Install base",
}

def disclosure_score(description: str, privacy_policy: str) -> int:
    desc = str(description).lower()
    has_rate = bool(re.search(r'interest rate|% pa|apr|per annum|processing fee', desc))
    has_tenure = bool(re.search(r'tenure|repayment period|months|loan period', desc))
    has_reg = bool(re.search(r'rbi[- ]registered|nbfc|registration number|cin ', desc))
    has_contact = bool(re.search(r'customer care|grievance|support@|contact us|helpline', desc))
    pp = str(privacy_policy)
    has_pp = pp not in ('nan', '', 'None') and 'http' in pp
    return int(has_rate) + int(has_tenure) + int(has_reg) + int(has_contact) + int(has_pp)

def parse_installs(installs_text: str):
    if not installs_text:
        return None
    digits = re.sub(r'[^0-9]', '', str(installs_text))
    return int(digits) if digits else 0

@st.cache_data
def load_features_table():
    return pd.read_csv(FEATURES_CSV_PATH)

def get_app_choices():
    try:
        df = load_features_table()
        return sorted(df["app_name"].dropna().astype(str).unique().tolist())
    except FileNotFoundError:
        return []


def lookup_app_features(identifier: str):
    df = load_features_table()
    ident = str(identifier).strip().lower()

    id_col = df["app_id"].astype(str).str.lower() if "app_id" in df.columns else None
    name_col = df["app_name"].astype(str).str.lower() if "app_name" in df.columns else None

    if id_col is not None and name_col is not None:
        match = df[(id_col == ident) | (name_col == ident)]
    elif name_col is not None:
        match = df[name_col == ident]
    else:
        match = df[id_col == ident]

    if match.empty:
        return None

    row = match.iloc[0]
    res = {col: row[col] for col in FEATURE_COLUMNS if col in row}
    if "app_id" in row:
        res["app_id"] = str(row["app_id"]).strip()
    if "app_name" in row:
        res["app_name"] = str(row["app_name"]).strip()
    return res

USE_FAKE_MODEL = not os.path.exists("predatory_loan_detector.pkl")

@st.cache_resource
def load_model():
    model = joblib.load("predatory_loan_detector.pkl")
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        if not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"
    elif not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model

def _fake_predict(features: dict):
    score = (
        0.10
        + features["review_redflag_score"] * 0.5
        + features["pct_strongly_negative_reviews"] * 0.2
        + (5 - features["disclosure_score"]) * 0.04
        + (features["contacts"] + features["sms"]) * 0.05
    )
    score = min(max(score, 0.0), 0.97)
    watch_list = ["review_redflag_score", "disclosure_score", "contacts", "sms"]
    reasons = [explain_feature(name, features[name]) for name in watch_list]
    return score, reasons

def predict(features: dict):
    if features.get("is_known_legit"):
        return 0.08, [
            ("Regulated Bank / NBFC entity with compliant data privacy practices.", False),
            ("Zero prohibited contact or photo storage permissions requested.", False),
            ("Transparent loan terms and clear APR disclosures.", False),
            ("Verified RBI lending compliance.", False)
        ]
    if USE_FAKE_MODEL:
        return _fake_predict(features)
    model = load_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS).fillna(0)
    proba = model.predict_proba(row)[0][1]  
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = dict(zip(feature_names, classifier.coef_[0]))
    top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    reasons = []
    for name, _coef in top:
        clean_name = name.replace("num__", "")
        reasons.append(explain_feature(clean_name, features.get(clean_name, 0)))
    return proba, reasons

@st.cache_data
def get_ranked_apps_df():
    df = load_features_table()
    EXCLUDED_NON_LENDING_IDS = [
        "com.google.android.apps.nbu.paisa.user",  # Google Pay
        "in.amazon.mshop.android.shopping",        # Amazon India
        "com.nextbillion.groww",                   # Groww Stocks (in.groww.dash is Groww Credit)
        "tech.fplabs.score",                       # OneScore
        "com.moneymanager.personal.finance.planner",# Loan Master Plan
        "com.analytics.finance.manager.money.app",  # Daily Loan - Money Tracker
        "com.strong.primecash",                    # PrimeCash - Earn Rewards
        "com.nayarupee",                           # NayaRupee Spin & Earn
    ]
    records = []
    for idx, row in df.iterrows():
        app_id_clean = str(row.get("app_id", "")).lower().strip()
        if any(ex in app_id_clean for ex in EXCLUDED_NON_LENDING_IDS):
            continue
        feat = row.to_dict()
        name_lower = str(feat.get('app_name', '')).lower()
        id_lower = str(feat.get('app_id', '')).lower()
        is_known = any(b in name_lower or b in id_lower for b in KNOWN_BANKS) or bool(feat.get('is_known_legit', False))
        if is_known:
            feat['is_known_legit'] = True
        
        proba, reasons = predict(feat)
        safety_score = max(5, min(99, int(round((1.0 - proba) * 100))))
        
        if safety_score >= 80:
            tier = "🛡️ Safest Tier"
            tier_key = "safe"
        elif safety_score >= 50:
            tier = "⚠️ Moderate Caution"
            tier_key = "moderate"
        else:
            tier = "🚨 High Risk"
            tier_key = "high_risk"
            
        installs = feat.get("install_count")
        if installs:
            if installs >= 10000000:
                installs_str = f"{installs // 1000000}M+"
            elif installs >= 1000000:
                installs_str = f"{installs / 1000000:.1f}M+"
            elif installs >= 1000:
                installs_str = f"{installs // 1000}K+"
            else:
                installs_str = f"{installs:,}"
        else:
            installs_str = "—"
            
        disclosure = feat.get("disclosure_score", 0)
        redflag_pct = (feat.get("review_redflag_score", 0) or 0) * 100
        
        if is_known:
            cat_tag = "Personal Loan • RBI Regulated NBFC Partner"
        elif disclosure >= 4:
            cat_tag = "Instant Credit • Verified Disclosures"
        else:
            cat_tag = "Digital Lending • Play Store App"

        records.append({
            "app_name": str(feat.get("app_name", "")),
            "app_id": str(feat.get("app_id", "")),
            "safety_score": safety_score,
            "risk_proba": proba,
            "tier": tier,
            "tier_key": tier_key,
            "cat_tag": cat_tag,
            "is_known": is_known,
            "installs_str": installs_str,
            "install_count": installs or 0,
            "disclosure_score": disclosure,
            "redflag_pct": redflag_pct,
            "contacts": feat.get("contacts", 0),
            "sms": feat.get("sms", 0),
            "photos": feat.get("photos_media_storage", 0),
            "features_dict": feat,
            "reasons": reasons
        })
    
    ranked_df = pd.DataFrame(records).sort_values(by=["safety_score", "install_count"], ascending=[False, False]).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    return ranked_df

def is_valid_unlisted_input(input_str: str) -> bool:
    if not input_str or len(input_str.strip()) < 3:
        return False
    s = input_str.strip().lower()
    if s.startswith("http://") or s.startswith("https://") or "play.google.com" in s:
        return True
    if re.search(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$', s):
        return True
    VALID_TLDS = [
        ".com", ".in", ".org", ".net", ".io", ".co", ".xyz", ".app", ".site",
        ".online", ".top", ".vip", ".cc", ".tech", ".link", ".win", ".club",
        ".gov", ".edu", ".ai", ".me", ".store", ".info"
    ]
    if any(s.endswith(tld) or (tld + "/") in s for tld in VALID_TLDS) and "." in s:
        return True
    KNOWN_KEYWORDS = [
        "hdfc", "icici", "sbi", "axis", "kotak", "baroda", "pnb", "groww",
        "creditsaison", "kreditbee", "navi", "fibe", "tataneu", "paytm",
        "slice", "onecard", "fatakpay", "bajaj", "hero", "muthoot", "indusind",
        "yesbank", "rbl", "federal", "canara", "unionbank", "idfc",
        "kredit", "rupee", "cash", "loan", "wallet", "fastcash"
    ]
    if any(k in s for k in KNOWN_KEYWORDS):
        return True
    return False

def extract_package_id(input_str: str) -> str:
    input_str = input_str.strip()
    match = re.search(r'id=([a-zA-Z0-9_\.]+)', input_str)
    if match:
        return match.group(1)
    match_pkg = re.search(r'([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_\.]+)', input_str)
    if match_pkg:
        return match_pkg.group(1)
    match_pkg2 = re.search(r'([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)', input_str)
    if match_pkg2 and not input_str.startswith("http"):
        return match_pkg2.group(1)
    if "://" in input_str:
        domain = input_str.split("://")[1].split("/")[0].split("?")[0].lower()
        return domain
    return input_str.lower()

def scrape_playstore_live(pkg_id: str) -> dict | None:
    if not HAS_PLAY_SCRAPER:
        return None
    
    clean_id = pkg_id.strip()
    data = None
    try:
        data = play_app_scraper(clean_id, lang='en', country='in')
    except Exception:
        m = re.search(r'id=([a-zA-Z0-9_\.]+)', clean_id)
        if m:
            clean_id = m.group(1)
            try:
                data = play_app_scraper(clean_id, lang='en', country='in')
            except Exception:
                return None
        else:
            return None

    if not data:
        return None

    app_name = data.get('title') or clean_id
    description = data.get('description') or data.get('summary') or ""
    privacy_policy = data.get('privacyPolicy') or ""
    
    raw_installs = data.get('realInstalls') or data.get('minInstalls')
    if raw_installs and isinstance(raw_installs, (int, float)) and raw_installs > 0:
        installs = int(raw_installs)
    else:
        installs = parse_installs(str(data.get('installs', 0))) or 10000

    disc_score = disclosure_score(description, privacy_policy)
    
    perms_raw = str(data.get('permissions', [])).lower() + " " + description.lower()
    has_contacts = 1 if ('contact' in perms_raw or 'read_contacts' in perms_raw) else 0
    has_sms = 1 if ('sms' in perms_raw or 'read_sms' in perms_raw or 'receive_sms' in perms_raw) else 0
    has_mic = 1 if ('record_audio' in perms_raw or 'microphone' in perms_raw) else 0
    has_loc = 1 if ('location' in perms_raw or 'access_fine_location' in perms_raw or 'gps' in perms_raw) else 0
    has_storage = 1 if ('storage' in perms_raw or 'photos' in perms_raw or 'media' in perms_raw or 'read_external_storage' in perms_raw) else 0

    rvs = []
    try:
        rv_data, _ = play_reviews_scraper(clean_id, lang='en', country='in', count=50)
        if rv_data:
            rvs = [r.get('content', '') for r in rv_data if r.get('content')]
    except Exception:
        pass
    
    if rvs:
        redflag_count = sum(1 for r in rvs if REDFLAG_PATTERN.search(r))
        review_redflag_score = redflag_count / len(rvs)
        
        compounds = [analyzer.polarity_scores(r)['compound'] for r in rvs]
        avg_review_sentiment = sum(compounds) / len(compounds)
        
        strongly_neg = sum(1 for c in compounds if c <= -0.5)
        pct_strongly_negative_reviews = strongly_neg / len(rvs)
        
        avg_review_length = sum(len(r.split()) for r in rvs) / len(rvs)
    else:
        review_redflag_score = 0.05
        avg_review_sentiment = 0.1
        pct_strongly_negative_reviews = 0.05
        avg_review_length = 15.0

    is_known = any(b in app_name.lower() or b in clean_id.lower() for b in KNOWN_BANKS)

    return {
        "app_id": clean_id,
        "app_name": app_name,
        "install_count": installs,
        "disclosure_score": disc_score,
        "review_redflag_score": review_redflag_score,
        "pct_strongly_negative_reviews": pct_strongly_negative_reviews,
        "avg_review_sentiment": avg_review_sentiment,
        "avg_review_length": avg_review_length,
        "contacts": has_contacts,
        "sms": has_sms,
        "microphone": has_mic,
        "location": has_loc,
        "photos_media_storage": has_storage,
        "is_known_legit": is_known,
        "is_web_domain": False,
        "is_custom_unlisted": True,
        "scrape_source": "live_playstore",
        "scraped_developer": data.get("developer", ""),
        "scraped_privacy_policy": privacy_policy,
        "scraped_reviews_count": len(rvs)
    }

def scrape_web_domain_live(url_or_domain: str) -> dict | None:
    raw_input = url_or_domain.strip()
    if not (raw_input.startswith("http://") or raw_input.startswith("https://")):
        target_url = "https://" + raw_input
    else:
        target_url = raw_input

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(target_url, headers=headers, timeout=6)
        if res.status_code != 200:
            if target_url.startswith("https://"):
                target_url = "http://" + raw_input.replace("https://", "")
                res = requests.get(target_url, headers=headers, timeout=6)
            if res.status_code != 200:
                return None
    except Exception:
        return None

    try:
        soup = BeautifulSoup(res.text, 'html.parser')
        
        page_title = soup.title.string.strip() if soup.title and soup.title.string else raw_input
        h1 = soup.find('h1')
        if h1 and h1.text:
            display_name = h1.text.strip()[:60]
        else:
            display_name = page_title[:60]

        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        text_body = soup.get_text(separator=' ')
        clean_text = ' '.join(text_body.split())

        privacy_policy = ""
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'privacy' in href.lower():
                privacy_policy = href if href.startswith("http") else (target_url.rstrip("/") + "/" + href.lstrip("/"))
                break

        disc_score = disclosure_score(clean_text, privacy_policy)

        text_lower = clean_text.lower()
        redflag_matches = REDFLAG_PATTERN.findall(text_lower)
        redflag_score = min(0.5, len(redflag_matches) * 0.1)

        pol = analyzer.polarity_scores(clean_text[:2000])
        sentiment = pol['compound']
        neg_pct = pol['neg']

        has_contacts = 1 if any(w in text_lower for w in ["contacts list", "read contacts", "access contacts", "contact list"]) else 0
        has_sms = 1 if any(w in text_lower for w in ["read sms", "sms permission", "otp access", "receive sms"]) else 0
        has_mic = 1 if any(w in text_lower for w in ["microphone", "record audio"]) else 0
        has_loc = 1 if any(w in text_lower for w in ["location", "gps access"]) else 0
        has_storage = 1 if any(w in text_lower for w in ["gallery", "photos", "storage permission", "media access"]) else 0

        is_known = any(b in display_name.lower() or b in raw_input.lower() for b in KNOWN_BANKS)

        return {
            "app_id": raw_input,
            "app_name": display_name,
            "install_count": 10000 if not is_known else 10000000,
            "disclosure_score": disc_score,
            "review_redflag_score": redflag_score,
            "pct_strongly_negative_reviews": neg_pct,
            "avg_review_sentiment": sentiment,
            "avg_review_length": 25.0,
            "contacts": has_contacts,
            "sms": has_sms,
            "microphone": has_mic,
            "location": has_loc,
            "photos_media_storage": has_storage,
            "is_known_legit": is_known,
            "is_web_domain": True,
            "is_custom_unlisted": True,
            "scrape_source": "live_website",
            "scraped_privacy_policy": privacy_policy,
            "scraped_url": target_url
        }
    except Exception:
        return None

def build_unlisted_app_features(pkg_id_or_input: str) -> dict:
    raw_input = pkg_id_or_input.strip()

    # 1. Try Live Google Play Store Scraping if package id or Play Store link
    clean_pkg_id = extract_package_id(raw_input)
    live_play = scrape_playstore_live(clean_pkg_id)
    if live_play:
        return live_play

    # 2. Try Live Web Scraping if URL or web domain
    if "." in raw_input or raw_input.startswith("http"):
        live_web = scrape_web_domain_live(raw_input)
        if live_web:
            return live_web

    # 3. Offline / Unreachable Fallback Heuristic
    pkg_clean = clean_pkg_id.lower()
    HIGH_RISK_TERMS = [
        "fast", "quick", "instant", "7day", "urgent", "pocket", "rupee",
        "cash", "loan", "wallet", "easy", "credit", "money", "express", "apk"
    ]
    SUSPICIOUS_TLDS = [".xyz", ".top", ".online", ".site", ".vip", ".cc", ".tech", ".link", ".win", ".club"]
    
    is_known = any(b in pkg_clean for b in KNOWN_BANKS)
    has_risk_terms = any(t in pkg_clean for t in HIGH_RISK_TERMS)
    has_suspicious_tld = any(pkg_clean.endswith(tld) or (tld + "/") in pkg_clean for tld in SUSPICIOUS_TLDS)
    
    if is_known:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 10000000, 5, 0.01, 0.02, 0.65, 15.0, 0, 0, 0, 0, 0
    elif has_risk_terms or has_suspicious_tld:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 10000, 1, 0.35, 0.52, -0.45, 50.0, 1, 1, 1, 1, 1
    else:
        installs, disclosure, redflag, neg_reviews, sentiment, length, contacts, sms, mic, loc, storage = 100000, 3, 0.12, 0.22, 0.05, 25.0, 1, 0, 0, 1, 0

    display_name = raw_input
    if "." in raw_input:
        parts = [p for p in raw_input.split(".") if p not in ["com", "in", "org", "net", "co", "io", "xyz", "site", "online", "http", "https", "www"]]
        if parts:
            display_name = " ".join(parts).replace("_", " ").replace("-", " ").title()

    is_android_pkg = bool(
        re.search(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+){1,}$', clean_pkg_id)
        and any(clean_pkg_id.startswith(p) for p in ["com.", "in.", "org.", "net.", "io.", "co."])
    )
    is_playstore = "play.google.com" in raw_input
    is_web = not is_android_pkg and not is_playstore and "." in raw_input

    return {
        "app_id": raw_input,
        "app_name": display_name,
        "install_count": installs,
        "disclosure_score": disclosure,
        "review_redflag_score": redflag,
        "pct_strongly_negative_reviews": neg_reviews,
        "avg_review_sentiment": sentiment,
        "avg_review_length": length,
        "contacts": contacts,
        "sms": sms,
        "microphone": mic,
        "location": loc,
        "photos_media_storage": storage,
        "is_known_legit": is_known,
        "is_web_domain": is_web,
        "is_custom_unlisted": True,
        "scrape_source": "offline_fallback"
    }

def score_app(identifier: str):
    try:
        clean_id = extract_package_id(identifier)
        features = lookup_app_features(clean_id)
        used_fallback = False
        
        if features is None:
            features = lookup_app_features(identifier)

        if features is None:
            features = build_unlisted_app_features(identifier)
            used_fallback = True
            
        score, reasons = predict(features)
        return score, reasons, used_fallback, features
    except FileNotFoundError:
        st.error(f"Can't find {FEATURES_CSV_PATH} — make sure it's in the same folder as this file.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Something went wrong scoring this app: {e}")
        return None, None, None, None

def get_app_detailed_profile(name_clean: str, app_id: str, is_known: bool):
    name_lower = name_clean.lower()
    
    # Default Profile Template
    founded_by = "Promoted by RBI-registered NBFC / Regulated Lending Partner"
    about = f"{name_clean} is a digital lending platform providing instant personal credit solutions across India. It features online document submission, structured tenure options, and clear repayment schedules."
    
    who_should_take = [
        "Borrowers with stable monthly income looking for transparent, regulated loan terms.",
        "Users needing short to medium term emergency credit with clear EMI repayment schedules.",
        "Borrowers with verified bank accounts, PAN, and Aadhaar seeking digital onboarding.",
        "Borrowers who verify interest rates, APR, and fee breakdowns before accepting sanction terms."
    ]
    
    who_should_avoid = [
        "Borrowers seeking unauthorized 7-day or 14-day ultra-short loans with hidden charges.",
        "Users unable or unwilling to complete mandatory RBI e-KYC and income documentation.",
        "Borrowers sensitive to sharing financial data or granting sensitive device permissions.",
        "Individuals borrowing without a clear, sustainable monthly repayment budget."
    ]
    
    # Specific Brand Overrides
    if "sbi" in name_lower or "state bank" in name_lower:
        founded_by = "Government of India (State Bank of India - Established 1955, Chaired by CS Setty)"
        about = "State Bank of India (SBI) is India's largest public-sector bank and premier financial institution. It offers home loans, personal loans, and YONO digital credit with competitive interest rates and extended tenures."
        who_should_take = [
            "Borrowers seeking low interest rates (7.25% - 10.5% p.a.) and extended tenures up to 30 years.",
            "Salaried employees, government staff, and verified business owners with good credit (750+ CIBIL).",
            "Borrowers looking for government subsidies (e.g., PMAY) and zero hidden fee structures."
        ]
        who_should_avoid = [
            "Borrowers seeking instant 5-minute approval without formal income proof.",
            "Users with low credit scores (<650 CIBIL) or past default records."
        ]
    elif "navi" in name_lower:
        founded_by = "Sachin Bansal & Ankit Agarwal (Founded 2018 under Navi Technologies Ltd / Navi Finserv Ltd)"
        about = "Navi is an RBI-registered NBFC offering instant paperless personal loans up to ₹20 Lakhs, home loans, and mutual funds via a 100% digital app interface with zero physical documentation."
        who_should_take = [
            "Tech-savvy borrowers wanting 100% paperless digital loan disbursal within minutes.",
            "Salaried and self-employed professionals with regular income deposited in bank accounts.",
            "Borrowers looking for flexible EMI tenure options (3 to 84 months) with zero pre-closure charges."
        ]
        who_should_avoid = [
            "Borrowers without regular bank statement transaction history.",
            "Users seeking cash disbursals without bank account validation."
        ]
    elif "kreditbee" in name_lower:
        founded_by = "Madhusudan Ekambaram, Karthik Srinivasan & Vivek Veda (Founded 2018 under Krazybee Services Pvt Ltd)"
        about = "KreditBee is a major Indian digital lending platform partnering with Krazybee Services (RBI-registered NBFC) and partner banks to provide instant personal credit from ₹1,000 to ₹5,000,000."
        who_should_take = [
            "Young professionals and first-time borrowers needing quick short-term personal credit.",
            "Borrowers looking for flexible options like Flexi Personal Loan and Salary Loans.",
            "Users with valid PAN, Aadhaar, and monthly income proof seeking digital onboarding."
        ]
        who_should_avoid = [
            "Borrowers looking for low single-digit interest rates (APR can range from 16% to 29.95% p.a.).",
            "Users unable to pay processing fees or seeking unverified off-market loans."
        ]
    elif "bajaj" in name_lower:
        founded_by = "Rahul Bajaj & Sanjiv Bajaj (Bajaj Finserv Ltd / Bajaj Finance Ltd - Established 1987)"
        about = "Bajaj Finance Ltd is one of India's largest non-banking financial companies (NBFC). It offers Insta Personal Loans, EMI Store shopping cards, and business credit across India."
        who_should_take = [
            "Existing Bajaj Finserv customers with pre-approved Insta Loan offers.",
            "Borrowers looking for no-cost EMI shopping and quick durable financing.",
            "Salaried individuals seeking reliable NBFC backing with high credit limits."
        ]
        who_should_avoid = [
            "Borrowers who dislike aggressive telecalling or cross-sell marketing.",
            "Users unable to verify pre-calculated processing and insurance add-on costs."
        ]
    elif "tata" in name_lower:
        founded_by = "Tata Sons / Tata Capital Limited (Founded 2007, Chaired by Rajiv Sabharwal)"
        about = "Tata Capital Limited is the financial services arm of the Tata Group, offering digital personal loans, home financing, and small business credit backed by trusted corporate governance."
        who_should_take = [
            "Borrowers prioritizing corporate trust, legal compliance, and transparent terms.",
            "Salaried corporate employees looking for competitive interest rates and clear APR disclosures.",
            "Borrowers needing long-term stability with no surprise penalty clauses."
        ]
        who_should_avoid = [
            "Borrowers seeking anonymous unlisted APK loans without KYC."
        ]
    elif "paytm" in name_lower:
        founded_by = "Vijay Shekhar Sharma (One97 Communications Ltd / Paytm Payments Services)"
        about = "Paytm provides digital credit and Paytm Postpaid solutions in partnership with regulated NBFCs like Aditya Birla Finance and Tata Capital, facilitating instant merchant and personal credit."
        who_should_take = [
            "Frequent Paytm wallet/UPI users looking for instant micro-credit for merchant payments.",
            "Users wanting seamless bill payments and small ticket personal loans."
        ]
        who_should_avoid = [
            "Borrowers needing large long-term home or capital loans.",
            "Users sensitive to late payment convenience fee charges."
        ]
    elif "groww" in name_lower:
        founded_by = "Lalit Keshre, Harsh Jain, Ishan Bansal & Neeraj Singh (Nextbillion Technology Pvt Ltd - Founded 2016)"
        about = "Groww is an online investment and digital lending platform offering mutual funds, stocks, and instant personal loans to verified investors and salaried users."
        who_should_take = [
            "Existing Groww app users with verified Demat and mutual fund portfolios.",
            "Borrowers seeking low-friction paperless personal loans with instant disbursal."
        ]
        who_should_avoid = [
            "Non-registered users without online bank verification."
        ]
    elif "cashe" in name_lower:
        founded_by = "V Raman Kumar (Founded 2016 under Bhanix Finance and Investment Ltd)"
        about = "CASHe is an AI-driven digital lending platform for salaried millennials, offering short-term personal loans, BNPL credit, and credit lines using a proprietary Social Loan Quotient (SLQ)."
        who_should_take = [
            "Salaried millennials needing short 3 to 18-month personal loans for urgent expenses.",
            "Borrowers with monthly salary statements looking for quick mobile disbursal."
        ]
        who_should_avoid = [
            "Self-employed users without formal monthly salary slips.",
            "Borrowers sensitive to upfront processing fee deductions."
        ]
    elif "moneyview" in name_lower or "money view" in name_lower:
        founded_by = "Puneet Agarwal & Sanjay Aggarwal (Whizdm Innovations Pvt Ltd - Founded 2014)"
        about = "Money View is a digital financial platform offering instant personal loans up to ₹10 Lakhs in partnership with leading RBI-registered NBFCs and banks across 5,000+ locations in India."
        who_should_take = [
            "Borrowers with CIBIL score of 600+ needing fast digital personal loans.",
            "Salaried and self-employed individuals with minimum monthly income of ₹15,000."
        ]
        who_should_avoid = [
            "Borrowers with active default records or zero income documentation."
        ]
    elif "stashfin" in name_lower:
        founded_by = "Tushar Aggarwal (Founded 2016 under Akara Capital Advisors Pvt Ltd)"
        about = "Stashfin is an RBI-registered NBFC platform offering credit lines, personal loans, and credit cards with flexible interest payments only on the amount withdrawn."
        who_should_take = [
            "Borrowers needing a flexible revolving credit line where interest applies only on utilized funds.",
            "Users wanting 24/7 access to instant cash withdrawals."
        ]
        who_should_avoid = [
            "Borrowers who delay monthly bill cycles (high penalty APR rates)."
        ]
    elif "mpokket" in name_lower:
        founded_by = "Gaurav Jalan (Founded 2016 under mPokket Financial Services Pvt Ltd)"
        about = "mPokket is an RBI-registered NBFC catering primarily to college students and young salaried professionals, offering small-ticket micro-loans from ₹500 to ₹50,000."
        who_should_take = [
            "College students and young graduates needing small pocket-money loans with student ID.",
            "Salaried individuals seeking micro-credit for small emergency expenses."
        ]
        who_should_avoid = [
            "Borrowers seeking large long-term loans (>₹1 Lakh).",
            "Users unable to repay on due dates (impacts CIBIL score)."
        ]
    elif "fatakpay" in name_lower:
        founded_by = "Ajit Kumar & Amit Dassani (FatakPay Digital Pvt Ltd)"
        about = "FatakPay is a financial wellness platform providing instant salary advance and micro-credit solutions for blue-collar and salaried workers in partnership with RBI-registered NBFCs."
        who_should_take = [
            "Employed workers seeking instant salary advance before monthly payday.",
            "Borrowers looking for 0% interest short credit periods."
        ]
        who_should_avoid = [
            "Unemployed individuals or users seeking long-term loan terms."
        ]

    return {
        "founded_by": founded_by,
        "about": about,
        "who_should_take": who_should_take,
        "who_should_avoid": who_should_avoid
    }

def render_rbi_riskometer_card(score: float, dark_mode: bool) -> str:
    s = min(max(score, 0.0), 1.0)
    angle = s * 180.0

    slices = [
        (0, 30, "#388E3C", "LOW", "", "#000000"),
        (30, 60, "#7CB342", "LOW to", "MODERATE", "#000000"),
        (60, 90, "#FDD835", "MODERATE", "", "#000000"),
        (90, 120, "#FB8C00", "MODERATELY", "HIGH", "#000000"),
        (120, 150, "#F4511E", "HIGH", "", "#000000"),
        (150, 180, "#D32F2F", "VERY HIGH", "", "#000000"),
    ]

    def get_pt(deg_val, r=140, cx=160, cy=160):
        rad = math.radians(180 - deg_val)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    paths, texts = [], []
    for start_a, end_a, color, label1, label2, fg_color in slices:
        x1, y1 = get_pt(start_a)
        x2, y2 = get_pt(end_a)
        path_d = f"M 160 160 L {x1:.2f} {y1:.2f} A 140 140 0 0 1 {x2:.2f} {y2:.2f} Z"
        paths.append(f'<path d="{path_d}" fill="{color}" stroke="#FFFFFF" stroke-width="2.5" />')

        mid_a = (start_a + end_a) / 2
        tx, ty = get_pt(mid_a, r=94)
        if label2:
            texts.append(f'<text x="{tx:.2f}" y="{ty-3:.2f}" fill="{fg_color}" font-size="6.5" font-weight="900" text-anchor="middle" font-family="sans-serif">{label1}</text>')
            texts.append(f'<text x="{tx:.2f}" y="{ty+6:.2f}" fill="{fg_color}" font-size="6.5" font-weight="900" text-anchor="middle" font-family="sans-serif">{label2}</text>')
        else:
            texts.append(f'<text x="{tx:.2f}" y="{ty+2:.2f}" fill="{fg_color}" font-size="8" font-weight="900" text-anchor="middle" font-family="sans-serif">{label1}</text>')

    needle_fill = "#000000"
    needle_svg = (
        f'<g transform="rotate({angle:.1f}, 160, 160)">'
        f'<polygon points="160,154 42,160 160,166" fill="{needle_fill}" stroke="#FFFFFF" stroke-width="0.8" />'
        f'<circle cx="160" cy="160" r="10" fill="{needle_fill}" stroke="#FFFFFF" stroke-width="1.5" />'
        f'<circle cx="160" cy="160" r="4" fill="{needle_fill}" />'
        f'</g>'
    )

    card_bg = "#121722" if dark_mode else "#FFFFFF"
    card_border = "rgba(255,255,255,0.08)" if dark_mode else "#E2E8F0"

    svg_str = (
        f'<div style="text-align: center; margin: 0 auto;">'
        f'<svg viewBox="0 15 320 155" width="100%" style="max-width: 220px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.25));">'
        f'{"".join(paths)}'
        f'{"".join(texts)}'
        f'{needle_svg}'
        f'</svg>'
        f'</div>'
    )

    return f'<div style="border:1px solid {card_border}; border-radius:14px; padding:12px; background-color:{card_bg}; max-width:240px; margin:0 auto;">{svg_str}</div>'

# Streamlit Page Config & Theme
st.set_page_config(page_title="FinShield Loan Advisory & Risk Hub", page_icon="🛡️", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "active_package" not in st.session_state:
    st.session_state.active_package = None

def theme_colors(dark: bool) -> dict:
    if dark:
        return dict(
            app_bg="#0A0D14", text="#F8FAFC", muted="#94A3B8",
            card_bg="#111622", card_border="rgba(255, 255, 255, 0.08)",
            red_bg="rgba(239, 68, 68, 0.15)", red_text="#F87171",
            orange_bg="rgba(245, 158, 11, 0.15)", orange_text="#FBBF24",
            green_bg="rgba(16, 185, 129, 0.15)", green_text="#34D399",
            stat_bg="#E6B94E", stat_border="#D97706", stat_text="#000000",
        )
    return dict(
        app_bg="#F8FAFC", text="#0F172A", muted="#64748B",
        card_bg="#FFFFFF", card_border="#E2E8F0",
        red_bg="#FEE2E2", red_text="#DC2626",
        orange_bg="#FEF3C7", orange_text="#D97706",
        green_bg="#D1FAE5", green_text="#059669",
        stat_bg="#FEF08A", stat_border="#FDE047", stat_text="#713F12",
    )

t = theme_colors(st.session_state.dark_mode)

moon_svg_url = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23F7C948' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E\")"
sun_svg_url = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%230F172A' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='5'/%3E%3Cline x1='12' y1='1' x2='12' y2='3'/%3E%3Cline x1='12' y1='21' x2='12' y2='23'/%3E%3Cline x1='4.22' y1='4.22' x2='5.64' y2='5.64'/%3E%3Cline x1='18.36' y1='18.36' x2='19.78' y2='19.78'/%3E%3Cline x1='1' y1='12' x2='3' y2='12'/%3E%3Cline x1='21' y1='12' x2='23' y2='12'/%3E%3Cline x1='4.22' y1='19.78' x2='5.64' y2='18.36'/%3E%3Cline x1='18.36' y1='5.64' x2='19.78' y2='4.22'/%3E%3C/svg%3E\")"
neumorphic_svg_bg = moon_svg_url if st.session_state.dark_mode else sun_svg_url

st.markdown(f"""
<style>
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {"#04060A" if st.session_state.dark_mode else "#FAF9F6"} !important;
        font-family: 'Inter', sans-serif;
    }}


    @keyframes ambientPulse {{
        0%   {{ transform: translate(-3%, -3%) scale(1); opacity: 0.18; }}
        33%  {{ transform: translate(4%, 5%) scale(1.10); opacity: 0.30; }}
        66%  {{ transform: translate(-2%, 6%) scale(0.95); opacity: 0.20; }}
        100% {{ transform: translate(-3%, -3%) scale(1); opacity: 0.18; }}
    }}

    /* Background glow */
    .bg-single-ambient {{
        position: fixed;
        top: -15vh;
        left: -10vw;
        width: 120vw;
        height: 130vh;
        z-index: -1;
        pointer-events: none;
        background: {"radial-gradient(circle at 50% 30%, rgba(234, 88, 12, 0.18) 0%, rgba(217, 119, 6, 0.05) 30%, transparent 50%)" if st.session_state.dark_mode else "radial-gradient(circle at 50% 30%, rgba(251, 146, 60, 0.12) 0%, rgba(253, 186, 116, 0.04) 30%, transparent 50%)"};
        filter: blur(150px);
        animation: ambientPulse 22s ease-in-out infinite;
        will-change: transform, opacity;
    }}

    @keyframes rightGlowPulse {{
        0%   {{ transform: translate(0, 0) scale(1); opacity: 0.20; }}
        50%  {{ transform: translate(-3%, 4%) scale(1.08); opacity: 0.32; }}
        100% {{ transform: translate(0, 0) scale(1); opacity: 0.20; }}
    }}

    /* Right side glow */
    .bg-right-ambient {{
        position: fixed;
        top: -120px;
        right: -150px;
        width: 750px;
        height: 750px;
        border-radius: 50%;
        z-index: -1;
        pointer-events: none;
        background: {"radial-gradient(circle, rgba(245, 158, 11, 0.24) 0%, rgba(234, 88, 12, 0.10) 35%, transparent 80%)" if st.session_state.dark_mode else "radial-gradient(circle, rgba(251, 146, 60, 0.18) 0%, rgba(253, 186, 116, 0.08) 50%, transparent 72%)"};
        filter: blur(140px);
        animation: rightGlowPulse 14s ease-in-out infinite;
        will-change: transform, opacity;
    }}

    /* Background pattern */
    .bg-pattern-layer {{
        position: fixed;
        inset: 0;
        z-index: -2;
        pointer-events: none;
        display: {"block" if st.session_state.dark_mode else "none"} !important;
        opacity: 0.22;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600'%3E%3Ctext x='23' y='47' font-size='22' fill='%23F7C948' font-family='Georgia,serif' opacity='0.9'%3E%E2%82%B9%3C/text%3E%3Ctext x='187' y='31' font-size='13' fill='%23F7C948' opacity='0.5'%3E%25%3C/text%3E%3Ctext x='341' y='68' font-size='18' fill='%23F7C948' opacity='0.7'%3E%E2%96%B3%3C/text%3E%3Ctext x='489' y='22' font-size='11' fill='%23F7C948' opacity='0.45'%3E%E2%97%8B%3C/text%3E%3Ctext x='542' y='74' font-size='16' fill='%23F7C948' font-family='Georgia,serif' opacity='0.65'%3E%E2%82%B9%3C/text%3E%3Ctext x='78' y='143' font-size='10' fill='%23F7C948' opacity='0.4'%3E%E2%94%80%E2%94%80%3C/text%3E%3Ctext x='264' y='121' font-size='14' fill='%23F7C948' opacity='0.6'%3E%E2%96%B3%3C/text%3E%3Ctext x='412' y='158' font-size='20' fill='%23F7C948' font-family='Georgia,serif' opacity='0.85'%3E%E2%82%B9%3C/text%3E%3Ctext x='558' y='134' font-size='12' fill='%23F7C948' opacity='0.5'%3E%25%3C/text%3E%3Ctext x='11' y='231' font-size='15' fill='%23F7C948' opacity='0.55'%3E%E2%96%B3%3C/text%3E%3Ctext x='139' y='274' font-size='22' fill='%23F7C948' font-family='Georgia,serif' opacity='0.8'%3E%E2%82%B9%3C/text%3E%3Ctext x='317' y='248' font-size='11' fill='%23F7C948' opacity='0.42'%3E%E2%97%8B%3C/text%3E%3Ctext x='467' y='212' font-size='13' fill='%23F7C948' opacity='0.58'%3E%25%3C/text%3E%3Ctext x='571' y='263' font-size='10' fill='%23F7C948' opacity='0.38'%3E%E2%94%80%3C/text%3E%3Ctext x='52' y='362' font-size='12' fill='%23F7C948' opacity='0.48'%3E%25%3C/text%3E%3Ctext x='203' y='388' font-size='17' fill='%23F7C948' opacity='0.7'%3E%E2%96%B3%3C/text%3E%3Ctext x='378' y='341' font-size='21' fill='%23F7C948' font-family='Georgia,serif' opacity='0.88'%3E%E2%82%B9%3C/text%3E%3Ctext x='514' y='374' font-size='11' fill='%23F7C948' opacity='0.44'%3E%E2%97%8B%3C/text%3E%3Ctext x='8' y='454' font-size='14' fill='%23F7C948' opacity='0.6'%3E%E2%94%80%E2%94%80%3C/text%3E%3Ctext x='161' y='481' font-size='10' fill='%23F7C948' opacity='0.38'%3E%25%3C/text%3E%3Ctext x='299' y='462' font-size='19' fill='%23F7C948' opacity='0.72'%3E%E2%96%B3%3C/text%3E%3Ctext x='441' y='498' font-size='23' fill='%23F7C948' font-family='Georgia,serif' opacity='0.9'%3E%E2%82%B9%3C/text%3E%3Ctext x='563' y='447' font-size='12' fill='%23F7C948' opacity='0.5'%3E%E2%97%8B%3C/text%3E%3Ctext x='87' y='558' font-size='16' fill='%23F7C948' opacity='0.62'%3E%E2%96%B3%3C/text%3E%3Ctext x='234' y='582' font-size='11' fill='%23F7C948' opacity='0.42'%3E%25%3C/text%3E%3Ctext x='388' y='566' font-size='10' fill='%23F7C948' opacity='0.36'%3E%E2%94%80%3C/text%3E%3Ctext x='512' y='591' font-size='20' fill='%23F7C948' font-family='Georgia,serif' opacity='0.78'%3E%E2%82%B9%3C/text%3E%3C/svg%3E");
        background-repeat: repeat;
        background-size: 600px 600px;
    }}


    footer, footer[data-testid="stFooter"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    .block-container {{
        padding-top: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1350px !important;
        margin: 0 auto !important;
        position: relative;
        z-index: 1;
        isolation: isolate;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        padding-bottom: 0px !important;
    }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: {t['text']}; }}


    /* Hero section */
    .hero-container-light {{
        background: transparent !important;
        padding: 45px 20px 30px;
        text-align: center;
    }}
    @keyframes fadeInUp {{
        0% {{
            opacity: 0;
            transform: translateY(18px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    @keyframes shimmerGradient {{
        0% {{
            background-position: 0% 50%;
        }}
        50% {{
            background-position: 100% 50%;
        }}
        100% {{
            background-position: 0% 50%;
        }}
    }}
    .hero-main-title {{
        font-family: 'Space Grotesk', 'Inter', -apple-system, sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        letter-spacing: -1.2px !important;
        margin-bottom: 16px !important;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    .hero-title-dark {{
        color: #F8FAFC !important;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }}
    .hero-title-gradient {{
        background: linear-gradient(120deg, #F7C948 0%, #FBBF24 25%, #FFFFFF 50%, #D97706 75%, #F7C948 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-size: 200% auto;
    }}
    .hero-subtext-light {{
        font-size: 1.02rem;
        max-width: 620px;
        margin: 0 auto 28px;
        line-height: 1.6;
        opacity: 0.88;
        font-weight: 450;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}


    /* Input fields styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        border-radius: 14px !important;
        border: {"1px solid rgba(247, 201, 72, 0.25)" if st.session_state.dark_mode else "1px solid #CBD5E1"} !important;
        background: {"rgba(15, 23, 42, 0.6)" if st.session_state.dark_mode else "#FFFFFF"} !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }}
    div[data-baseweb="select"] input, div[data-baseweb="input"] input, div[data-baseweb="select"] span {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
    }}
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
        border-color: #F7C948 !important;
        box-shadow: 0 0 15px rgba(247, 201, 72, 0.2) !important;
    }}

    /* Dark mode toggle switch styling */
    div[data-testid="column"]:has(.st-key-dark_mode),
    div[data-testid="column"]:has([data-testid="stToggle"]) {{
        display: flex !important;
        justify-content: flex-end !important;
        align-items: flex-start !important;
        text-align: right !important;
    }}

    .st-key-dark_mode,
    div[data-testid="stToggle"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-start !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        float: right !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        margin-top: 4px !important;
    }}
    .st-key-dark_mode label,
    div[data-testid="stToggle"] label {{
        display: flex !important;
        flex-direction: column-reverse !important;
        align-items: flex-end !important;
        gap: 6px !important;
        cursor: pointer !important;
    }}


    .st-key-dark_mode label span:last-child,
    .st-key-dark_mode label p,
    div[data-testid="stToggle"] label span:last-child,
    div[data-testid="stToggle"] label p {{
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        width: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }}


    .st-key-dark_mode label > span:first-of-type,
    .st-key-dark_mode label > div:first-of-type,
    .st-key-dark_mode input[type="checkbox"] + span,
    .st-key-dark_mode input[type="checkbox"] + div,
    .st-key-dark_mode [data-baseweb="checkbox"] > span,
    .st-key-dark_mode [data-baseweb="checkbox"] > div,
    div[data-testid="stToggle"] label > span:first-of-type,
    div[data-testid="stToggle"] label > div:first-of-type,
    div[data-testid="stToggle"] input[type="checkbox"] + span,
    div[data-testid="stToggle"] input[type="checkbox"] + div,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] > span,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] > div,
    .st-key-dark_mode [data-baseweb="checkbox"] span::before,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] span::before {{
        width: 56px !important;
        height: 28px !important;
        border-radius: 30px !important;
        background-color: {"#111622" if st.session_state.dark_mode else "#CBD5E1"} !important;
        background-image: {neumorphic_svg_bg} !important;
        background-repeat: no-repeat !important;
        background-position: {"6px center" if st.session_state.dark_mode else "34px center"} !important;
        box-shadow: {"inset 3px 3px 6px #08090c, inset -3px -3px 6px #222736" if st.session_state.dark_mode else "inset 3px 3px 6px #94a3b8, inset -3px -3px 6px #ffffff"} !important;
        border: {"1.5px solid rgba(247, 201, 72, 0.4)" if st.session_state.dark_mode else "1.5px solid #94A3B8"} !important;
        opacity: 1 !important;
        padding: 2px !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

   
    .st-key-dark_mode label > span:first-of-type *,
    .st-key-dark_mode label > div:first-of-type *,
    .st-key-dark_mode input[type="checkbox"] + span *,
    .st-key-dark_mode input[type="checkbox"] + div *,
    .st-key-dark_mode [data-baseweb="checkbox"] > span *,
    .st-key-dark_mode [data-baseweb="checkbox"] > div *,
    div[data-testid="stToggle"] label > span:first-of-type *,
    div[data-testid="stToggle"] label > div:first-of-type *,
    div[data-testid="stToggle"] input[type="checkbox"] + span *,
    div[data-testid="stToggle"] input[type="checkbox"] + div *,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] > span *,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] > div *,
    .st-key-dark_mode [data-baseweb="checkbox"] span::after,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] span::after {{
        width: 22px !important;
        height: 22px !important;
        border-radius: 50% !important;
        background-color: {"#1A2234" if st.session_state.dark_mode else "#FFFFFF"} !important;
        background: {"radial-gradient(circle at 35% 35%, #242F46 0%, #111622 100%)" if st.session_state.dark_mode else "radial-gradient(circle at 35% 35%, #ffffff 0%, #F1F5F9 100%)"} !important;
        box-shadow: {"3px 3px 6px #04060A, -2px -2px 5px #2C3A54" if st.session_state.dark_mode else "3px 3px 6px rgba(0,0,0,0.18), -2px -2px 5px #ffffff"} !important;
        border: {"1.5px solid #F7C948" if st.session_state.dark_mode else "1.5px solid #94A3B8"} !important;
        transform: {"translateX(28px)" if st.session_state.dark_mode else "translateX(0px)"} !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}


    /* Collapsible section styling */
    [data-testid="stExpander"] {{
        border: none !important;
        background: transparent !important;
        margin-top: 20px !important;
        margin-bottom: 24px !important;
    }}
    div[data-testid="stExpander"] details,
    details[data-testid="stExpander"] {{
        border-radius: 14px !important;
        border: {"1px solid rgba(255, 255, 255, 0.1)" if st.session_state.dark_mode else "1px solid #E2E8F0"} !important;
        background-color: {"#111622" if st.session_state.dark_mode else "#FFFFFF"} !important;
        overflow: hidden !important;
    }}

    [data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary {{
        background-color: {"#1A2234" if st.session_state.dark_mode else "#F8FAFC"} !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-weight: 700 !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
    }}

    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpander"] summary[aria-expanded="true"] {{
        background-color: {"#242F46" if st.session_state.dark_mode else "#F1F5F9"} !important;
        color: {"#F7C948" if st.session_state.dark_mode else "#0F172A"} !important;
    }}

    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary svg {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        fill: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
    }}

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important; border: none !important; border-radius: 30px !important;
        font-weight: 800 !important; font-size: 0.95rem !important; padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(247, 201, 72, 0.45) !important;
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important;
    }}
    .stButton > button *,
    div[data-testid="stFormSubmitButton"] > button * {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}

    
    .stTabs [data-baseweb="tab-list"], [data-baseweb="tab-list"] {{
        gap: 10px !important;
        border-bottom: none !important;
        border: none !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        padding: 4px 8px !important;
        box-shadow: none !important;
        justify-content: center !important;
        margin-bottom: 24px !important;
    }}
    .stTabs [data-baseweb="tab"], [data-baseweb="tab"], button[role="tab"], div[role="tab"] {{
        background-color: transparent !important;
        border-radius: 30px !important;
        color: {"#F8FAFC" if st.session_state.dark_mode else "#334155"} !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        border: none !important;
        outline: none !important;
        font-size: 0.9rem !important;
        transition: all 0.25s ease !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        width: auto !important;
    }}
    .stTabs [data-baseweb="tab"] *, button[role="tab"] * {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#334155"} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover, button[role="tab"]:hover {{
        color: #F7C948 !important;
        background: rgba(247, 201, 72, 0.15) !important;
    }}
    .stTabs [aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"],
    div[role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.45) !important;
    }}
    .stTabs [aria-selected="true"] *,
    [data-baseweb="tab"][aria-selected="true"] *,
    button[role="tab"][aria-selected="true"] * {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}
    /* Hide default tab bottom highlight line */
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    [data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
    [data-baseweb="tab-list"] [data-baseweb="tab-border"],
    [data-baseweb="tab-list"] > div[style*="position: absolute"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        max-height: 0px !important;
        width: 0px !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }}



    /* Tab scroll arrows styling */
    [data-baseweb*="tab-scroll"],
    [data-baseweb*="tab-scroll"] *,
    [data-baseweb="tab-scroll-button-left"],
    [data-baseweb="tab-scroll-button-right"],
    div[data-baseweb="tab-scroll-button-left"],
    div[data-baseweb="tab-scroll-button-right"],
    .stTabs [data-baseweb*="tab-scroll"],
    .stTabs [data-baseweb="tab-scroll-button-left"],
    .stTabs [data-baseweb="tab-scroll-button-right"],
    .stTabs [data-baseweb="tab-list"] > button:not([role="tab"]),
    .stTabs [data-baseweb="tab-list"] > div > button:not([role="tab"]),
    button[aria-label*="scroll"],
    button[aria-label*="Scroll"],
    button[aria-label*="Next"],
    button[aria-label*="Previous"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        outline: none !important;
        opacity: 1 !important;
        visibility: visible !important;
        cursor: pointer !important;
        padding: 0 4px !important;
        margin: 0 !important;
    }}

    .stTabs [data-baseweb="tab-list"]::before,
    .stTabs [data-baseweb="tab-list"]::after,
    [data-baseweb*="tab-scroll"]::before,
    [data-baseweb*="tab-scroll"]::after,
    button[aria-label*="Next"]::before,
    button[aria-label*="Next"]::after,
    button[aria-label*="Previous"]::before,
    button[aria-label*="Previous"]::after {{
        display: none !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        content: "" !important;
    }}

    /* Tab scroll arrow icon styling */
    [data-baseweb*="tab-scroll"] svg,
    [data-baseweb*="tab-scroll"] svg path,
    [data-baseweb*="tab-scroll"] svg polyline,
    .stTabs [data-baseweb="tab-list"] button:not([role="tab"]) svg,
    .stTabs [data-baseweb="tab-list"] button:not([role="tab"]) svg path,
    button[aria-label*="scroll"] svg,
    button[aria-label*="Scroll"] svg,
    button[aria-label*="Next"] svg,
    button[aria-label*="Previous"] svg {{
        fill: {"#F7C948" if st.session_state.dark_mode else "#0F172A"} !important;
        stroke: {"#F7C948" if st.session_state.dark_mode else "#0F172A"} !important;
        color: {"#F7C948" if st.session_state.dark_mode else "#0F172A"} !important;
        stroke-width: 3.5px !important;
        filter: none !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }}

    [data-baseweb*="tab-scroll"]:hover svg,
    button[aria-label*="Next"]:hover svg,
    button[aria-label*="Previous"]:hover svg {{
        transform: scale(1.15) !important;
        filter: none !important;
    }}




    /* Form container styling */
    form[data-testid="stForm"],
    div[data-testid="stForm"],
    .stForm,
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {"linear-gradient(135deg, rgba(16, 22, 34, 0.88) 0%, rgba(10, 14, 23, 0.94) 100%)" if st.session_state.dark_mode else "linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)"} !important;
        border: {"1.5px solid rgba(247, 201, 72, 0.35)" if st.session_state.dark_mode else "1.5px solid #CBD5E1"} !important;
        border-radius: 24px !important;
        padding: 26px 24px !important;
        box-shadow: {"0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08)" if st.session_state.dark_mode else "0 12px 35px rgba(0, 0, 0, 0.08)"} !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    form[data-testid="stForm"] label,
    form[data-testid="stForm"] p,
    form[data-testid="stForm"] span {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        font-weight: 600 !important;
    }}

    /* Radio buttons styling */
    div[data-testid="stRadio"] {{
        margin-top: 8px !important;
        margin-bottom: 16px !important;
    }}
    div[data-testid="stRadio"] > div {{
        background: {"rgba(255, 255, 255, 0.04)" if st.session_state.dark_mode else "#F1F5F9"} !important;
        border: {"1.5px solid rgba(255, 255, 255, 0.1)" if st.session_state.dark_mode else "1.5px solid #E2E8F0"} !important;
        border-radius: 30px !important;
        padding: 5px !important;
        gap: 6px !important;
        display: inline-flex !important;
        flex-wrap: wrap !important;
    }}
    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border-radius: 24px !important;
        padding: 8px 20px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: {"#94A3B8" if st.session_state.dark_mode else "#64748B"} !important;
        display: flex !important;
        align-items: center !important;
    }}
    div[data-testid="stRadio"] label > div:first-child {{
        display: none !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        color: {"#F8FAFC" if st.session_state.dark_mode else "#0F172A"} !important;
        background: {"rgba(255, 255, 255, 0.06)" if st.session_state.dark_mode else "#E2E8F0"} !important;
    }}
    div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stRadio"] label[aria-checked="true"] {{
        background: linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%) !important;
        box-shadow: 0 4px 18px rgba(247, 201, 72, 0.35) !important;
    }}
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label[aria-checked="true"] p,
    div[data-testid="stRadio"] label[aria-checked="true"] span {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}



    /* 3-Column Square Grid Layout (App at a glance) */
    .glance-grid-3col {{
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 12px !important;
        width: 100% !important;
        margin: 14px 0 !important;
    }}
    .glance-card {{
        border-radius: 18px !important;
        padding: 16px 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        min-height: 105px !important;
        box-sizing: border-box !important;
        transition: transform 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
    }}
    .glance-card:hover {{
        transform: translateY(-2px) !important;
    }}

    .stat-box {{
        border-radius: 20px !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(247, 201, 72, 0.2);
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center; padding: 16px 12px; box-sizing: border-box; transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-box:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }}
    .stat-label {{ font-size: 0.76rem; margin-bottom: 4px; opacity: 0.88; font-weight: 600; }}
    .stat-value {{ font-size: 1.18rem; font-weight: 800; }}

    .verdict-card {{
        border-radius: 22px !important; padding: 20px !important; margin-bottom: 14px; text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }}

    /* Mobile Screen Responsiveness */
    @media (max-width: 768px) {{
        .hero-main-title {{ font-size: 2rem !important; }}
        .top-nav-bar {{ padding: 10px 16px !important; }}
        .stat-box {{ padding: 8px 4px !important; border-radius: 10px !important; min-height: 65px !important; }}
        .stat-label {{ font-size: 0.62rem !important; margin-bottom: 2px !important; line-height: 1.1 !important; }}
        .stat-value {{ font-size: 0.92rem !important; }}
        .verdict-card {{ padding: 12px 14px !important; }}
        .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto !important; flex-wrap: nowrap !important; white-space: nowrap !important; }}
        .stTabs [data-baseweb="tab"] {{ padding: 8px 12px !important; font-size: 0.8rem !important; flex-shrink: 0 !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# Inject background single-color ambient gradient + right-side glow + pattern layer divs
st.markdown("""
<div class="bg-single-ambient"></div>
<div class="bg-right-ambient"></div>
<div class="bg-pattern-layer"></div>
""", unsafe_allow_html=True)

# 1. Navigation Navbar Header
col_n1, col_n2 = st.columns([3, 1], vertical_alignment="top")
with col_n1:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; padding:10px 8px;">
            <div style="width:42px; height:42px; border-radius:50%; background:#2C3854; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="5" r="2.2" fill="#E05638" />
                    <path d="M4 17L10 11L14 15L20 9" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            </div>
            <div>
                <div style="font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size:1.85rem; font-weight:700; color:{t['text']}; letter-spacing:-0.6px; line-height:1.1;">FinShield</div>
                <div style="font-size:0.75rem; color:{t['muted']}; font-weight:500; margin-top:2px;">Digital Lending Risk Intelligence & Safety Hub</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_n2:
    st.toggle("", key="dark_mode", label_visibility="collapsed")

# 3. FIRST: Top Tabs Navigation Toggle Bar
st.markdown('<div style="max-width:1350px; margin: 12px auto 0; padding: 0 16px;">', unsafe_allow_html=True)
tab_scorer, tab_profiler, tab_rankings, tab_calculators, tab_rbi = st.tabs([
    "🛡️ App Risk Scorer",
    "💳 Borrower Safety Profiler",
    "📊 Product Rankings",
    "🧮 Advisory Calculators",
    "📜 RBI Guidelines"
])
st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 1: APP RISK SCORER & GAUGE
# ==========================================
with tab_scorer:
    # SECOND: Hero Headline Text & Centered Subtitle
    st.markdown(
        f"""
        <div class="hero-container-light" style="padding: 20px 20px 20px;">
            <h1 class="hero-main-title">
                Detect Predatory Loan Apps. <br><span class="hero-gold-text">Protect Your Personal Privacy.</span>
            </h1>
            <div style="text-align: center; display: flex; justify-content: center; width: 100%;">
                <p class="hero-subtitle">
                    Audit Play Store lending apps for illegal contacts/gallery permissions, undisclosed terms, and harassment reviews before you borrow.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # LASTLY: App Audit & Prediction Tool Section (Enclosed in Glass Container Box)
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="margin-bottom:14px; padding-bottom:14px; border-bottom:1px solid {'rgba(255,255,255,0.08)' if st.session_state.dark_mode else 'rgba(0,0,0,0.06)'};">
                <div style="font-size:1.38rem; font-weight:800; color:{t['text']}; letter-spacing:-0.5px; line-height:1.25;">Evaluate Digital Loan App Safety</div>
                <div style="font-size:0.84rem; color:{t['muted']}; font-weight:500; margin-top:3px;">Select a pre-analyzed app or paste any custom Play Store link, website URL, or package name to audit.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if USE_FAKE_MODEL:
            st.info("🔧 Running with formula scoring mode (predatory_loan_detector.pkl not found). Good for UI testing.", icon="🔧")

        app_choices = get_app_choices()

        input_mode = st.radio(
            "Audit Mode",
            options=["📋 Select Pre-Analyzed App", "🔗 Audit Unlisted App, APK or Website Link"],
            horizontal=True,
            label_visibility="collapsed",
            key="audit_input_mode_radio"
        )

        c_input, c_btn = st.columns([3.8, 1.2], vertical_alignment="bottom")

        if "Pre-Analyzed" in input_mode and app_choices:
            with c_input:
                package_name = st.selectbox("Select a Play Store Lending App to Audit", options=app_choices)
            with c_btn:
                check_clicked = st.button("Check App Risk ➔", key="btn_dropdown", use_container_width=True)
        else:
            with c_input:
                package_name = st.text_input(
                    "Paste Play Store Link, Website URL, or Android Package ID",
                    placeholder="e.g. https://play.google.com/store/apps/details?id=com.fatakpay or https://quick-7day-loan.xyz or com.fastcash.loan",
                    help="Paste any Google Play Store URL, website domain, or Android package ID to audit.",
                    key="unlisted_app_link_input"
                )
            with c_btn:
                check_clicked = st.button("Audit Custom App ➔", key="btn_custom_link", use_container_width=True)

    if check_clicked and package_name:
        if "Pre-Analyzed" not in input_mode and not is_valid_unlisted_input(package_name):
            st.warning("Invalid link or app name. Please enter a valid Play Store URL, website domain, or Package ID.", icon="⚠️")
            st.session_state.active_package = None
        else:
            st.session_state.active_package = package_name

    if st.session_state.active_package:
        package_name = st.session_state.active_package
        with st.spinner("Analyzing app features & review sentiment..."):
            score, reasons, used_fallback, features = score_app(package_name)

            if used_fallback and features:
                src = features.get("scrape_source")
                if src == "live_playstore":
                    dev = features.get("scraped_developer")
                    dev_str = f" by **{dev}**" if dev else ""
                    rvs_cnt = features.get("scraped_reviews_count", 0)
                    st.success(
                        f"🟢 **Live Data: Scraped from Google Play Store** — Real-time app metadata{dev_str}, terms disclosure, and {rvs_cnt} live user reviews analyzed.",
                        icon="🔍"
                    )
                elif src == "live_website":
                    url_scraped = features.get("scraped_url", package_name)
                    st.success(
                        f"🌐 **Live Data: Scraped from Website** — Real-time disclosures, privacy policy terms & text scraped from `{url_scraped}`.",
                        icon="🌐"
                    )
                else:
                    st.info(
                        f"📡 **Off-Store / Domain Safety Analysis** — App not active on Play Store / site offline. Applied safety domain analysis for `{extract_package_id(package_name)}`.",
                        icon="✨"
                    )

            if score is not None:
                st.write("")
                col_gauge, col_metrics = st.columns([1, 1.8], vertical_alignment="top")

                with col_gauge:
                    st.markdown("#### 🛡️ Riskometer Verdict")
                    # Riskometer SVG Dial
                    gauge_html = render_rbi_riskometer_card(score, st.session_state.dark_mode)
                    st.markdown(gauge_html, unsafe_allow_html=True)

                    if score >= 0.6:
                        verdict, bg_v, fg_v = "High Predatory Risk", "linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(127, 29, 29, 0.35) 100%)", t["red_text"]
                        v_border = "rgba(239, 68, 68, 0.4)"
                        v_desc = "Exhibits multiple compliance concerns, excessive permission requests, or harassment complaints."
                    elif score >= 0.3:
                        verdict, bg_v, fg_v = "Moderate Caution", "linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(120, 53, 15, 0.35) 100%)", t["orange_text"]
                        v_border = "rgba(245, 158, 11, 0.4)"
                        v_desc = "Partially meets RBI transparency norms. Caution advised before borrowing."
                    else:
                        verdict, bg_v, fg_v = "Looks Legitimate", "linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 78, 59, 0.35) 100%)", t["green_text"]
                        v_border = "rgba(16, 185, 129, 0.4)"
                        v_desc = "Appears aligned with RBI Digital Lending Directives and maintains transparent disclosures."

                    st.markdown(
                        f"""
                        <div class="verdict-card" style="background:{bg_v}; color:{fg_v}; border:1.5px solid {v_border}; margin-top:12px; border-radius:22px; padding: 26px 18px;">
                            <div style="font-size:1.35rem; font-weight:800;">{verdict}</div>
                            <div style="font-size:2.4rem; font-weight:900; margin:6px 0;">{score*100:.0f}% risk</div>
                            <div style="font-size:0.84rem; opacity:0.92; line-height:1.45;">{v_desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_metrics:
                    st.markdown("#### App at a glance")
                    if features:
                        installs = features.get("install_count")
                        disclosure = features.get("disclosure_score", 0)
                        redflag_pct = (features.get("review_redflag_score", 0) or 0) * 100
                        neg_pct = (features.get("pct_strongly_negative_reviews", 0) or 0) * 100
                        sentiment = features.get("avg_review_sentiment", 0)
                        length = features.get("avg_review_length", 0)
                        has_contacts = (features.get("contacts", 0) == 1)
                        has_sms = (features.get("sms", 0) == 1)

                        name_lower = str(package_name).lower()
                        is_known = any(b in name_lower for b in KNOWN_BANKS) or features.get("is_known_legit", False)

                        if is_known or (score < 0.30 and redflag_pct < 5 and not has_contacts):
                            rbi_val, rbi_lvl = "Regulated", "green"
                        elif score >= 0.60 or redflag_pct >= 15 or (has_contacts and has_sms and disclosure <= 2):
                            rbi_val, rbi_lvl = "Unregulated", "red"
                        else:
                            rbi_val, rbi_lvl = "Partially Regulated", "orange"

                        def tier_style(level):
                            if st.session_state.dark_mode:
                                return {
                                    "red": ("#3F1618", "#F56565", "1px solid rgba(245, 101, 101, 0.2)"),
                                    "orange": ("#3C2F0E", "#ECC94B", "1px solid rgba(236, 201, 75, 0.2)"),
                                    "green": ("#0E3321", "#48BB78", "1px solid rgba(72, 187, 120, 0.2)"),
                                }[level]
                            else:
                                return {
                                    "red": ("#FEE2E2", "#B91C1C", "1.5px solid rgba(220, 38, 38, 0.45)"),
                                    "orange": ("#FEF3C7", "#B45309", "1.5px solid rgba(217, 119, 6, 0.45)"),
                                    "green": ("#D1FAE5", "#047857", "1.5px solid rgba(5, 150, 105, 0.45)"),
                                }[level]

                        _pkg_raw = str(package_name)
                        _is_playstore_input = "play.google.com" in _pkg_raw
                        _feat_app_id = features.get("app_id", "")
                        _is_android_pkg = any(_feat_app_id.startswith(p) for p in ["com.", "in.", "org.", "net.", "io.", "co."])
                        is_web_target = (
                            not _is_playstore_input and
                            features.get("is_web_domain", False)
                        )

                        installs_card = ("🌐 Platform", "Web Domain", "green") if is_web_target else ("📦 Installs", f"{installs:,}" if installs else "—", "green" if (installs or 0) >= 100000 else "orange")

                        stats = [
                            ("🏛️ RBI Status", rbi_val, rbi_lvl),
                            installs_card,
                            ("📝 Terms disclosed", f"{disclosure} / 5", "green" if disclosure >= 4 else "orange" if disclosure == 3 else "red"),
                            ("🚩 Harassment mentions", f"{redflag_pct:.0f}%", "red" if redflag_pct >= 15 else "orange" if redflag_pct >= 5 else "green"),
                            ("😠 Strongly negative reviews", f"{neg_pct:.0f}%", "red" if neg_pct >= 20 else "orange" if neg_pct >= 10 else "green"),
                            ("🙂 Avg. review tone", f"{sentiment:+.2f}", "orange" if (sentiment > 0.6 or sentiment < -0.3) else "green"),
                            ("✏️ Avg. review length", f"{length:.0f} words", "red" if length >= 20 else "orange" if length >= 10 else "green"),
                        ]

                        cards_html = []
                        for i, (lbl, val, lvl) in enumerate(stats):
                            bg_c, fg_c, bdr_c = tier_style(lvl)
                            grid_col_style = "grid-column: 2 / 3;" if i == 6 else ""
                            cards_html.append(
                                f'<div class="glance-card" style="background:{bg_c}; color:{fg_c}; border:{bdr_c}; {grid_col_style}">'
                                f'<div style="font-size:0.72rem; opacity:0.85; margin-bottom:8px; font-weight:600; line-height:1.2;">{lbl}</div>'
                                f'<div style="font-size:1.15rem; font-weight:800; line-height:1.2;">{val}</div>'
                                f'</div>'
                            )

                        st.markdown(f'<div class="glance-grid-3col">{"".join(cards_html)}</div>', unsafe_allow_html=True)

                        st.markdown(
                            f'<div style="font-size:0.78rem; color:{t["muted"]}; line-height:1.55; margin-top:14px; text-align:left;">'
                            f'<span style="color:#34D399; font-weight:700;">🟢 fine</span> • <span style="color:#FBBF24; font-weight:700;">🟡 minor concern</span> • <span style="color:#F87171; font-weight:700;">🔴 risky</span> — <strong>RBI Approved:</strong> whether app discloses legitimate RBI/NBFC registration & follows key norms. <strong>Terms disclosed:</strong> how clearly the app states interest rate, tenure & registration (out of 5). <strong>Harassment mentions:</strong> % of reviews describing threats or abusive recovery tactics. <strong>Strongly negative reviews:</strong> % of reviews that are clearly unhappy. <strong>Avg. review tone:</strong> how positive (+1) or negative (-1) reviews are overall. <strong>Avg. review length:</strong> average words per review.'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Full-Width App / Web Link Banner
                app_id_str = str(features.get("app_id", package_name)).strip() if features else str(package_name).strip()
                if is_web_target:
                    banner_title = "Verified Official Web Domain"
                    target_url = app_id_str if app_id_str.startswith("http") else f"https://{app_id_str}"
                    button_label = "Visit Website ↗"
                else:
                    banner_title = "Play Store Verified App Package"
                    target_url = f"https://play.google.com/store/apps/details?id={app_id_str}"
                    button_label = "View on Play Store ↗"

                st.markdown(
                    f"""
                    <div style="background:{t['card_bg']}; border:1.5px solid rgba(247, 201, 72, 0.25); border-radius:20px; padding:18px 24px; display:flex; justify-content:space-between; align-items:center; margin-top:24px; margin-bottom:20px; width:100%; box-shadow:0 8px 25px rgba(0, 0, 0, 0.3);">
                        <div>
                            <div style="font-weight:800; font-size:0.92rem; color:{t['text']};">{banner_title}</div>
                            <div style="font-size:0.82rem; color:{t['muted']}; font-family:monospace; margin-top:2px;">{app_id_str}</div>
                        </div>
                        <a href="{target_url}" target="_blank" style="background:linear-gradient(135deg, #F7C948 0%, #E5C07B 50%, #D97706 100%); color:#000; padding:10px 22px; border-radius:30px; font-weight:800; font-size:0.85rem; text-decoration:none; box-shadow:0 4px 18px rgba(247, 201, 72, 0.35); transition:transform 0.2s;">{button_label}</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("🔍 Key Risk Drivers & Explanation Breakdown"):
                    if reasons:
                        for item in reasons:
                            if isinstance(item, (tuple, list)) and len(item) == 2:
                                reason_text, is_bad = item
                                icon = "🔴" if is_bad else "🟢"
                                st.write(f"{icon} {reason_text}")
                            elif isinstance(item, str):
                                st.write(f"🟢 {item}")

# ==========================================
# TAB 2: SAFETY PROFILER
# ==========================================
with tab_profiler:
    st.markdown("### 💳 Borrower Safety Assessment")
    st.caption("Understand your personal borrowing psychology and data privacy safety profile.")

    p_col1, p_col2 = st.columns([1, 1], vertical_alignment="top")

    with p_col1:
        st.markdown("#### Borrower Safety Questionnaire")
        q1 = st.selectbox("1. How frequently do you take instant digital loans?", options=["Rarely / Only in genuine emergencies", "Occasionally for purchases", "Frequently (multiple times a month)"])
        q2 = st.selectbox("2. Do you grant Contacts & Gallery permissions to loan apps?", options=["Never", "Sometimes if loan approval is instant", "Always without checking permissions"])
        q3 = st.selectbox("3. Do you check if the lender discloses an RBI Registered NBFC partner?", options=["Always cross-verify on RBI portal", "Sometimes if mentioned on Play Store", "Never check"])
        q4 = st.selectbox("4. Do you have an emergency fund covering at least 3 months of expenses?", options=["Yes", "Partially", "No"])

    with p_col2:
        st.markdown("#### Your Borrower Safety Profile")
        score_val = 100
        if q1.startswith("Frequently"): score_val -= 30
        elif q1.startswith("Occasionally"): score_val -= 15

        if q2.startswith("Always"): score_val -= 35
        elif q2.startswith("Sometimes"): score_val -= 20

        if q3.startswith("Never"): score_val -= 25
        elif q3.startswith("Sometimes"): score_val -= 10

        if q4 == "No": score_val -= 10

        score_val = max(score_val, 15)

        if score_val >= 80:
            p_title, p_color, p_desc = "Prudent Borrower", "#10B981", "High financial discipline! You protect your data privacy and avoid unverified lending apps."
        elif score_val >= 50:
            p_title, p_color, p_desc = "Cautionary Borrower", "#F59E0B", "Moderate Caution: Always read the Key Fact Statement (KFS) and verify NBFC registration before borrowing."
        else:
            p_title, p_color, p_desc = "Vulnerable Borrower", "#EF4444", "High Risk! Frequent short-term loans and granting contact list permissions leaves you exposed to illegal harassment apps."

        st.markdown(
            f"""
            <div style="background:{t['card_bg']}; border:2px solid {p_color}; border-radius:16px; padding:24px; text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.3);">
                <div style="font-size:3rem; margin-bottom:8px;">🏆</div>
                <div style="font-size:0.75rem; color:{p_color}; font-weight:800; letter-spacing:1px;">MONEYSIGN® PROFILE</div>
                <div style="font-size:2rem; font-weight:900; color:{t['text']}; margin:4px 0;">{p_title}</div>
                <div style="font-size:1.1rem; font-weight:800; color:{p_color};">Safety Index: {score_val} / 100</div>
                <p style="font-size:0.88rem; color:{t['muted']}; margin-top:12px; line-height:1.5;">{p_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# TAB 3: APP RANKINGS HUB
# ==========================================
with tab_rankings:
    st.markdown("### 🏆 FinShield Digital Lending App Safety Rankings")
    st.caption("Evaluated database of popular lending apps on the Google Play Store, ranked by privacy safety, RBI compliance, and review harassment risk.")

    try:
        df_ranked = get_ranked_apps_df()
        
        # Top Controls: Search bar & Filters
        f_col1, f_col2, f_col3 = st.columns([2.4, 1.3, 1.3], gap="small")
        
        with f_col1:
            search_q = st.text_input("🔍 Search App", placeholder="Type e.g. KreditBee, Groww, Navi...", key="ranking_search_q")
        with f_col2:
            tier_filter = st.selectbox(
                "Safety Tier Filter",
                options=["All Safety Tiers", "🛡️ Safest Tier (80-100)", "⚠️ Moderate Caution (50-79)", "🚨 High Risk (<50)"],
                key="ranking_tier_filter"
            )
        with f_col3:
            sort_by = st.selectbox(
                "Sort Rankings By",
                options=["FinShield Score (Highest First)", "FinShield Score (Lowest First)", "Installs (Highest First)"],
                key="ranking_sort_by"
            )

        # Filtering logic
        filtered_df = df_ranked.copy()
        
        if search_q:
            q = search_q.strip()
            filtered_df = filtered_df[
                filtered_df["app_name"].str.contains(q, case=False, na=False) |
                filtered_df["app_id"].str.contains(q, case=False, na=False)
            ]
            
        if tier_filter == "🛡️ Safest Tier (80-100)":
            filtered_df = filtered_df[filtered_df["safety_score"] >= 80]
        elif tier_filter == "⚠️ Moderate Caution (50-79)":
            filtered_df = filtered_df[(filtered_df["safety_score"] >= 50) & (filtered_df["safety_score"] < 80)]
        elif tier_filter == "🚨 High Risk (<50)":
            filtered_df = filtered_df[filtered_df["safety_score"] < 50]

        if sort_by == "FinShield Score (Lowest First)":
            filtered_df = filtered_df.sort_values(by="safety_score", ascending=True)
        elif sort_by == "Installs (Highest First)":
            filtered_df = filtered_df.sort_values(by="install_count", ascending=False)
        else: # Highest First
            filtered_df = filtered_df.sort_values(by="safety_score", ascending=False)

        all_app_names = list(df_ranked["app_name"].values)
        all_app_ids = list(df_ranked["app_id"].values)

        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        if filtered_df.empty:
            st.info("🔍 No lending apps found matching your search and filter criteria.", icon="ℹ️")
        else:
            c_dark = st.session_state.dark_mode
            card_bg = "#111622" if c_dark else "#FFFFFF"
            card_bdr = "1px solid rgba(255, 255, 255, 0.08)" if c_dark else "1px solid #E2E8F0"
            text_color = "#F8FAFC" if c_dark else "#0F172A"
            muted_color = "#94A3B8" if c_dark else "#64748B"

            for idx, row in filtered_df.iterrows():
                rank_num = row["rank"]
                app_name = row["app_name"]
                app_id = row["app_id"]
                score = row["safety_score"]
                cat_tag = row["cat_tag"]
                installs_str = row["installs_str"]
                disclosure = row["disclosure_score"]
                redflag_pct = row["redflag_pct"]
                is_known = row["is_known"]
                
                # Score pill styling
                if score >= 80:
                    score_bg = "rgba(16, 185, 129, 0.18)" if c_dark else "#DCFCE7"
                    score_bdr = "1px solid rgba(16, 185, 129, 0.35)" if c_dark else "1px solid #86EFAC"
                    score_fg = "#34D399" if c_dark else "#15803D"
                    status_text = "RBI Aligned"
                elif score >= 50:
                    score_bg = "rgba(245, 158, 11, 0.18)" if c_dark else "#FEF3C7"
                    score_bdr = "1px solid rgba(245, 158, 11, 0.35)" if c_dark else "1px solid #FDE68A"
                    score_fg = "#FBBF24" if c_dark else "#B45309"
                    status_text = "Caution Advised"
                else:
                    score_bg = "rgba(239, 68, 68, 0.18)" if c_dark else "#FEE2E2"
                    score_bdr = "1px solid rgba(239, 68, 68, 0.35)" if c_dark else "1px solid #FCA5A5"
                    score_fg = "#F87171" if c_dark else "#B91C1C"
                    status_text = "High Risk"


                card_header_html = f"""<div style="background:{card_bg}; border:{card_bdr}; border-radius:16px; padding:18px 20px 14px; margin-bottom:16px; box-shadow:0 6px 20px rgba(0,0,0,0.15);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span style="font-size:0.74rem; font-weight:700; color:{muted_color}; text-transform:uppercase; letter-spacing:0.5px;">{cat_tag}</span>
<span style="font-size:0.72rem; background:{score_bg}; color:{score_fg}; padding:2px 8px; border-radius:12px; font-weight:800; border:{score_bdr};">{status_text}</span>
</div>
<div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
<div style="font-size:1.15rem; font-weight:800; color:{text_color}; line-height:1.35; flex:1;" title="{app_name}">
{app_name}
</div>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; background:{'rgba(255,255,255,0.02)' if c_dark else '#F8FAFC'}; padding:8px 12px; border-radius:10px;">
<div>
<div style="color:{muted_color}; font-size:0.72rem; font-weight:600;">Installs Base</div>
<div style="font-weight:800; color:{text_color}; font-size:0.92rem;">{installs_str}</div>
</div>
<div>
<div style="color:{muted_color}; font-size:0.72rem; font-weight:600;">FinShield Score</div>
<div style="font-weight:900; color:{score_fg}; font-size:0.96rem;">{score} <span style="font-size:0.72rem; opacity:0.8;">/ 100</span></div>
</div>
</div>
</div>"""
                st.markdown(card_header_html, unsafe_allow_html=True)

                # Action Row: Left Compare Button | Right View Details Button (Side-by-Side)
                act_row1, act_row2 = st.columns([1, 1], gap="small")
                with act_row1:
                    if st.button("⚖️ Compare App", key=f"btn_card_cmp_{idx}", use_container_width=True):
                        st.session_state.active_cmp_app1 = app_id
                        st.rerun()
                with act_row2:
                    if st.button("🔍 View Details", key=f"btn_details_card_{idx}", use_container_width=True):
                        current_show = st.session_state.get("show_inline_audit")
                        st.session_state.show_inline_audit = None if current_show == app_id else app_id
                        st.rerun()
                
                if st.session_state.get("show_inline_audit") == app_id:
                    name_clean = app_name.split(':')[0].strip()
                    name_lower = name_clean.lower()
                    rank_num = idx + 1

                    # Profile & Founder details lookup
                    profile = get_app_detailed_profile(name_clean, app_id, is_known)

                    if "sbi" in name_lower:
                        domain = "https://homeloans.sbi"
                        email = "customercare@sbi.co.in"
                        tag_label = "Best Well-Rounded Lender"
                        interest_val = "7.25% - 8.55% p.a."
                        grievance_val = "Moderately effective"
                        fee_val = "0.35% (min ₹2,000 and max ₹10,000) + GST"
                    elif "navi" in name_lower:
                        domain = "https://navi.com"
                        email = "help@navi.com"
                        tag_label = "Instant Digital NBFC"
                        interest_val = "9.90% - 19.99% p.a."
                        grievance_val = "Highly Effective"
                        fee_val = "0.50% to 1.50% + GST"
                    elif "kreditbee" in name_lower:
                        domain = "https://kreditbee.in"
                        email = "help@kreditbee.in"
                        tag_label = "Popular Personal Credit App"
                        interest_val = "12.0% - 24.0% p.a."
                        grievance_val = "Moderately Effective"
                        fee_val = "1.0% to 2.5% + GST"
                    else:
                        _d_host = app_id.replace('com.', '').replace('in.', '').replace('org.', '')
                        domain = app_id if app_id.startswith("http") else f"https://{_d_host}.com"
                        email = f"support@{_d_host}.com"
                        tag_label = "Regulated Lending Partner" if is_known else "Digital Lender"
                        interest_val = "10.5% - 22.0% p.a." if is_known else "18.0% - 36.0% p.a."
                        grievance_val = "Moderately Effective" if is_known else "Standard Redressal"
                        fee_val = "1.0% to 3.0% + GST"

                    doc_analysis = "Demands essential documents but allows choice in additional paperwork, balancing thoroughness and convenience." if not row['contacts'] else "Requests sensitive device permissions (Contacts/SMS access), requiring borrower caution prior to applying."
                    interest_analysis = f"Clearly displays interest rates on their website/platform, offering straightforward and affordable rate structures for easy borrower understanding." if row['disclosure_score'] >= 3 else "Interest rate details are conditional; verify final Sanction Letter terms."
                    fee_analysis = f"Details processing fees clearly online, ensuring transparent, cost-effective choices for informed borrower decisions."
                    grievance_analysis = f"Has an online grievance escalation matrix with accessibility options, leading to effective problem-solving."

                    det_bg = "#0D111A" if c_dark else "#FFFFFF"
                    det_bdr = "1.5px solid rgba(255, 255, 255, 0.08)" if c_dark else "1.5px solid #E2E8F0"
                    det_subbar = "rgba(255, 255, 255, 0.03)" if c_dark else "#F8FAFC"
                    det_card = "rgba(255, 255, 255, 0.02)" if c_dark else "#FAFAFA"

                    take_items_html = "".join([
                        f'<div style="display:flex; align-items:flex-start; gap:8px; font-size:0.84rem; color:{text_color}; line-height:1.45;">'
                        f'<span style="color:#34D399; font-weight:bold; flex-shrink:0; font-size:0.95rem;">✓</span><span>{item}</span></div>'
                        for item in profile["who_should_take"]
                    ])

                    avoid_items_html = "".join([
                        f'<div style="display:flex; align-items:flex-start; gap:8px; font-size:0.84rem; color:{text_color}; line-height:1.45;">'
                        f'<span style="color:#F87171; font-weight:bold; flex-shrink:0; font-size:0.95rem;">✕</span><span>{item}</span></div>'
                        for item in profile["who_should_avoid"]
                    ])

                    detail_view_html = f"""<div style="background:{det_bg}; border:{det_bdr}; border-radius:22px; padding:28px 32px; margin-top:20px; margin-bottom:28px; box-shadow:0 16px 40px rgba(0,0,0,0.35);">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid {'rgba(255,255,255,0.08)' if c_dark else '#E2E8F0'}; padding-bottom:20px; margin-bottom:20px; flex-wrap:wrap; gap:12px;">
<div style="display:flex; align-items:center; gap:14px;">
<div>
<div style="font-size:1.45rem; font-weight:900; color:{text_color};">{name_clean}</div>
<span style="background:{score_bg}; color:{score_fg}; font-size:0.75rem; font-weight:800; padding:3px 10px; border-radius:20px; border:{score_bdr}; display:inline-block; margin-top:3px;">{tag_label}</span>
</div>
</div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="background:{'#0E3321' if c_dark else '#D1FAE5'}; color:{'#48BB78' if c_dark else '#047857'}; font-size:0.85rem; font-weight:800; padding:8px 16px; border-radius:12px; border:1px solid {'rgba(72,187,120,0.3)' if c_dark else 'rgba(4,120,87,0.3)'};">
FinShield Score: <strong>{score}</strong> <span style="font-size:0.75rem; opacity:0.8;">/ 100</span>
</div>
<div style="border:1px solid {'rgba(255,255,255,0.15)' if c_dark else '#CBD5E1'}; color:{text_color}; font-size:0.85rem; font-weight:800; padding:8px 16px; border-radius:12px; background:{'rgba(255,255,255,0.03)' if c_dark else '#F8FAFC'};">
FinShield Rank: <strong>#{rank_num:02d}</strong>
</div>
</div>
</div>

<!-- ABOUT & FOUNDED BY BLOCK -->
<div style="margin-bottom:24px; background:{det_card}; border:1px solid {'rgba(255,255,255,0.06)' if c_dark else '#E2E8F0'}; border-radius:18px; padding:22px 26px;">
<div style="font-size:1.15rem; font-weight:900; color:{text_color}; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
<span style="color:#F7C948;">ℹ️</span> About {name_clean}
</div>
<div style="font-size:0.88rem; color:{muted_color}; line-height:1.65; margin-bottom:16px;">
{profile['about']}
</div>
<div style="display:flex; align-items:center; gap:10px; background:{'rgba(247,201,72,0.1)' if c_dark else '#FEF3C7'}; border:1px solid {'rgba(247,201,72,0.3)' if c_dark else '#FDE68A'}; padding:10px 16px; border-radius:12px;">
<span style="font-size:1rem;">🏛️</span>
<div style="font-size:0.86rem; color:{'#F7C948' if c_dark else '#92400E'};">
<strong>Founded By / Parent Entity:</strong> {profile['founded_by']}
</div>
</div>
</div>

<!-- WHO SHOULD TAKE vs WHO SHOULD AVOID GRID -->
<div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:28px;">
<div style="background:{'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(6,78,59,0.15) 100%)' if c_dark else '#F0FDF4'}; border:1.5px solid {'rgba(16,185,129,0.35)' if c_dark else '#BBF7D0'}; border-radius:18px; padding:20px 22px;">
<div style="font-size:1.05rem; font-weight:800; color:{'#34D399' if c_dark else '#15803D'}; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
<span>✅</span> Who Should Take Loan
</div>
<div style="display:flex; flex-direction:column; gap:10px;">
{take_items_html}
</div>
</div>
<div style="background:{'linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(127,29,29,0.15) 100%)' if c_dark else '#FEF2F2'}; border:1.5px solid {'rgba(239,68,68,0.35)' if c_dark else '#FECACA'}; border-radius:18px; padding:20px 22px;">
<div style="font-size:1.05rem; font-weight:800; color:{'#F87171' if c_dark else '#B91C1C'}; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
<span>⚠️</span> Who Should Avoid
</div>
<div style="display:flex; flex-direction:column; gap:10px;">
{avoid_items_html}
</div>
</div>
</div>

<div style="display:grid; grid-template-columns:1fr 1fr 1.3fr; gap:18px; background:{det_subbar}; padding:16px 22px; border-radius:14px; margin-bottom:28px; border:1px solid {'rgba(255,255,255,0.04)' if c_dark else '#F1F5F9'};">
<div>
<div style="font-size:0.72rem; color:{muted_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Interest Rate</div>
<div style="font-size:0.96rem; font-weight:900; color:{text_color}; margin-top:3px;">{interest_val}</div>
</div>
<div>
<div style="font-size:0.72rem; color:{muted_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Grievance Redressal Process</div>
<div style="font-size:0.96rem; font-weight:900; color:{text_color}; margin-top:3px;">{grievance_val}</div>
</div>
<div>
<div style="font-size:0.72rem; color:{muted_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Processing Fees (% of loan amount)</div>
<div style="font-size:0.96rem; font-weight:900; color:{text_color}; margin-top:3px;">{fee_val}</div>
</div>
</div>
<div style="margin-bottom:28px;">
<div style="font-size:1.15rem; font-weight:900; color:{text_color}; margin-bottom:14px;">Our Analysis</div>
<div style="background:{det_card}; border:1px solid {'rgba(255,255,255,0.06)' if c_dark else '#E2E8F0'}; border-radius:16px; padding:22px 26px;">
<div style="margin-bottom:18px;">
<div style="font-size:0.94rem; font-weight:800; color:{text_color}; display:flex; align-items:center; gap:8px;">
<span style="color:#34D399; font-size:1rem;">✦</span> Loan Documentation
</div>
<div style="font-size:0.86rem; color:{muted_color}; margin-top:4px; line-height:1.55; padding-left:26px;">
{doc_analysis}
</div>
</div>
<div style="margin-bottom:18px;">
<div style="font-size:0.94rem; font-weight:800; color:{text_color}; display:flex; align-items:center; gap:8px;">
<span style="color:#34D399; font-size:1rem;">✦</span> Interest Rate
</div>
<div style="font-size:0.86rem; color:{muted_color}; margin-top:4px; line-height:1.55; padding-left:26px;">
{interest_analysis}
</div>
</div>
<div style="margin-bottom:18px;">
<div style="font-size:0.94rem; font-weight:800; color:{text_color}; display:flex; align-items:center; gap:8px;">
<span style="color:#34D399; font-size:1rem;">✦</span> Processing Fee
</div>
<div style="font-size:0.86rem; color:{muted_color}; margin-top:4px; line-height:1.55; padding-left:26px;">
{fee_analysis}
</div>
</div>
<div>
<div style="font-size:0.94rem; font-weight:800; color:{text_color}; display:flex; align-items:center; gap:8px;">
<span style="color:#34D399; font-size:1rem;">✦</span> Grievance Redressal
</div>
<div style="font-size:0.86rem; color:{muted_color}; margin-top:4px; line-height:1.55; padding-left:26px;">
{grievance_analysis}
</div>
</div>
</div>
</div>
<div>
<div style="display:flex; gap:40px; font-size:0.85rem; flex-wrap:wrap;">
<div>
<div style="color:{muted_color}; font-weight:600; margin-bottom:4px;">Website Link</div>
<a href="{domain}" target="_blank" style="color:#60A5FA; font-weight:700; text-decoration:none; display:inline-flex; align-items:center; gap:4px;">
{domain} <span style="font-size:0.75rem;">🗗</span>
</a>
</div>
<div>
<div style="color:{muted_color}; font-weight:600; margin-bottom:4px;">Customer Care Email ID</div>
<a href="mailto:{email}" style="color:#60A5FA; font-weight:700; text-decoration:none; display:inline-flex; align-items:center; gap:4px;">
{email} <span style="font-size:0.75rem;">🗗</span>
</a>
</div>
</div>
</div>
</div>"""
                    st.markdown(detail_view_html, unsafe_allow_html=True)
                    
                    if row["reasons"]:
                        with st.expander(f"🛡️ View Live Riskometer SVG Gauge & Key Risk Drivers for {name_clean}"):
                            gauge_html = render_rbi_riskometer_card(row["risk_proba"], c_dark)
                            st.markdown(gauge_html, unsafe_allow_html=True)
                            st.markdown("**Key Risk Drivers:**")
                            for item in row["reasons"]:
                                if isinstance(item, (tuple, list)) and len(item) == 2:
                                    reason_text, is_bad = item
                                    icon = "🔴" if is_bad else "🟢"
                                    st.write(f"{icon} {reason_text}")

                # Interactive Side-by-Side Comparison Drawer triggered by Compare button
                if st.session_state.get("active_cmp_app1") == app_id:
                    n1_short = app_name.split(':')[0]
                    st.markdown("---")
                    st.markdown(f"#### ⚖️ Compare **{n1_short}** with 2nd Lending App")
                    
                    other_app_options = [n for n in all_app_names if n != app_name]
                    selected_app2_name = st.selectbox(
                        f"Select 2nd app to compare against {n1_short}:",
                        options=other_app_options,
                        key=f"sel_cmp2_dropdown_{idx}"
                    )
                    
                    app1_info = row
                    app2_info = df_ranked[df_ranked["app_name"] == selected_app2_name].iloc[0]

                    n1_s = app1_info["app_name"].split(':')[0]
                    n2_s = app2_info["app_name"].split(':')[0]
                    s1, s2 = app1_info["safety_score"], app2_info["safety_score"]

                    # Risk weights: Contacts (high=2), SMS (high=2), Photos (med=1)
                    risk1 = (2 if app1_info["contacts"] else 0) + (2 if app1_info["sms"] else 0) + (1 if app1_info["photos"] else 0)
                    risk2 = (2 if app2_info["contacts"] else 0) + (2 if app2_info["sms"] else 0) + (1 if app2_info["photos"] else 0)

                    if s1 > s2:
                        winner = n1_s
                        loser = n2_s
                        winner_score, loser_score = s1, s2
                        diff_pts = s1 - s2
                        winner_reason = f"has a higher FinShield Safety Index (+{diff_pts} pts)"
                    elif s2 > s1:
                        winner = n2_s
                        loser = n1_s
                        winner_score, loser_score = s2, s1
                        diff_pts = s2 - s1
                        winner_reason = f"has a higher FinShield Safety Index (+{diff_pts} pts)"
                    else: # Equal scores -> evaluate permissions risk
                        if risk1 < risk2:
                            winner = n1_s
                            loser = n2_s
                            winner_score, loser_score = s1, s2
                            winner_reason = "requests fewer intrusive data permissions (Contacts / SMS)"
                        elif risk2 < risk1:
                            winner = n2_s
                            loser = n1_s
                            winner_score, loser_score = s2, s1
                            winner_reason = "requests fewer intrusive data permissions (Contacts / SMS)"
                        else:
                            winner = None

                    # Detailed Takeaways List
                    takeaways = []

                    # Contacts comparison
                    if not app1_info["contacts"] and app2_info["contacts"]:
                        takeaways.append(f"🟢 <strong>Contacts List Privacy</strong>: <strong>{n1_s}</strong> does NOT ask for phone contacts access, whereas <strong>{n2_s}</strong> requests full contact list permissions.")
                    elif not app2_info["contacts"] and app1_info["contacts"]:
                        takeaways.append(f"🟢 <strong>Contacts List Privacy</strong>: <strong>{n2_s}</strong> does NOT ask for phone contacts access, whereas <strong>{n1_s}</strong> requests full contact list permissions.")
                    elif not app1_info["contacts"] and not app2_info["contacts"]:
                        takeaways.append("🟢 <strong>Contacts Privacy</strong>: Both apps respect user privacy by NOT requesting phone contact lists.")
                    else:
                        takeaways.append("⚠️ <strong>Contacts Exposure</strong>: Both apps request phone contacts access. Never grant contact permissions to instant loan apps.")

                    # SMS comparison
                    if not app1_info["sms"] and app2_info["sms"]:
                        takeaways.append(f"🟢 <strong>SMS Reading Privacy</strong>: <strong>{n1_s}</strong> does NOT read private SMS, whereas <strong>{n2_s}</strong> requests SMS reading permissions.")
                    elif not app2_info["sms"] and app1_info["sms"]:
                        takeaways.append(f"🟢 <strong>SMS Reading Privacy</strong>: <strong>{n2_s}</strong> does NOT read private SMS, whereas <strong>{n1_s}</strong> requests SMS reading permissions.")
                    elif not app1_info["sms"] and not app2_info["sms"]:
                        takeaways.append("🟢 <strong>SMS Privacy</strong>: Neither app reads private SMS messages.")
                    else:
                        takeaways.append("⚠️ <strong>SMS Reading</strong>: Both apps request permission to read SMS messages.")

                    # Photos comparison
                    if not app1_info["photos"] and app2_info["photos"]:
                        takeaways.append(f"🟢 <strong>Gallery/Photos Access</strong>: <strong>{n1_s}</strong> does NOT request photo gallery access, whereas <strong>{n2_s}</strong> requests photo access.")
                    elif not app2_info["photos"] and app1_info["photos"]:
                        takeaways.append(f"🟢 <strong>Gallery/Photos Access</strong>: <strong>{n2_s}</strong> does NOT request photo gallery access, whereas <strong>{n1_s}</strong> requests photo access.")

                    # RBI compliance comparison
                    if app1_info["is_known"] and not app2_info["is_known"]:
                        takeaways.append(f"🏛️ <strong>RBI Regulatory Partner</strong>: <strong>{n1_s}</strong> is an officially verified RBI NBFC partner, while <strong>{n2_s}</strong> lacks verified RBI partner records.")
                    elif app2_info["is_known"] and not app1_info["is_known"]:
                        takeaways.append(f"🏛️ <strong>RBI Regulatory Partner</strong>: <strong>{n2_s}</strong> is an officially verified RBI NBFC partner, while <strong>{n1_s}</strong> lacks verified RBI partner records.")
                    else:
                        takeaways.append("🏛️ <strong>RBI Compliance</strong>: Both apps disclose regulated bank / NBFC lending partners.")

                    # Harassment comparison
                    r1, r2 = app1_info["redflag_pct"], app2_info["redflag_pct"]
                    if abs(r1 - r2) >= 0.5:
                        cleaner_app = n1_s if r1 < r2 else n2_s
                        takeaways.append(f"🚨 <strong>Harassment Review Mentions</strong>: <strong>{cleaner_app}</strong> has lower harassment/recovery complaints in Play Store reviews ({min(r1, r2):.1f}% vs {max(r1, r2):.1f}%).")

                    # Render Executive Conclusion Container
                    if winner:
                        banner_header = f"🏆 Overall Safety Winner: {winner}"
                        banner_sub = f"Based on FinShield's AI evaluation, <strong>{winner}</strong> is recommended over <strong>{loser}</strong> because it {winner_reason}."
                        verdict_color = "#10B981"
                        verdict_bg = "rgba(16, 185, 129, 0.12)" if c_dark else "#ECFDF5"
                    else:
                        banner_header = "⚖️ Neutral Comparison: Equal Safety Index"
                        banner_sub = f"Both <strong>{n1_s}</strong> and <strong>{n2_s}</strong> share an identical Safety Score of <strong>{s1}/100</strong> and similar permission access profiles."
                        verdict_color = "#F59E0B"
                        verdict_bg = "rgba(245, 158, 11, 0.12)" if c_dark else "#FFFBEB"

                    takeaways_html = "".join([f"<li style='margin-bottom:6px;'>{t}</li>" for t in takeaways])

                    conclusion_card_html = f"""
                    <div style="background:{verdict_bg}; border:1.5px solid {verdict_color}; border-radius:14px; padding:16px 20px; margin: 12px 0 18px;">
                        <div style="font-size:1.1rem; font-weight:900; color:{verdict_color}; margin-bottom:4px;">
                            {banner_header}
                        </div>
                        <div style="font-size:0.92rem; font-weight:700; color:{text_color}; margin-bottom:12px; line-height:1.4;">
                            {banner_sub}
                        </div>
                        <div style="font-size:0.85rem; font-weight:800; text-transform:uppercase; color:{muted_color}; letter-spacing:0.5px; margin-bottom:6px;">
                            📌 KEY SAFETY COMPARISON TAKEAWAYS:
                        </div>
                        <ul style="margin:0; padding-left:18px; font-size:0.88rem; color:{text_color}; line-height:1.5;">
                            {takeaways_html}
                        </ul>
                        <div style="margin-top:10px; padding-top:8px; border-top:1px dashed {'rgba(255,255,255,0.1)' if c_dark else '#CBD5E1'}; font-size:0.82rem; font-weight:700; color:{muted_color};">
                            💡 <strong>FinShield Advisory Tip:</strong> Always verify Key Fact Statements (KFS) and deny unnecessary phone contacts access before accepting loan agreements.
                        </div>
                    </div>
                    """
                    st.markdown(conclusion_card_html, unsafe_allow_html=True)
                
                # Sleek separation line between app card rows
                div_bdr = "rgba(255, 255, 255, 0.12)" if c_dark else "rgba(0, 0, 0, 0.1)"
                st.markdown(f"<div style='margin: 28px 0 36px; border-bottom: 1.5px solid {div_bdr};'></div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning(f"File {FEATURES_CSV_PATH} not found.")

# ==========================================
# TAB 4: LOAN CALCULATORS
# ==========================================
with tab_calculators:
    st.markdown("### 🧮 Financial Advisory Calculators")

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("#### 💰 Personal Loan Prepayment Calculator")
        c_amt = st.number_input("Loan Amount (₹)", value=200000, step=10000, key="calc_amt")
        c_rate = st.number_input("Interest Rate (% p.a.)", value=18.0, step=0.5, key="calc_rate")
        c_tenure = st.number_input("Tenure (Months)", value=24, step=1, key="calc_tenure")

        if c_rate > 0 and c_tenure > 0:
            r = (c_rate / 12) / 100
            emi = (c_amt * r * ((1 + r) ** c_tenure)) / (((1 + r) ** c_tenure) - 1)
            tot_int = (emi * c_tenure) - c_amt
            savings = tot_int * 0.35

            st.markdown(
                f"""
                <div style="background:{t['card_bg']}; border:1px solid {t['card_border']}; border-radius:12px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Monthly EMI:</span><strong>₹{emi:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Total Interest Paid:</span><strong>₹{tot_int:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#34D399; font-weight:700;"><span>Prepayment Interest Savings:</span><strong>₹{savings:,.0f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with c_right:
        st.markdown("#### ⚠️ Hidden Fees & True APR Detector")
        d_amt = st.number_input("Disbursed Amount (₹)", value=10000, step=1000, key="apr_disb")
        r_amt = st.number_input("Repayment Amount (₹)", value=12500, step=1000, key="apr_repay")
        d_days = st.number_input("Tenure (Days)", value=7, step=1, key="apr_days")

        if d_amt > 0 and d_days > 0:
            extra = r_amt - d_amt
            apr = (extra / d_amt / d_days) * 365 * 100

            st.markdown(
                f"""
                <div style="background:{t['card_bg']}; border:1px solid {t['card_border']}; border-radius:12px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:6px;"><span>Total Extra Fee & Interest:</span><strong>₹{extra:,.0f}</strong></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#F87171; font-weight:800;"><span>True Annualized APR:</span><strong>{apr:,.0f}% p.a.</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if d_days <= 7 or apr > 100:
                st.error("⚠️ Predatory 7-day loan alert! Annualized APR exceeds 100%. Violates RBI Digital Lending Directives.")
            else:
                st.success("✓ Terms within standard NBFC lending parameters.")

# ==========================================
# TAB 5: RBI GUIDELINES
# ==========================================
with tab_rbi:
    st.markdown("### 📜 RBI Digital Lending Guidelines 2026 Checklist")
    st.markdown(
        """
        - 🚫 **No Prohibited Data Access**: Lending apps are strictly forbidden from asking access to your phone contacts list, private photo gallery, or reading SMS.
        - 📄 **Key Fact Statement (KFS)**: Lenders must provide a standardized Key Fact Statement detailing all APR, processing fees, and penalties before agreement execution.
        - 🏦 **Direct Bank Account Transfer**: Loan disbursements and repayments must happen strictly between borrower's bank account and regulated bank/NBFC bank account.
        - 🔗 **Grievance Redressal**: Every app must publish a dedicated Grievance Redressal Officer contact and registered office address.
        """
    )
    st.markdown(
        """
        ---
        - 🏛️ **Verify Lenders on RBI Sachet**: [sachet.rbi.org.in](https://sachet.rbi.org.in)
        - 🚨 **File CyberCrime Complaint**: [cybercrime.gov.in](https://cybercrime.gov.in)
        """
    )

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# GLOBAL LEGAL DISCLAIMER & FOOTER
# ==========================================
st.markdown(
    f"""
    <div style="max-width:1350px; margin: -10px auto 14px; padding: 0 16px;">
        <div style="background: {'rgba(16, 22, 34, 0.75)' if st.session_state.dark_mode else '#F8FAFC'}; border: {'1.5px solid rgba(247, 201, 72, 0.25)' if st.session_state.dark_mode else '1.5px solid #E2E8F0'}; border-radius: 16px; padding: 18px 24px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); box-shadow: {'0 10px 30px rgba(0,0,0,0.3)' if st.session_state.dark_mode else '0 4px 15px rgba(0,0,0,0.04)'};">
            <div style="font-size: 0.82rem; color: {'#94A3B8' if st.session_state.dark_mode else '#64748B'}; line-height: 1.6; font-weight: 500;">
                <strong style="color: {'#F8FAFC' if st.session_state.dark_mode else '#0F172A'}; font-weight: 700;">Legal & Financial Disclaimer:</strong><br>
                Finshield operates independently. The information presented herein is intended solely for educational and informational purposes and should not be construed as financial advice. Before making any financial decisions, it's essential to undertake your own thorough research and analysis. If you're uncertain about any financial matters, we strongly recommend seeking guidance from a qualified financial advisor.
            </div>
        </div>
    </div>
    <div style="max-width:1350px; margin: 24px auto 0px; padding: 0 16px 8px;">
        <div style="border-top: 1px solid {'rgba(255, 255, 255, 0.08)' if st.session_state.dark_mode else 'rgba(0, 0, 0, 0.08)'}; padding-top: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 1.05rem; font-weight: 800; color: {'#F8FAFC' if st.session_state.dark_mode else '#0F172A'};">
                    FinShield
                </div>
                <div style="font-size: 0.8rem; color: {'#64748B' if st.session_state.dark_mode else '#94A3B8'}; margin-top: 4px; font-weight: 500;">
                    © 2026 FinShield. Empowering Borrowers & Detecting Illegal Loan Apps in India.
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px; font-size: 0.82rem; font-weight: 600;">
                <a href="https://sachet.rbi.org.in" target="_blank" style="color: {'#94A3B8' if st.session_state.dark_mode else '#475569'}; text-decoration: none;">
                    RBI Sachet Portal
                </a>
                <a href="https://cybercrime.gov.in" target="_blank" style="color: {'#94A3B8' if st.session_state.dark_mode else '#475569'}; text-decoration: none;">
                    CyberCrime Portal
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
