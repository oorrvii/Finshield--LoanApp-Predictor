import time
import pandas as pd
from google_play_scraper import Sort, app, reviews, search

# ==========================================
# 1. APP LISTS (Good vs Bad Apps)
# ==========================================

good_app_queries = [
    "Navi",
    "Bajaj Finserv",
    "KreditBee",
    "Moneyview",
    "Fibe EarlySalary",
    "CashE",
    "mPokket",
    "LendingKart",
    "Tata Capital Moneyfy",
    "Stashfin",
    "PaySense",
    "Flipkart Finance",
    "My Mudra",
    "DMI Finance",
    "True Balance",
    "Kissht",
    "LazyPay",
    "Piramal Finance",
    "Home Credit India",
    "Lendingplate",
    "IIFL Finance",
    "SMFG India Credit",
    "PayMe India",
    "LoanTap",
    "FlexSalary",
    "Branch Personal Loan App",
    "Lenditt",
    "IDFC FIRST Bank Personal Loan",
    "Axis Bank Insta Personal Loan",
    "Kotak Mahindra Bank InstaLoan",
    "HDFC Bank Personal Loan App",
    "ICICI Bank iMobile",
    "SBI YONO",
    "IndusInd Bank INDIE App",
    "Federal Bank FedMobile Fi Money",
    "Bank of Baroda bob World",
    "Punjab National Bank PNB One",
    "Canara Bank ai1 App",
    "GroMo",
    "Prefr",
    "Nest",
    "Freo MoneyTap",
    "Zype",
    "Finnable",
    "Olyv SmartCoin",
    "NIRA Finance",
    "Paisabazaar",
    "BankBazaar",
    "Cred",
    "Kotak 811",
    "Airtel Thanks Airtel Finance",
    "JioFinance",
    "Jupiter Money",
    "OneScore OneCard",
    "Google Pay",
    "PhonePe",
    "Buddy Loan",
    "IndiaLends",
    "super.money",
]

bad_app_queries = [
    # Real Flagged / Unauthorized Apps
    "Agile Loan App",
    "ApnaAroham",
    "Apna Paisa",
    "Asan Loan",
    "Bellono Loan",
    "CashFish",
    "Flip Cash",
    "FlyCash",
    "Kredipe",
    "LoanCube",
    "Rupee Master",
    "QuickCashNow",
    "InstaLoanFast",
    "EasyMoneyPro",
    "LoanBee Instant",
    "CashAdvance Go",
    "RupeePocket",
    "CashTap Instant",
    "LoanSure+",
    "PayDayNow",
    "SmartRupees",
    "GoldenCash Loan",
    "UltraLoanX",
    "CashTree Easy Loan",
    "SpeedLoan Pro",
    "DailyRupee Credit",
    "Easy Pocket Credit",
    "Speed Wallet Loan",
    "Loan Express 7",
    "Cash King India",
    "Fast Rupee Express",
    "Minute Credit App",
    "Super Cash Now",
    "InstanPaisa",
    "Cash Star Pro",
    "Daily Credit Go",
    "Magic Loan",
    "Cash Credit India",
    "Rupee Boss",
    "Fast Wallet Pro",
    "Insta Pocket",
    "Cash Advance X",
    "Loan Time",
    "Instant Credit Hub",
    "Easy Rupee Club",
    "Quick Wallet",
    "Cash Express India",
    "Rupee Link",
    "Flash Loan App",
    "Speed Credit Go",
    "Quick Loan Pro",
    "Pocket Credit Instant",
    "Cash Well India",
    "Rupee Zone",
    "Insta Money Club",
    "Smart Credit App",
    "Loan Master X",
    "Daily Cash Advance",
    "Cash Spark",
    "Rupee Tree",
    "Fast Loan Hub",
    "Instant Rupee Go",
    "Easy Credit Star",
    "Speed Cash Now",
    "Pocket Loan India",
    "Cash Hero",
    "Rupee Express Pro",
    "Flash Credit App",
    "Quick Cash Club",
    "Insta Wallet India",
    "Cash Hub Pro",
    "Rupee Advance",
    "Smart Loan Hub",
    "Daily Rupee Star",
    "Fast Credit India",
    "Instant Cash Tree",
    "Pocket Money Express",
    "Cash Link Pro",
    "Rupee King App",
    "Flash Loan India",
    "Speed Rupee",
    "Quick Credit Star",
    "Insta Loan Hub",
    "Cash Star Express",
    "Rupee Pocket Pro",
    "Smart Cash Club",
    "Daily Loan Go",
    "Fast Rupee Star",
    "Instant Credit India",
    "Pocket Cash Pro",
    "Cash Zone India",
    "Rupee Master Pro",
    "Flash Credit Go",
    "Speed Money Hub",
    "Quick Loan India",
    "Insta Cash Express",
    "Cash Tree India",
    "Rupee Credit Hub",
    "Smart Wallet Pro",
    "Daily Cash Hub",
    # Pattern/Synthetic Search Keywords
    "CashMatrix Fast",
    "RupeeVelocity",
    "QuickFund Express",
    "InstaLend Pro",
    "TurboCash India",
    "SpeedFund App",
    "EasyLend Hub",
    "FlashPaisa Go",
    "RapidCredit India",
    "MetroRupee",
    "PrimeCash Now",
    "FlexiRupee Express",
    "ZapLoan Pro",
    "VelocityCash",
    "PocketLend India",
    "SwiftPaisa",
    "ReadyCredit Now",
    "CashMagnet",
    "RupeeRocket",
    "HyperLoan Express",
    "UltraFund Pro",
    "InstanCredit Go",
    "ExpressPaisa",
    "CashPulse India",
    "RupeeDirect",
    "SpeedLend Hub",
    "EasyFund Pro",
    "FlashMoney Go",
    "RapidPaisa",
    "MetroCash India",
    "PrimeRupee",
    "FlexiCash Now",
    "ZapRupee",
    "VelocityLend",
    "PocketFund",
    "SwiftCash Pro",
    "ReadyRupee",
    "CashNexus",
    "RupeeOrbital",
    "HyperCash Hub",
    "UltraRupee",
    "InstanFund",
    "ExpressRupee",
    "CashWave India",
    "RupeeSwift",
    "SpeedMoney Go",
    "EasyRupee Pro",
    "FlashLend",
    "RapidRupee Now",
    "MetroCredit",
    "PrimeLend",
    "FlexiFund Express",
    "ZapCredit",
    "VelocityRupee",
    "PocketCash Go",
    "SwiftCredit",
    "ReadyLend India",
    "CashVertex",
    "RupeeSprint",
    "HyperRupee",
    "UltraCredit",
    "InstanPaisa Hub",
    "ExpressCredit Pro",
    "CashFlow Now",
    "RupeeZoom",
    "SpeedFund Go",
    "EasyMoney Hub",
    "FlashRupee",
    "RapidLend Pro",
    "MetroFund",
    "PrimeCredit Now",
    "FlexiLend",
    "ZapPaisa",
    "VelocityCredit",
    "PocketRupee Express",
    "SwiftFund",
    "ReadyCash Pro",
    "CashHorizon",
    "RupeeDash",
    "HyperFund India",
    "UltraPaisa",
    "InstanLend Go",
    "ExpressFund",
    "CashVibe",
    "RupeeBlitz",
    "SpeedCredit Pro",
    "EasyPaisa Hub",
    "FlashFund Now",
    "RapidCash Express",
    "MetroLend",
    "PrimePaisa",
    "FlexiCredit Hub",
    "ZapCash",
    "VelocityPaisa",
    "PocketLend Pro",
    "SwiftRupee Now",
    "ReadyMoney",
    "CashZenith",
    "RupeeBolt",
    "HyperLend Pro",
]

# ==========================================
# 2. SEARCH PLAY STORE APP IDs
# ==========================================

collected_apps = {}  # Format: {app_id: is_predatory_label}


def search_apps(query_list, is_predatory_label):
    for q in query_list:
        try:
            results = search(q, lang="en", country="in", n_hits=1)
            for item in results:
                app_id = item["appId"]
                if app_id not in collected_apps:
                    collected_apps[app_id] = is_predatory_label
                    print(f"Found ID for '{q}': {app_id}")
        except Exception as e:
            print(f"Search failed for '{q}': {e}")


print("--- Searching for Legitimate Apps ---")
search_apps(good_app_queries, is_predatory_label=0)

print("\n--- Searching for Predatory Apps ---")
search_apps(bad_app_queries, is_predatory_label=1)

print(f"\nTotal Unique Live Apps Found: {len(collected_apps)}")

# ==========================================
# 3. SCRAPE METADATA & REVIEWS
# ==========================================

raw_reviews_dataset = []  # For raw_reviews.csv (CSV #1)
app_metadata_dataset = []  # For app_raw_metadata.csv (CSV #2)

for app_id, is_predatory in collected_apps.items():
    try:
        # Fetch App Details
        info = app(app_id, lang="en", country="in")
        app_name = info.get("title")

        # Add to App Dataset
        app_metadata_dataset.append(
            {
                "app_id": app_id,
                "app_name": app_name,
                "developer": info.get("developer"),
                "description": info.get("description"),
                "privacy_policy": info.get("privacyPolicy"),
                "score": info.get("score"),
                "installs": info.get("installs"),
                "label": is_predatory,
            }
        )

        # Fetch Top 50 Reviews
        revs, _ = reviews(
            app_id, lang="en", country="in", sort=Sort.MOST_RELEVANT, count=50
        )

        for r in revs:
            if r.get("content"):
                raw_reviews_dataset.append(
                    {
                        "app_id": app_id,
                        "app_name": app_name,
                        "review_text": r.get("content"),
                        "rating": r.get("score"),
                        "review_date": r.get("at"),
                    }
                )

        print(f"✓ Scraped: {app_name}")
        time.sleep(0.3)

    except Exception as e:
        print(f"✗ Failed to scrape {app_id}: {e}")

# ==========================================
# 4. SAVE TO CSV
# ==========================================

df_reviews = pd.DataFrame(raw_reviews_dataset)
df_reviews.to_csv("raw_reviews.csv", index=False)

df_apps = pd.DataFrame(app_metadata_dataset)
df_apps.to_csv("app_raw_metadata.csv", index=False)

print("\n==========================================")
print(f"Saved {len(df_reviews)} reviews to 'raw_reviews.csv'")
print(f"Saved {len(df_apps)} apps to 'app_raw_metadata.csv'")
print("==========================================")
