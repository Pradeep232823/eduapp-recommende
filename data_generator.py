import pandas as pd
import random
import re
from datetime import datetime, timedelta
from google_play_scraper import app, reviews

# -------------------------------
# CONFIGURATION
# -------------------------------
INPUT_FILE = "real_apps_list.csv"
TOTAL_USERS = 500
MIN_SESSIONS_PER_USER = 4
MAX_SESSIONS_PER_USER = 8
TARGET_TOTAL_RECORDS = 2000
MAX_REVIEWS_PER_APP = 200

random.seed(42)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def clean_html(text):
    return re.sub(r'<[^>]*>', '', text)

def is_english(text):
    return text.isascii() and len(text.strip()) > 10

import random

def derive_rating_from_usage(usage_minutes):
    if usage_minutes <= 25:
        return round(random.uniform(1.0, 1.9), 1)
    elif usage_minutes <= 100:
        return round(random.uniform(2.1, 2.9), 1)
    elif usage_minutes <= 200:
        return round(random.uniform(3.1, 3.9), 1)
    elif usage_minutes <= 300:
        return round(random.uniform(4.1, 4.9), 1)
    else:
        return 5.0

def generate_usage_over_time(session_index):
    """
    User behavior evolves:
    earlier sessions → low usage
    later sessions → higher or varied usage
    """
    base = random.randint(10, 60)
    growth = session_index * random.randint(20, 60)
    noise = random.randint(-10, 20)
    return max(5, base + growth + noise)

# -------------------------------
# LOAD INPUT APPS
# -------------------------------
df_input = pd.read_csv(INPUT_FILE)

apps_data = []
ratings_data = []
app_id_map = {}
app_id_counter = 1

print("\n📦 Fetching real app metadata...")

# -------------------------------
# GENERATE apps.csv
# -------------------------------
for _, row in df_input.iterrows():
    try:
        package_name = row["App_Link"].split("id=")[-1].strip()
        app_info = app(package_name, lang="en", country="in")

        apps_data.append({
            "app_id": app_id_counter,
            "app_name": app_info["title"],
            "category": row["Category"],
            "description": clean_html(app_info["description"])[:500],
            "avg_rating": round(app_info["score"], 1)
        })

        app_id_map[app_id_counter] = {
            "package": package_name,
            "reviews": []
        }

        print(f"✔ {app_info['title']}")
        app_id_counter += 1

    except Exception as e:
        print(f"❌ Skipped {row.get('App_name', 'Unknown')} → {e}")

# -------------------------------
# FETCH ENGLISH REVIEWS
# -------------------------------
print("\n📥 Fetching real user reviews...")

for app_id, info in app_id_map.items():
    try:
        review_list, _ = reviews(
            info["package"],
            lang="en",
            country="in",
            count=MAX_REVIEWS_PER_APP
        )
        info["reviews"] = [
            r for r in review_list if is_english(r["content"])
        ]
    except:
        info["reviews"] = []

# -------------------------------
# GENERATE TIME-AWARE USER DATA
# -------------------------------
print("\n🧠 Generating time-aware user behavior...")

all_app_ids = list(app_id_map.keys())
record_count = 0
start_time = datetime.now() - timedelta(days=45)

for user_id in range(1, TOTAL_USERS + 1):
    num_sessions = random.randint(
        MIN_SESSIONS_PER_USER,
        MAX_SESSIONS_PER_USER
    )

    session_time = start_time + timedelta(
        days=random.randint(0, 5)
    )

    used_apps = random.sample(
        all_app_ids,
        min(num_sessions, len(all_app_ids))
    )

    for session_index, app_id in enumerate(used_apps):
        if record_count >= TARGET_TOTAL_RECORDS:
            break

        usage_minutes = generate_usage_over_time(session_index)
        rating = derive_rating_from_usage(usage_minutes)

        reviews_pool = app_id_map[app_id]["reviews"]
        review_text = (
            random.choice(reviews_pool)["content"]
            if reviews_pool else
            "Useful educational app."
        )

        ratings_data.append({
            "user_id": user_id,
            "app_id": app_id,
            "usage_minutes": usage_minutes,
            "rating": rating,
            "timestamp": session_time,
            "review": review_text
        })

        session_time += timedelta(
            days=random.randint(1, 3),
            hours=random.randint(1, 5)
        )

        record_count += 1

    if record_count >= TARGET_TOTAL_RECORDS:
        break

# -------------------------------
# SAVE OUTPUT FILES
# -------------------------------
df_apps = pd.DataFrame(apps_data)
df_ratings = pd.DataFrame(ratings_data)

df_apps.to_csv("apps.csv", index=False, encoding="utf-8-sig")
df_ratings.to_csv("ratings.csv", index=False, encoding="utf-8-sig")

print("\n✅ DATASET GENERATION COMPLETED")
print(f"✔ apps.csv saved with → {len(df_apps)} apps")
print(f"✔ ratings.csv saved with → {len(df_ratings)} records")
print(f"✔ unique users are → {df_ratings['user_id'].nunique()}")
