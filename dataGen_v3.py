"""
dataGen_v3.py

Generates the synthetic Comcast recommender demo dataset (v3):
- users_v3.csv
- items_v3.csv
- interactions_v3.csv

Key design choices:
- All users are Comcast Internet customers (has_internet=1)
- Only a small percent of existing customers have Comcast Mobile (has_mobile=1)
- New users (is_new_customer=1) have NO interactions (cold start demo)
- Storm-Ready (Pro Extender) correlates with outage risk + WFH dependence + budget
- Internet-only users with multiple mobile lines + high mobile bill are more likely to
  interact with "Switch to Xfinity Mobile (Savings Offer)" / "Internet+Mobile Bundle".
  (This teaches the model the upsell pattern without hard-coded rules at inference time.)

Run:
  python dataGen_v3.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------
# Items (catalog)
# ----------------------------
def make_items_v3() -> pd.DataFrame:
    rows = []
    item_id = 0

    # Internet tiers
    for name, price, speed in [
        ("Internet 300", 60, 300),
        ("Internet 500", 75, 500),
        ("Internet 1000", 95, 1000),
        ("Internet 2000", 120, 2000),
    ]:
        rows.append((item_id, name, "internet_tier", price, float(speed), np.nan))
        item_id += 1

    # Mobile plans (only available to internet customers; included as items)
    for name, price in [
        ("Mobile By-the-Gig", 25),
        ("Mobile Unlimited", 45),
        ("Mobile Unlimited+", 60),
    ]:
        rows.append((item_id, name, "mobile_plan", price, np.nan, np.nan))
        item_id += 1

    # Add-ons
    for name, price, notes in [
        ("xFi Complete", 25, np.nan),
        ("WiFi Extender", 10, np.nan),
        ("xFi Pro Extender (Storm-Ready)", 20, "Battery backup for power outage connectivity"),
        ("Device Protection", 15, np.nan),
        ("International Calling", 10, np.nan),
        ("Extra Mobile Line", 30, np.nan),
    ]:
        rows.append((item_id, name, "addon", price, np.nan, notes))
        item_id += 1

    # Bundles + Offer
    for name, price, cat in [
        ("Internet+Mobile Bundle", 15, "bundle"),
        ("Premium Reliability Bundle", 25, "bundle"),
        ("Switch to Xfinity Mobile (Savings Offer)", 0, "offer"),
    ]:
        rows.append((item_id, name, cat, price, np.nan, np.nan))
        item_id += 1

    return pd.DataFrame(rows, columns=["item_id", "item_name", "category", "price", "speed_mbps", "notes"])


# ----------------------------
# Users (profiles)
# ----------------------------
def make_users_v3(
    rng: np.random.Generator,
    n_existing: int = 800,
    n_new: int = 200,
    pct_existing_with_mobile: float = 0.12,
) -> pd.DataFrame:
    n_total = n_existing + n_new
    user_id = np.arange(n_total)

    regions = np.array(["NE", "SE", "MW", "W"])
    region = rng.choice(regions, size=n_total, p=[0.35, 0.20, 0.25, 0.20])

    household_size = rng.integers(1, 7, size=n_total)

    # Counts within household
    wfh_count = np.array([rng.integers(0, hs + 1) for hs in household_size])
    gamer_count = np.array([rng.integers(0, hs + 1) for hs in household_size])
    creator_count = np.array([rng.integers(0, min(3, hs) + 1) for hs in household_size])

    # Device counts
    devices = np.clip(
        household_size * rng.integers(2, 5, size=n_total) + rng.integers(0, 4, size=n_total),
        2,
        18,
    )
    iot_devices = np.clip(
        (household_size - 1) * rng.integers(2, 8, size=n_total) + rng.integers(0, 6, size=n_total),
        0,
        60,
    )

    # Outage risk (demo assumption by region): SE higher, MW/NE medium, W lower
    base_risk = np.where(region == "SE", 2, np.where(region == "W", 0, 1))
    outage_risk = np.clip(base_risk + rng.integers(-1, 2, size=n_total), 0, 2)

    # Budget correlated with household size and region
    region_adj = np.where(region == "NE", 15, np.where(region == "W", 10, np.where(region == "MW", 5, 0)))
    budget = np.clip(40 + household_size * 15 + region_adj + rng.normal(0, 18, size=n_total), 40, 220).astype(int)

    # Broadband monthly data usage (GB)
    base = devices * rng.normal(55, 12, size=n_total)
    base += wfh_count * rng.normal(180, 50, size=n_total)
    base += gamer_count * rng.normal(220, 65, size=n_total)
    base += creator_count * rng.normal(350, 90, size=n_total)
    base += iot_devices * rng.normal(2.0, 0.8, size=n_total)
    base += rng.normal(0, 80, size=n_total)
    monthly_data_gb = np.clip(base, 50, 2500).astype(int)

    # Mobile context (even if they don't have Comcast Mobile)
    mobile_line_count = np.clip((household_size - rng.integers(0, 2, size=n_total)), 0, household_size)

    mobile_data = mobile_line_count * rng.normal(8, 3, size=n_total)
    mobile_data += gamer_count * rng.normal(6, 2, size=n_total)
    mobile_data += creator_count * rng.normal(10, 3, size=n_total)
    mobile_data += rng.normal(0, 3, size=n_total)
    mobile_data_gb = np.clip(mobile_data, 0, 200).round(0).astype(int)

    # Current (non-Comcast) mobile bill estimate (demo)
    current_mobile_bill = 25 * mobile_line_count + 0.4 * mobile_data_gb + rng.normal(0, 15, size=n_total)
    current_mobile_bill = np.clip(current_mobile_bill, 0, 320).round(0).astype(int)

    # Flags
    is_new_customer = np.zeros(n_total, dtype=int)
    is_new_customer[-n_new:] = 1

    # Comcast Internet: available for all in this synthetic demo
    has_internet = np.ones(n_total, dtype=int)

    # Comcast Mobile: small percent of existing customers
    has_mobile = np.zeros(n_total, dtype=int)
    existing_idx = np.arange(n_existing)
    has_mobile_existing = rng.random(n_existing) < pct_existing_with_mobile
    has_mobile[existing_idx] = has_mobile_existing.astype(int)
    # New customers default to 0 mobile in this demo

    return pd.DataFrame(
        {
            "user_id": user_id,
            "is_new_customer": is_new_customer,
            "has_internet": has_internet,
            "has_mobile": has_mobile,
            "region": region,
            "outage_risk": outage_risk,
            "household_size": household_size,
            "devices": devices,
            "iot_devices": iot_devices,
            "wfh_count": wfh_count,
            "gamer_count": gamer_count,
            "creator_count": creator_count,
            "budget": budget,
            "monthly_data_gb": monthly_data_gb,
            "mobile_line_count": mobile_line_count,
            "mobile_data_gb": mobile_data_gb,
            "current_mobile_bill": current_mobile_bill,
        }
    )


# ----------------------------
# Interaction logic
# ----------------------------
def choose_internet_tier(user_row: dict) -> str:
    score = 0
    score += user_row["monthly_data_gb"] >= 350
    score += user_row["monthly_data_gb"] >= 750
    score += user_row["monthly_data_gb"] >= 1200
    score += user_row["devices"] >= 8
    score += user_row["devices"] >= 12
    score += user_row["creator_count"] >= 1

    tier_idx = int(np.clip(score // 2, 0, 3))

    # budget nudges down if tight
    if user_row["budget"] < 70 and tier_idx > 0:
        tier_idx -= 1
    if user_row["budget"] < 60 and tier_idx > 0:
        tier_idx -= 1

    return ["Internet 300", "Internet 500", "Internet 1000", "Internet 2000"][tier_idx]


def choose_mobile_plan(user_row: dict) -> str | None:
    if user_row["mobile_line_count"] <= 0:
        return None
    per_line = user_row["mobile_data_gb"] / max(user_row["mobile_line_count"], 1)

    if (user_row["mobile_data_gb"] >= 60 or per_line >= 15) and user_row["budget"] >= 90:
        return "Mobile Unlimited+"
    if user_row["mobile_data_gb"] >= 25 or per_line >= 8:
        return "Mobile Unlimited"
    return "Mobile By-the-Gig"


def est_xm_cost(user_row: dict, plan_name: str) -> int:
    """
    Demo-only simplified pricing (not real pricing).
    Only depends on number of lines.
    Plan names match item catalog.
    """
    lines = max(int(user_row["mobile_line_count"]), 1)

    if plan_name == "Mobile Unlimited":
        return 45 * lines
    if plan_name == "Mobile Unlimited+":
        return 60 * lines
    if plan_name == "Mobile By-the-Gig":
        return 30 * lines

    return 0


def best_plan_and_savings(user_row: dict) -> tuple[str, int, int]:
    bill = int(user_row["current_mobile_bill"])
    plans = ["Mobile Unlimited", "Mobile Unlimited+", "Mobile By-the-Gig"]

    costs = {p: est_xm_cost(user_row, p) for p in plans}
    best_plan = min(costs, key=costs.get)
    best_cost = costs[best_plan]
    savings = bill - best_cost

    return best_plan, best_cost, savings


def generate_interactions_v3(
    rng: np.random.Generator,
    users: pd.DataFrame,
    items: pd.DataFrame,
    max_per_user: int = 9,
) -> pd.DataFrame:
    name_to_id = {r.item_name: int(r.item_id) for r in items.itertuples(index=False)}

    interactions = []
    existing = users[users["is_new_customer"] == 0]

    for r in existing.to_dict(orient="records"):
        uid = int(r["user_id"])

        # Internet tier always for existing users
        inet = choose_internet_tier(r)
        interactions.append((uid, name_to_id[inet], int(rng.integers(2, 6))))  # strength 2..5

        # Storm-Ready: outage risk + WFH dependence + budget (power outage connectivity)
        if (r["outage_risk"] >= 1) and (r["wfh_count"] >= 1) and (r["budget"] >= 85):
            p_sr = 0.15 + 0.10 * (r["outage_risk"] == 2) + 0.08 * (r["wfh_count"] >= 2)
            if rng.random() < min(p_sr, 0.45):
                interactions.append((uid, name_to_id["xFi Pro Extender (Storm-Ready)"], int(rng.integers(1, 4))))

        # WiFi Extender: coverage/concurrency need (devices + IoT + large household)
        if (r["devices"] >= 12) or (r["iot_devices"] >= 18) or (r["household_size"] >= 4):
            if rng.random() < 0.50:
                interactions.append((uid, name_to_id["WiFi Extender"], int(rng.integers(1, 4))))

        # xFi Complete: higher tiers / many devices / WFH
        if (inet in ["Internet 1000", "Internet 2000"]) or (r["devices"] >= 10) or (r["wfh_count"] >= 1):
            if rng.random() < 0.62:
                interactions.append((uid, name_to_id["xFi Complete"], int(rng.integers(1, 5))))

        # Mobile: only if has_mobile=1 (small percent)
        if int(r["has_mobile"]) == 1:
            plan = choose_mobile_plan(r)
            if plan:
                interactions.append((uid, name_to_id[plan], int(rng.integers(1, 5))))

                # Extra line signal if multi-line household
                if r["mobile_line_count"] >= 2 and rng.random() < 0.55:
                    interactions.append((uid, name_to_id["Extra Mobile Line"], int(rng.integers(1, 4))))

                # Device protection slightly more likely with mobile
                if rng.random() < 0.35:
                    interactions.append((uid, name_to_id["Device Protection"], int(rng.integers(1, 4))))

            # Some bundle interest
            if rng.random() < 0.35:
                interactions.append((uid, name_to_id["Internet+Mobile Bundle"], int(rng.integers(1, 4))))

        else:
            # Internet-only customers: teach the model when to recommend XM offers/bundles
            lines = int(r["mobile_line_count"])
            bill = int(r["current_mobile_bill"])

            if lines >= 1:
                _, _, savings = best_plan_and_savings(r)

                is_strong_target = (lines >= 2) and (bill >= 150) and (savings >= 20)
                is_medium_target = (lines >= 2) and (bill >= 120) and (savings >= 10)

                if is_strong_target:
                    if rng.random() < 0.75:
                        interactions.append(
                            (uid, name_to_id["Switch to Xfinity Mobile (Savings Offer)"], int(rng.integers(2, 5)))
                        )
                    if rng.random() < 0.35:
                        interactions.append(
                            (uid, name_to_id["Internet+Mobile Bundle"], int(rng.integers(1, 4)))
                        )

                elif is_medium_target:
                    if rng.random() < 0.45:
                        interactions.append(
                            (uid, name_to_id["Internet+Mobile Bundle"], int(rng.integers(1, 4)))
                        )
                    if rng.random() < 0.20:
                        interactions.append(
                            (uid, name_to_id["Switch to Xfinity Mobile (Savings Offer)"], int(rng.integers(1, 3)))
                        )

                else:
                    if rng.random() < 0.08:
                        interactions.append(
                            (uid, name_to_id["Internet+Mobile Bundle"], int(rng.integers(1, 3)))
                        )

        # Trim per user (keep the first internet interaction always)
        user_rows = [x for x in interactions if x[0] == uid]
        if len(user_rows) > max_per_user:
            keep = [user_rows[0]]
            rest = user_rows[1:]
            rng.shuffle(rest)
            keep += rest[: max_per_user - 1]
            interactions = [x for x in interactions if x[0] != uid] + keep

    interactions_df = pd.DataFrame(interactions, columns=["user_id", "item_id", "interaction_strength"])
    interactions_df = interactions_df.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    return interactions_df


# ----------------------------
# Main
# ----------------------------
def main():
    rng = np.random.default_rng(123)

    n_existing = 800
    n_new = 200
    pct_existing_with_mobile = 0.12

    items = make_items_v3()
    users = make_users_v3(
        rng=rng,
        n_existing=n_existing,
        n_new=n_new,
        pct_existing_with_mobile=pct_existing_with_mobile,
    )
    interactions = generate_interactions_v3(
        rng=rng,
        users=users,
        items=items,
        max_per_user=9,
    )

    users.to_csv("users_v3.csv", index=False)
    items.to_csv("items_v3.csv", index=False)
    interactions.to_csv("interactions_v3.csv", index=False)

    existing_with_mobile = int(((users["is_new_customer"] == 0) & (users["has_mobile"] == 1)).sum())
    print("Wrote users_v3.csv, items_v3.csv, interactions_v3.csv")
    print(f"Users: {len(users)} (existing={n_existing}, new={n_new})")
    print(f"Existing with mobile: {existing_with_mobile} ({existing_with_mobile/n_existing:.1%})")
    print(f"Items: {len(items)}")
    print(f"Interactions: {len(interactions)} (avg per existing user = {len(interactions)/n_existing:.2f})")


if __name__ == "__main__":
    main()
