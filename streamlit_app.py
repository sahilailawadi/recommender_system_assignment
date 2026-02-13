import streamlit as st
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset

# Page config
st.set_page_config(
    page_title="Comcast Product Recommender",
    page_icon="📡",
    layout="wide"
)

# Title and description
st.title("📡 Comcast Product Recommender System")
st.markdown("""
This AI-powered recommender system suggests personalized Comcast products based on customer profiles.
Built using **LightFM** hybrid matrix factorization for both existing and new customers.
""")

# Cache the model loading
@st.cache_resource
def load_model_and_data():
    """Load the trained model and data"""
    
    # Load data
    users = pd.read_csv("users_v3.csv")
    items = pd.read_csv("items_v3.csv")
    interactions = pd.read_csv("interactions_v3.csv")
    
    # Feature bins (same as notebook)
    def bin_token(prefix: str, value: int, bins: list) -> str:
        for low, high, label in bins:
            if low <= value <= high:
                return f"{prefix}={label}"
        return f"{prefix}=unknown"

    def count_bin(prefix: str, value: int) -> str:
        if value <= 0:
            return f"{prefix}=0"
        if value == 1:
            return f"{prefix}=1"
        return f"{prefix}=2+"

    def outage_bin(value: int) -> str:
        if value <= 0:
            return "outage=low"
        if value == 1:
            return "outage=med"
        return "outage=high"

    BUDGET_BINS = [(0, 59, "0-59"), (60, 79, "60-79"), (80, 119, "80-119"), (120, 9999, "120+")]
    BB_DATA_BINS = [(0, 299, "0-299GB"), (300, 699, "300-699GB"), (700, 1199, "700-1199GB"), (1200, 99999, "1200GB+")]
    MOBILE_DATA_BINS = [(0, 9, "0-9GB"), (10, 24, "10-24GB"), (25, 59, "25-59GB"), (60, 9999, "60GB+")]
    IOT_BINS = [(0, 5, "0-5"), (6, 15, "6-15"), (16, 30, "16-30"), (31, 9999, "31+")]
    DEV_BINS = [(0, 5, "0-5"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 9999, "16+")]
    MOBILE_BILL_BINS = [(0, 49, "0-49"), (50, 99, "50-99"), (100, 149, "100-149"), (150, 9999, "150+")]

    def build_user_feature_tokens(row: pd.Series, include_identity: bool = True) -> list:
        feats = []
        if include_identity:
            uid = int(row["user_id"])
            feats.append(f"user_id={uid}")
        
        feats.append(f"region={row['region']}")
        feats.append(outage_bin(int(row["outage_risk"])))
        feats.append(bin_token("budget", int(row["budget"]), BUDGET_BINS))
        feats.append(bin_token("bb_data", int(row["monthly_data_gb"]), BB_DATA_BINS))
        feats.append(bin_token("devices", int(row["devices"]), DEV_BINS))
        feats.append(bin_token("iot", int(row["iot_devices"]), IOT_BINS))
        feats.append(count_bin("wfh", int(row["wfh_count"])))
        feats.append(count_bin("gamer", int(row["gamer_count"])))
        feats.append(count_bin("creator", int(row["creator_count"])))
        feats.append(f"has_mobile={int(row['has_mobile'])}")
        feats.append(count_bin("lines", int(row["mobile_line_count"])))
        feats.append(bin_token("m_data", int(row["mobile_data_gb"]), MOBILE_DATA_BINS))
        feats.append(bin_token("m_bill", int(row["current_mobile_bill"]), MOBILE_BILL_BINS))
        feats.append(f"is_new={int(row.get('is_new_customer', 1))}")
        return feats

    def item_feature_tokens(row: pd.Series) -> list:
        feats = []
        iid = int(row["item_id"])
        feats.append(f"item_id={iid}")
        feats.append(f"category={row['category']}")
        price = int(row["price"])
        feats.append(bin_token("price", price, [(0, 0, "0"), (1, 14, "1-14"), (15, 29, "15-29"), (30, 59, "30-59"), (60, 9999, "60+")]))
        
        if pd.notna(row["speed_mbps"]):
            speed = int(row["speed_mbps"])
            feats.append(bin_token("speed", speed, [(0, 399, "0-399"), (400, 799, "400-799"), (800, 1499, "800-1499"), (1500, 99999, "1500+")]))
        else:
            feats.append("speed=na")
        
        if isinstance(row.get("notes", None), str) and "power outage" in row["notes"].lower():
            feats.append("storm_ready=yes")
        return feats

    # Build features
    user_features_map = {
        int(r.user_id): build_user_feature_tokens(pd.Series(r._asdict()), include_identity=True)
        for r in users.itertuples(index=False)
    }
    
    item_features_map = {
        int(r.item_id): item_feature_tokens(pd.Series(r._asdict()))
        for r in items.itertuples(index=False)
    }

    # Build dataset
    dataset = Dataset()
    all_user_ids = users["user_id"].astype(int).tolist()
    all_item_ids = items["item_id"].astype(int).tolist()
    
    all_user_feature_tokens = set()
    for feats in user_features_map.values():
        all_user_feature_tokens.update(feats)
    
    all_item_feature_tokens = set()
    for feats in item_features_map.values():
        all_item_feature_tokens.update(feats)
    
    dataset.fit(
        users=all_user_ids,
        items=all_item_ids,
        user_features=list(all_user_feature_tokens),
        item_features=list(all_item_feature_tokens),
    )

    triples = list(zip(
        interactions["user_id"].astype(int),
        interactions["item_id"].astype(int),
        interactions["interaction_strength"].astype(float),
    ))

    interactions_matrix, _ = dataset.build_interactions(triples)
    
    user_features = dataset.build_user_features(
        [(uid, feats) for uid, feats in user_features_map.items()],
        normalize=False
    )
    
    item_features = dataset.build_item_features(
        [(iid, feats) for iid, feats in item_features_map.items()],
        normalize=False
    )

    # Train model
    model = LightFM(no_components=32, loss="warp", learning_rate=0.05, 
                    item_alpha=1e-6, user_alpha=1e-6, random_state=42)
    model.fit(interactions_matrix, user_features=user_features, 
              item_features=item_features, epochs=20, num_threads=4, verbose=False)

    return model, dataset, users, items, user_features, item_features, build_user_feature_tokens

# Load everything
with st.spinner("Loading model and data..."):
    model, dataset, users, items, user_features, item_features, build_user_feature_tokens = load_model_and_data()
    user_id_map, _, item_id_map, _ = dataset.mapping()
    inv_item_id_map = {v: k for k, v in item_id_map.items()}

st.success("✅ Model loaded successfully!")

# Sidebar for mode selection
st.sidebar.header("Select Mode")
mode = st.sidebar.radio(
    "Choose recommendation type:",
    ["🆕 New Customer", "👤 Existing Customer"]
)

def get_recommendations(user_internal, user_feat_matrix, n_pool=10):
    """Generate recommendations"""
    all_item_internal = np.arange(len(item_id_map), dtype=np.int32)
    scores = model.predict(user_internal, all_item_internal, 
                          user_features=user_feat_matrix, 
                          item_features=item_features)
    top_internal = np.argsort(-scores)[:n_pool]
    
    ranked_item_ids = [int(inv_item_id_map[i]) for i in top_internal]
    df = items[items["item_id"].isin(ranked_item_ids)][["item_id", "item_name", "category", "price"]].copy()
    df["rank"] = df["item_id"].apply(lambda x: ranked_item_ids.index(int(x)))
    df = df.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)
    return df

def display_recommendations(df):
    """Display grouped recommendations"""
    if df.empty:
        st.warning("No recommendations available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Top Pick")
        top = df.iloc[[0]]
        st.dataframe(top[["item_name", "price"]], hide_index=True, width='stretch')
    
    with col2:
        st.subheader("➕ Add-Ons")
        addons = df[df["category"] == "addon"].head(3)
        if not addons.empty:
            st.dataframe(addons[["item_name", "price"]], hide_index=True, width='stretch')
        else:
            st.info("No add-ons available")
    
    with col3:
        st.subheader("📱 Mobile/Bundles")
        mobile = df[df["category"].isin(["bundle", "offer", "mobile_plan"])].head(3)
        if not mobile.empty:
            st.dataframe(mobile[["item_name", "price"]], hide_index=True, width='stretch')
        else:
            st.info("No mobile offers available")

# Main content
if mode == "🆕 New Customer":
    st.header("New Customer Profile")
    st.markdown("Enter customer details to get personalized recommendations:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("Region", ["NE", "SE", "MW", "W"])
        household_size = st.slider("Household Size", 1, 6, 3)
        wfh_level = st.selectbox("Work from Home Level", ["none", "some", "heavy"])
        gamer = st.checkbox("Any Gamers?")
    
    with col2:
        creator = st.checkbox("Any Content Creators?")
        mobile_lines = st.slider("Mobile Lines", 0, 6, 0)
        current_bill = st.number_input("Current Mobile Bill ($/mo)", 0, 500, 0 if mobile_lines == 0 else 100)
        budget = st.slider("Monthly Budget ($)", 40, 400, 150)
    
    if st.button("🔮 Get Recommendations", type="primary", width='stretch'):
        with st.spinner("Generating recommendations..."):
            # Create profile
            def infer_counts(household_size, wfh_level, gamer, creator):
                wfh_level = wfh_level.lower().strip()
                wfh_count = 0 if wfh_level == "none" else 1 if wfh_level == "some" else min(2, household_size)
                gamer_count = 1 if gamer else 0
                creator_count = 1 if creator else 0
                base_devices = 2 + household_size * 2 + (2 if wfh_count >= 1 else 0) + (2 if gamer else 0) + (2 if creator else 0)
                devices = int(np.clip(base_devices, 2, 18))
                iot_devices = int(np.clip((household_size - 1) * 5 + (3 if creator else 0), 0, 60))
                monthly_data_gb = 250 + household_size * 120 + (350 if wfh_count >= 1 else 0) + (350 if gamer else 0) + (450 if creator else 0) + int(iot_devices * 2)
                monthly_data_gb = int(np.clip(monthly_data_gb, 50, 2500))
                return wfh_count, gamer_count, creator_count, devices, iot_devices, monthly_data_gb
            
            outage_risk = 2 if region == "SE" else 0 if region == "W" else 1
            wfh_count, gamer_count, creator_count, devices, iot_devices, monthly_data_gb = infer_counts(household_size, wfh_level, gamer, creator)
            mobile_data_gb = int(np.clip(mobile_lines * (12 + (6 if gamer else 0) + (10 if creator else 0)), 0, 200))
            
            profile = {
                "user_id": -1, "is_new_customer": 1, "has_internet": 1, "has_mobile": 0,
                "region": region, "outage_risk": outage_risk, "household_size": household_size,
                "devices": devices, "iot_devices": iot_devices, "wfh_count": wfh_count,
                "gamer_count": gamer_count, "creator_count": creator_count, "budget": budget,
                "monthly_data_gb": monthly_data_gb, "mobile_line_count": mobile_lines,
                "mobile_data_gb": mobile_data_gb, "current_mobile_bill": current_bill,
            }
            
            row = pd.Series(profile)
            feats = build_user_feature_tokens(row, include_identity=False)
            
            any_known_user_id = next(iter(user_id_map.keys()))
            any_known_user_internal = user_id_map[any_known_user_id]
            
            temp_user_features = dataset.build_user_features([(any_known_user_id, feats)], normalize=False)
            
            recommendations = get_recommendations(any_known_user_internal, temp_user_features, n_pool=10)
            
            st.success("✨ Recommendations generated!")
            display_recommendations(recommendations)
            
            # Show savings if mobile
            if mobile_lines > 0 and current_bill > 0:
                with st.expander("💰 Potential Mobile Savings"):
                    plans = {"Unlimited": 45 * mobile_lines, "Unlimited+": 60 * mobile_lines, "By-the-Gig": 30 * mobile_lines}
                    best_plan = min(plans, key=plans.get)
                    best_cost = plans[best_plan]
                    savings = current_bill - best_cost
                    if savings > 0:
                        st.success(f"💵 Save **${savings}/mo** by switching to **{best_plan}** (${best_cost}/mo)")
                    else:
                        st.info("Your current plan is already competitive!")

else:  # Existing Customer
    st.header("Existing Customer Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, max_value=int(users["user_id"].max()), value=10)
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 15, 10)
    
    if st.button("🔮 Get Recommendations", type="primary", width='stretch'):
        if user_id in user_id_map:
            with st.spinner("Generating recommendations..."):
                u_internal = user_id_map[user_id]
                recommendations = get_recommendations(u_internal, user_features, n_pool=n_recommendations)
                
                # Show user info
                user_info = users[users["user_id"] == user_id].iloc[0]
                with st.expander("👤 Customer Profile"):
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric("Region", user_info["region"])
                        st.metric("Household Size", int(user_info["household_size"]))
                    with info_col2:
                        st.metric("Budget", f"${int(user_info['budget'])}")
                        st.metric("Data Usage", f"{int(user_info['monthly_data_gb'])} GB")
                    with info_col3:
                        st.metric("Mobile Lines", int(user_info["mobile_line_count"]))
                        st.metric("Has Mobile", "Yes" if user_info["has_mobile"] else "No")
                
                st.success("✨ Recommendations generated!")
                display_recommendations(recommendations)
        else:
            st.error(f"❌ User ID {user_id} not found in database")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using LightFM | Sahil Ailawadi | INFO 629</p>
</div>
""", unsafe_allow_html=True)
