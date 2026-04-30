import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Analytics App", layout="wide")

st.title("📊 Intelligent Business Analytics System")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔹 Initial Data Info")
    st.write(f"\"{uploaded_file.name}\" Dataset has {df.shape[0]} Rows and {df.shape[1]} Columns")

    # -------------------------------
    # DATA CLEANING
    # -------------------------------
    before_cols = df.shape[1]
    df = df.loc[:, ~df.columns.duplicated()]
    after_cols = df.shape[1]

    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]

    missing_before = df.isnull().sum().sum()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    missing_after = df.isnull().sum().sum()
    


    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------

    if 'registration_date' in df.columns:
        df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')

        import datetime as dt
        today = dt.datetime.today()

        df['customer_tenure_days'] = (today - df['registration_date']).dt.days
        df['customer_tenure_days'] = df['customer_tenure_days'].fillna(0)
        df['customer_tenure_days'] = df['customer_tenure_days'].replace(0, 1)

        if 'total_orders' in df.columns:
            df['orders_per_month'] = df['total_orders'] / (df['customer_tenure_days'] / 30)

        if 'total_spend_usd' in df.columns:
            df['spend_per_month'] = df['total_spend_usd'] / (df['customer_tenure_days'] / 30)

        st.subheader("🔹 Data Preview(Cleaned)")
        st.dataframe(df.head())
    
    # -------------------------------
    # CUSTOMER SEGMENTATION
    # -------------------------------
    st.subheader("📊 Customer Segmentation")

    seg_features = [
        'total_spend_usd',
        'total_orders',
        'days_since_last_purchase',
        'returns_made'
    ]

    missing_cols = [col for col in seg_features if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns for segmentation: {missing_cols}")

    else:
        X = df[seg_features]

        # -------------------------------
        # SCALING
        # -------------------------------
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------------------
        # AUTO SELECT K (MAX 5)
        # -------------------------------
        from sklearn.cluster import KMeans
        import numpy as np

        inertia = []
        K_range = range(2, 6)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        inertia = np.array(inertia)
        diffs = np.diff(inertia)
        diffs2 = np.diff(diffs)

        optimal_k = K_range[np.argmin(diffs2) + 1]

        st.success(f"Customers are categorized into {optimal_k} segments")

        # -------------------------------
        # APPLY KMEANS
        # -------------------------------
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # -------------------------------
        # CLUSTER SUMMARY
        # -------------------------------
        cluster_summary = df.groupby('cluster')[seg_features].mean()

        
        # -------------------------------
        # RANKING LOGIC
        # -------------------------------
        cluster_summary['spend_rank'] = cluster_summary['total_spend_usd'].rank(ascending=False)
        cluster_summary['orders_rank'] = cluster_summary['total_orders'].rank(ascending=False)
        cluster_summary['recency_rank'] = cluster_summary['days_since_last_purchase'].rank(ascending=True)

        
        # -------------------------------
        # SEGMENT LABELING (SCORE-BASED)
        # -------------------------------
        from sklearn.preprocessing import MinMaxScaler

        SEGMENT_PROFILES = {
            "High Value": {
                "total_spend_usd":            +1,
                "total_orders":               +1,
                "days_since_last_purchase":   -1,
                "returns_made":               -1,
            },
            "Medium Value": {
                "total_spend_usd":            +1,
                "total_orders":               +1,
                "days_since_last_purchase":   -1,
                "returns_made":               -1,
            },
            "Low Value": {
                "total_spend_usd":            -1,
                "total_orders":               -1,
                "days_since_last_purchase":   +1,
                "returns_made":               +1,
            },
            "At Risk": {
                "total_spend_usd":            +1,   # historically high spend
                "total_orders":               -1,   # but orders dropping
                "days_since_last_purchase":   +1,   # not buying recently
                "returns_made":               +1,   # returning items
            },
            # activates only if optimal_k = 5
            "Occasional": {
                "total_spend_usd":             0,
                "total_orders":               -1,
                "days_since_last_purchase":    0,
                "returns_made":               -1,
            },
        }

        # Rank features across clusters on 0-1 scale
        ranked = pd.DataFrame(
            MinMaxScaler().fit_transform(cluster_summary[seg_features]),
            index=cluster_summary.index,
            columns=seg_features,
        )

        # Score every cluster × segment pair
        scores = {}
        for cluster_id in ranked.index:
            scores[cluster_id] = {}
            for seg_name, weights in SEGMENT_PROFILES.items():
                score = 0.0
                for feat in seg_features:
                    w = weights.get(feat, 0)
                    val = ranked.loc[cluster_id, feat]
                    score += val * abs(w) if w > 0 else (1 - val) * abs(w)
                scores[cluster_id][seg_name] = score

        scores_df = pd.DataFrame(scores).T

        # Greedy assignment — best score wins, no duplicate labels
        cluster_labels = {}
        used_segments  = set()

        all_pairs = sorted(
            [(c, s, scores_df.loc[c, s]) for c in scores_df.index for s in scores_df.columns],
            key=lambda x: x[2], reverse=True
        )

        for cluster, seg, score in all_pairs:
            if cluster in cluster_labels or seg in used_segments:
                continue
            cluster_labels[cluster] = seg
            used_segments.add(seg)
            if len(cluster_labels) == len(ranked):
                break

        # Fallback for k > number of defined profiles
        label_counts = {}
        for cluster in ranked.index:
            if cluster not in cluster_labels:
                best_seg = scores_df.loc[cluster].idxmax()
                label_counts[best_seg] = label_counts.get(best_seg, 1) + 1
                cluster_labels[cluster] = f"{best_seg} ({label_counts[best_seg]})"

        df['segment'] = df['cluster'].map(cluster_labels)

        # Add segment names
        cluster_summary['segment'] = cluster_summary.index.map(cluster_labels)

        # Set segment as index
        cluster_summary = cluster_summary.set_index('segment')

        # Optional: drop cluster index completely
        cluster_summary = cluster_summary.reset_index()

        st.subheader("📊 Segment Summary")
        st.dataframe(cluster_summary)

        # -------------------------------
        # VISUALIZATION
        # -------------------------------
        import plotly.express as px
        import plotly.graph_objects as go

        st.subheader("📊 Customer Segments Overview")

        segment_counts = df['segment'].value_counts().reset_index()
        segment_counts.columns = ['segment', 'count']

        # -------------------------------
        # SEGMENT SELECTOR
        # -------------------------------
        options = ["All Customers"] + segment_counts['segment'].tolist()

        selected_segment = st.radio(
            "Select a segment to explore:",
            options=options,
            index=0,  # default
            horizontal=True
            )

        # -------------------------------
        # LAYOUT: PIE + DETAILS SIDE BY SIDE
        # -------------------------------
        col1, col2 = st.columns([1, 1])

        with col1:
            # Build pull array based on selection
            pull = [
                0.1 if seg == selected_segment else 0
                for seg in segment_counts['segment']
            ]

            fig = go.Figure(go.Pie(
                labels=segment_counts['segment'],
                values=segment_counts['count'],
                pull=pull,
                hole=0.35,
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>"
            ))

            fig.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if selected_segment == "All Customers":
                filtered_df = df
            else:
                filtered_df = df[df['segment'] == selected_segment]

            if selected_segment == "All Customers":
                st.markdown("### 📋 All Customers Overview")
            else:
                st.markdown(f"### 📋 {selected_segment}")

            # KPI cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers",    f"{len(filtered_df):,}")
            m2.metric("Total Revenue", f"${filtered_df['total_spend_usd'].sum():,.0f}")
            m3.metric("Avg Orders",   f"{filtered_df['total_orders'].mean():.1f}")

            st.write("**📊 Average Behavior:**")

            avg_values = filtered_df[seg_features].mean()

            def highlight(value, color="#00BFFF"):
                return f"<span style='color:{color}; font-weight:bold'>{value}</span>"

            st.markdown(
                f"""
                - 💰 Average Spending: {highlight(f"${avg_values['total_spend_usd']:.2f}", "#2ecc71")}
                - 🛒 Average Orders: {highlight(f"{avg_values['total_orders']:.2f}", "#3498db")}
                - ⏱ Days Since Last Purchase: {highlight(f"{avg_values['days_since_last_purchase']:.1f}", "#e67e22")}
                - 🔁 Average Returns made: {highlight(f"{avg_values['returns_made']:.2f}", "#e74c3c")}
                """,
                unsafe_allow_html=True
            )

            st.write("**Sample Customers:**")
            st.dataframe(filtered_df[seg_features].head(), use_container_width=True)
        
    st.subheader("⚠️ Customer Churn Analysis")

    # -------------------------------
    # PREPARE DATA
    # -------------------------------
    churn_features = [
        'total_spend_usd',
        'total_orders',
        'days_since_last_purchase',
        'returns_made',
        'segment',
        'reviews_given',
        'avg_review_score',
        'wishlist_items',
        'newsletter_subscribed'
    ]

    target = 'churned'

    missing_cols = [col for col in churn_features + [target] if col not in df.columns]

    if missing_cols:
        st.warning(f"Missing columns for churn model: {missing_cols}")

    else:
        df_model = pd.get_dummies(df[churn_features + [target]], drop_first=True)

        from sklearn.model_selection import train_test_split
        X = df_model.drop('churned', axis=1)
        y = df_model['churned']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        from sklearn.ensemble import RandomForestClassifier

        rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight={0:1, 1:3}
        )

        rf_model.fit(X_train, y_train)

        from sklearn.metrics import classification_report

        y_prob = rf_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.2).astype(int)

        from sklearn.metrics import classification_report

        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']

        st.subheader("📊 Churn Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Precision", f"{precision:.2f}")
        col2.metric("Recall", f"{recall:.2f}")
        col3.metric("F1 Score", f"{f1:.2f}")

        st.write("### 🧠 Model Interpretation")

        st.markdown(f"""
        - 🎯 **Precision ({precision:.2f})** → Of predicted churn customers, how many actually churned  
        - 🔍 **Recall ({recall:.2f})** → Of actual churn customers, how many we correctly identified  
        - ⚖️ **F1 Score ({f1:.2f})** → Balance between precision and recall  
        """)

        df['churn_probability'] = rf_model.predict_proba(X)[:, 1]

        st.write("### 📊 Segment-wise Churn")

        churn_seg = df.groupby('segment').agg({
            'churn_probability': 'mean',
            'total_spend_usd': 'sum'
        }).reset_index()

        churn_seg.columns = ['Segment', 'Avg Churn Probability', 'Revenue']

        import plotly.express as px

        fig = px.bar(
            churn_seg,
            x='Segment',
            y='Avg Churn Probability',
            color='Avg Churn Probability',
            title="Churn Risk by Segment"
        )

        st.plotly_chart(fig, use_container_width=True)

        df['revenue_at_risk'] = df['total_spend_usd'] * df['churn_probability']

        total_risk = df['revenue_at_risk'].sum()

        st.metric("💸 Total Revenue at Risk", f"${total_risk:,.0f}")

        risk_seg = df.groupby('segment')['revenue_at_risk'].sum().reset_index()

        fig2 = px.bar(
            risk_seg,
            x='segment',
            y='revenue_at_risk',
            title="Revenue at Risk by Segment"
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.write("### 🔍 Top Churn Drivers")

        # import shap
        # import numpy as np

        # X_train = X_train.astype(float)
        # X_test = X_test.astype(float)
        # X_test = X_test[X_train.columns]

        # explainer = shap.TreeExplainer(rf_model)
        # shap_values = explainer.shap_values(X_test)

        # if isinstance(shap_values, list):
        #     sv = shap_values[1]
        # else:
        #     sv = shap_values[:, :, 1]

        # # IMPORTANT: matplotlib backend for Streamlit
        # import matplotlib.pyplot as plt

        # fig_shap = plt.figure()
        # shap.summary_plot(sv, X_test, show=False)

        # st.pyplot(fig_shap)

        # st.subheader("🧠 Churn Insights (Key Drivers)")

        # # Mean absolute SHAP importance
        # feature_importance = pd.DataFrame({
        #     'feature': X_test.columns,
        #     'importance': np.abs(sv).mean(axis=0)
        # }).sort_values(by='importance', ascending=False)

        # top_features = feature_importance.head(3)

        # insights = []

        # for _, row in top_features.iterrows():
        #     feature = row['feature']

        #     if 'days_since_last_purchase' in feature:
        #         insights.append("📉 Customers with longer inactivity tend to churn more.")

        #     elif 'total_orders' in feature:
        #         insights.append("🛒 Low purchase frequency is driving customer churn.")

        #     elif 'total_spend_usd' in feature:
        #         insights.append("💰 Lower spending customers are more likely to churn.")

        #     elif 'returns_made' in feature:
        #         insights.append("🔁 High product returns are contributing to customer dissatisfaction and churn.")

        #     elif 'reviews_given' in feature:
        #         insights.append("📝 Low engagement (fewer reviews) is linked to higher churn.")

        #     elif 'avg_review_score' in feature:
        #         insights.append("⭐ Poor customer experience (low review scores) is increasing churn risk.")

        #     elif 'wishlist_items' in feature:
        #         insights.append("🧾 Customers with low intent (fewer wishlist items) tend to churn more.")

        #     elif 'newsletter_subscribed' in feature:
        #         insights.append("📧 Customers not subscribed to newsletters are less engaged and more likely to churn.")

        #     elif 'segment_' in feature:
        #         insights.append("👥 Certain customer segments have higher churn tendencies.")

        # # Remove duplicates and show
        # for insight in list(set(insights)):
        #     st.write(f"- {insight}")
        
    # -------------------------------
    # SALES FORECASTING
    # -------------------------------
    """
    Sales Forecasting Dashboard with Prophet + Churn Adjustment
    Run: streamlit run sales_forecast_app.py
    """

    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from prophet import Prophet
    from datetime import timedelta
    import warnings
    warnings.filterwarnings("ignore")

    # ─── Page config ───────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Sales Forecast Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .metric-card {
        background: #f8f9fa;
        color: #1a1a1a;   /* FIX */
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
    }

    .insight-box {
        background: #eef4ff;
        color: #1a1a1a;   /* FIX */
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 15px;
    }

    .action-box {
        background: #fff8e1;
        color: #1a1a1a;   /* FIX */
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 15px;
        border-left: 4px solid #f5a623;
    }

    .warning-box {
        background: #fff0f0;
        color: #1a1a1a;   /* FIX */
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #e74c3c;
    }

    .success-box {
        background: #f0fff4;
        color: #1a1a1a;   /* FIX */
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)
    # ─── Helper functions ──────────────────────────────────────────────────────────

    def fmt_inr(value):
        """Format a USD value into readable INR-style lacs/crore label."""
        if value >= 1_00_00_000:
            return f"₹{value/1_00_00_000:.2f} Cr"
        elif value >= 1_00_000:
            return f"₹{value/1_00_000:.2f} L"
        else:
            return f"₹{value:,.0f}"

    def fmt_usd(value):
        """Format USD value."""
        if value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.0f}"

    def compute_period_churn_rate(df_orders, period_days):
        """
        Compute a period-specific churn rate from order history.
        
        Logic:
        1. Identify each customer's last purchase date.
        2. A customer is "at risk of churning" if their last purchase
            was more than (avg_purchase_gap * 2) days ago.
        3. The period churn impact = fraction of recent revenue
            from at-risk customers × a decay factor for the forecast window.
        
        This avoids the lifetime-churn mistake of applying a cumulative
        churn rate directly to a short forecast window.
        
        Returns:
            churn_rate_for_period (float): fraction of forecast revenue at risk
            churn_stats (dict): diagnostic info for the UI
        """
        df = df_orders.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])

        # Per-customer stats
        customer_stats = df.groupby("customer_id").agg(
            last_order=("order_date", "max"),
            first_order=("order_date", "min"),
            order_count=("order_id", "count"),
            total_revenue=("total_amount_usd", "sum"),
        ).reset_index()

        dataset_end = df["order_date"].max()

        # Average inter-purchase gap (days) across repeat customers
        repeat_customers = customer_stats[customer_stats["order_count"] > 1].copy()
        if len(repeat_customers) > 0:
            repeat_customers["lifespan_days"] = (
                repeat_customers["last_order"] - repeat_customers["first_order"]
            ).dt.days
            repeat_customers["avg_gap"] = (
                repeat_customers["lifespan_days"] / (repeat_customers["order_count"] - 1)
            )
            avg_gap_days = repeat_customers["avg_gap"].median()
        else:
            avg_gap_days = 30  # fallback

        # Churn threshold: customer hasn't ordered in 2× their avg gap
        churn_threshold_days = max(avg_gap_days * 2, 30)

        customer_stats["days_since_last"] = (
            dataset_end - customer_stats["last_order"]
        ).dt.days

        # At-risk = last purchase was more than churn_threshold ago
        at_risk = customer_stats[
            customer_stats["days_since_last"] > churn_threshold_days
        ]

        total_customers = len(customer_stats)
        at_risk_count = len(at_risk)
        lifetime_churn_rate = at_risk_count / total_customers if total_customers > 0 else 0

        # Revenue share from at-risk customers
        at_risk_revenue_share = (
            at_risk["total_revenue"].sum() / customer_stats["total_revenue"].sum()
            if customer_stats["total_revenue"].sum() > 0 else 0
        )

        # Period-specific churn impact:
        # Not all at-risk customers will churn in the next N days.
        # We scale by period_days / 90 (assuming churn manifests over ~90 days)
        # and cap at 0.5 to prevent nonsensical negative forecasts.
        period_churn_rate = min(
            at_risk_revenue_share * (period_days / 90.0), 0.5
        )

        churn_stats = {
            "lifetime_churn_pct": lifetime_churn_rate * 100,
            "at_risk_customers": at_risk_count,
            "total_customers": total_customers,
            "avg_purchase_gap_days": avg_gap_days,
            "churn_threshold_days": churn_threshold_days,
            "at_risk_revenue_share_pct": at_risk_revenue_share * 100,
            "period_churn_rate": period_churn_rate,
        }

        return period_churn_rate, churn_stats


    def generate_insights(forecast, sales_df, df_orders, period_days, churn_stats, period_churn_rate):
        """Generate text insights from the forecast and raw data."""
        insights = []
        actions = []

        future_fc = forecast[forecast["ds"] > sales_df["ds"].max()].copy()
        if future_fc.empty:
            return insights, actions

        total_forecast = future_fc["yhat"].sum()
        churn_adjusted = total_forecast * (1 - period_churn_rate)
        churn_drop_pct = period_churn_rate * 100

        # Unit label
        unit = "next 30 days" if period_days == 30 else (
            "next quarter" if period_days == 90 else f"next {period_days} days"
        )

        # 1. Core forecast vs churn-adjusted
        insights.append(
            f"📊 Forecast for the {unit} is **{fmt_usd(total_forecast)}** "
            f"but due to churn risk, expected sales drop to **{fmt_usd(churn_adjusted)}** "
            f"(a {churn_drop_pct:.1f}% reduction)."
        )

        # 2. YoY comparison
        last_year_same_start = sales_df["ds"].max() - timedelta(days=365)
        last_year_same_end = last_year_same_start + timedelta(days=period_days)
        yoy_mask = (sales_df["ds"] >= last_year_same_start) & (sales_df["ds"] <= last_year_same_end)
        yoy_revenue = sales_df.loc[yoy_mask, "y"].sum()
        if yoy_revenue > 0:
            yoy_change = ((total_forecast - yoy_revenue) / yoy_revenue) * 100
            direction = "grow" if yoy_change >= 0 else "decline"
            insights.append(
                f"📅 Sales expected to **{direction} {abs(yoy_change):.1f}%** "
                f"compared to the same period last year."
            )
            if yoy_change < -10:
                actions.append("⚠️ Year-over-year decline detected — review pricing strategy and discount effectiveness.")
            elif yoy_change > 20:
                actions.append("🚀 Strong YoY growth projected — consider scaling inventory and marketing spend.")

        # 3. Peak months detection (using historical data)
        df_orders["order_date"] = pd.to_datetime(df_orders["order_date"])
        monthly_avg = (
            df_orders.groupby(df_orders["order_date"].dt.month)["total_amount_usd"]
            .sum()
            .reset_index()
        )
        monthly_avg.columns = ["month", "revenue"]
        top_months = monthly_avg.nlargest(3, "revenue")["month"].tolist()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        spike_months = ", ".join([month_names[m] for m in top_months])
        insights.append(f"📈 Historical sales spikes occur during **{spike_months}** — plan promotions accordingly.")

        # 4. Top revenue category
        if "category" in df_orders.columns:
            top_cat = (
                df_orders.groupby("category")["total_amount_usd"].sum().idxmax()
            )
            insights.append(f"🏆 Top revenue-generating category: **{top_cat}** — consider bundling or upsell strategies here.")
            actions.append(f"💡 Create targeted campaigns for '{top_cat}' to capitalize on its revenue dominance.")

        # 5. Churn-specific actions
        if churn_stats["at_risk_revenue_share_pct"] > 20:
            actions.append(
                f"🔴 {churn_stats['at_risk_revenue_share_pct']:.1f}% of historical revenue comes from at-risk customers "
                f"— launch a re-engagement campaign immediately."
            )
        elif churn_stats["at_risk_revenue_share_pct"] > 10:
            actions.append(
                f"🟡 {churn_stats['at_risk_revenue_share_pct']:.1f}% revenue at-risk from churning customers "
                f"— consider loyalty incentives or win-back emails."
            )

        # 6. Day-of-week pattern
        if "day_of_week" in df_orders.columns:
            dow_revenue = df_orders.groupby("day_of_week")["total_amount_usd"].mean()
            best_day = dow_revenue.idxmax()
            worst_day = dow_revenue.idxmin()
            insights.append(
                f"📆 Best sales day: **{best_day}** | Slowest day: **{worst_day}**."
            )
            actions.append(
                f"📢 Consider weekend/slow-day promotions on {worst_day}s to flatten revenue curve."
            )

        # 7. AOV trend
        if "order_id" in df_orders.columns:
            recent_3m = df_orders[df_orders["order_date"] >= df_orders["order_date"].max() - timedelta(days=90)]
            older_3m = df_orders[
                (df_orders["order_date"] >= df_orders["order_date"].max() - timedelta(days=180)) &
                (df_orders["order_date"] < df_orders["order_date"].max() - timedelta(days=90))
            ]
            if len(recent_3m) > 0 and len(older_3m) > 0:
                aov_recent = recent_3m["total_amount_usd"].mean()
                aov_older = older_3m["total_amount_usd"].mean()
                aov_change = ((aov_recent - aov_older) / aov_older) * 100
                direction = "rising" if aov_change > 0 else "falling"
                insights.append(
                    f"🛒 Average order value is **{direction}** ({aov_change:+.1f}% vs prior quarter) "
                    f"— current AOV: {fmt_usd(aov_recent)}."
                )
                if aov_change < -5:
                    actions.append("💰 Declining AOV — test upsell prompts, product bundles, or minimum order thresholds.")

        # 8. Spike upcoming months
        future_fc["month"] = future_fc["ds"].dt.month
        if not future_fc.empty:
            upcoming_peak_month = future_fc.groupby("month")["yhat"].sum().idxmax()
            insights.append(
                f"🗓️ Highest forecasted revenue month in the next period: **{month_names.get(upcoming_peak_month, upcoming_peak_month)}** "
                f"— ensure stock and staffing are ready."
            )
            actions.append(
                f"📦 Pre-stock inventory for {month_names.get(upcoming_peak_month, upcoming_peak_month)} — forecasted peak month."
            )

        return insights, actions


    # ─── Plots ─────────────────────────────────────────────────────────────────────

    def plot_individual(forecast, sales_df, title, color, show_churn=False, churn_rate=0.0):
        """Individual forecast plot."""
        hist_mask = forecast["ds"] <= sales_df["ds"].max()
        fut_mask = forecast["ds"] > sales_df["ds"].max()

        fig = go.Figure()

        # Historical actuals
        fig.add_trace(go.Scatter(
            x=sales_df["ds"], y=sales_df["y"],
            mode="lines",
            name="Actual Sales",
            line=dict(color="#636EFA", width=1.5),
            opacity=0.8,
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast.loc[fut_mask, "ds"], forecast.loc[fut_mask, "ds"].iloc[::-1]]),
            y=pd.concat([forecast.loc[fut_mask, "yhat_upper"], forecast.loc[fut_mask, "yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.15)" if color == "#00CC96" else "rgba(239, 85, 59, 0.15)",        line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
            showlegend=True,
        ))

        y_forecast = forecast.loc[fut_mask, "yhat"]
        if show_churn:
            y_forecast = y_forecast * (1 - churn_rate)

        fig.add_trace(go.Scatter(
            x=forecast.loc[fut_mask, "ds"],
            y=y_forecast,
            mode="lines",
            name="Churn-Adjusted Forecast" if show_churn else "Sales Forecast",
            line=dict(color=color, width=2.5, dash="dash" if show_churn else "solid"),
        ))

        # Vertical divider
        split_date = sales_df["ds"].max()

        fig.add_shape(
            type="line",
            x0=split_date,
            x1=split_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", dash="dot"),
        )

        fig.add_annotation(
            x=split_date,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast →",
            showarrow=False,
            xanchor="left",
            yanchor="bottom"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales (USD)",
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        return fig


    def plot_overlapping(forecast, sales_df, churn_rate):
        """Overlapping plot: raw forecast vs churn-adjusted."""
        fut_mask = forecast["ds"] > sales_df["ds"].max()
        future = forecast[fut_mask].copy()

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=sales_df["ds"], y=sales_df["y"],
            mode="lines", name="Actual Sales",
            line=dict(color="#636EFA", width=1.5), opacity=0.7,
        ))

        # Raw forecast
        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"],
            mode="lines", name="Prophet Forecast",
            line=dict(color="#00CC96", width=2.5),
        ))

        # CI for raw
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,204,150,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prophet CI",
        ))

        # Churn-adjusted
        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"] * (1 - churn_rate),
            mode="lines", name="Churn-Adjusted Forecast",
            line=dict(color="#EF553B", width=2.5, dash="dash"),
        ))

        # Gap fill between the two lines
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
            y=pd.concat([future["yhat"], (future["yhat"] * (1 - churn_rate)).iloc[::-1]]),
            fill="toself", fillcolor="rgba(239,85,59,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Churn Impact Zone",
        ))

        split_date = sales_df["ds"].max()
        fig.add_shape(
            type="line",
            x0=split_date,
            x1=split_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", dash="dot"),
        )

        fig.add_annotation(
            x=split_date,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast →",
            showarrow=False,
            xanchor="left",
            yanchor="bottom"
        )

        fig.update_layout(
            title="Forecast vs Churn-Adjusted Forecast (Overlapping)",
            xaxis_title="Date", yaxis_title="Sales (USD)",
            hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        return fig


    def plot_components_custom(forecast):
        """Trend + weekly + yearly components."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["trend"],
            mode="lines", name="Trend",
            line=dict(color="#AB63FA", width=2),
        ))
        fig.update_layout(
            title="Sales Trend Component",
            xaxis_title="Date", yaxis_title="Trend",
            plot_bgcolor="white", paper_bgcolor="white",
            height=280, margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        return fig


    # ─── Main App ──────────────────────────────────────────────────────────────────

    def main():
        st.title("📈 Sales Forecasting Dashboard")
        st.markdown("Upload your orders CSV to generate a Prophet forecast with churn-adjusted projections.")

        # ── Sidebar ──
        with st.sidebar:
            st.header("⚙️ Settings")
            uploaded_file = st.file_uploader("Upload orders.csv", type=["csv"])
            forecast_days = st.selectbox(
                "Forecast horizon",
                options=[30, 60, 90, 180, 365],
                index=0,
                format_func=lambda x: f"{x} days ({x//30} month{'s' if x>30 else ''})"
            )
            st.markdown("---")
            st.markdown("**Churn adjustment settings**")
            churn_mode = st.radio(
                "Churn rate calculation",
                ["Auto (from data)", "Manual override"],
                index=0,
            )
            manual_churn = None
            if churn_mode == "Manual override":
                manual_churn = st.slider(
                    "Monthly churn rate (%)", min_value=1, max_value=30, value=5
                ) / 100

            st.markdown("---")
            run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

        if not uploaded_file:
            st.info("👆 Upload your orders CSV to get started. Expected columns: order_id, customer_id, order_date, total_amount_usd, category, day_of_week, etc.")
            st.markdown("""
            **Expected CSV columns (minimum required):**
            - `order_id` — unique order identifier
            - `customer_id` — unique customer identifier  
            - `order_date` — date of the order (YYYY-MM-DD or DD-MM-YYYY)
            - `total_amount_usd` — order value in USD
            
            **Optional columns for richer insights:**
            - `category` — product category
            - `day_of_week` — day name (Monday, Tuesday, …)
            """)
            return

        # ── Load data ──
        df_orders = pd.read_csv(uploaded_file)

        # Normalize column names
        df_orders.columns = df_orders.columns.str.strip().str.lower().str.replace(" ", "_")

        required_cols = {"order_date", "total_amount_usd"}
        if not required_cols.issubset(df_orders.columns):
            st.error(f"Missing required columns: {required_cols - set(df_orders.columns)}")
            return

        df_orders["order_date"] = pd.to_datetime(df_orders["order_date"], errors="coerce")
        df_orders = df_orders.dropna(subset=["order_date"])
        
        if not run_btn:
            st.success(f"✅ Loaded {len(df_orders):,} orders. Configure settings in the sidebar and click **Run Forecast**.")
            st.dataframe(df_orders.head(5), use_container_width=True)
            return

        # ── Prepare time series ──
        sales_df = (
        df_orders
        .groupby("order_date", as_index=False)
        .agg({"total_amount_usd": "sum"})
        .rename(columns={"order_date": "ds", "total_amount_usd": "y"})
    )

        # 🔥 FIX: ensure unique dates
        sales_df = sales_df.drop_duplicates(subset=["ds"])

        sales_df = sales_df.sort_values("ds").reset_index(drop=True)
        
        with st.spinner("Training Prophet model..."):
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            if sales_df["ds"].duplicated().any():
                st.error("Duplicate dates found — fix your dataset")
                st.stop()
            st.write("Duplicate dates:", sales_df["ds"].duplicated().sum())
            st.write(sales_df[sales_df["ds"].duplicated()])
            model.fit(sales_df)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

        # ── Churn calculation ──
        if manual_churn is not None:
            # User provided a monthly rate — convert to period rate
            months_in_period = forecast_days / 30.0
            period_churn_rate = 1 - (1 - manual_churn) ** months_in_period
            period_churn_rate = min(period_churn_rate, 0.5)
            churn_stats = {
                "lifetime_churn_pct": manual_churn * 100,
                "at_risk_customers": "N/A (manual)",
                "total_customers": "N/A",
                "avg_purchase_gap_days": "N/A",
                "churn_threshold_days": "N/A",
                "at_risk_revenue_share_pct": period_churn_rate * 100,
                "period_churn_rate": period_churn_rate,
            }
        else:
            period_churn_rate, churn_stats = compute_period_churn_rate(df_orders, forecast_days)

        # ── Metrics ──
        future_fc = forecast[forecast["ds"] > sales_df["ds"].max()]
        total_forecast = future_fc["yhat"].sum()
        churn_adjusted_total = total_forecast * (1 - period_churn_rate)
        churn_impact = total_forecast - churn_adjusted_total

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Raw Forecast", fmt_usd(total_forecast),
                    help=f"Prophet forecast for next {forecast_days} days")
        with col2:
            st.metric("Churn-Adjusted", fmt_usd(churn_adjusted_total),
                    delta=f"-{fmt_usd(churn_impact)} at risk",
                    delta_color="inverse")
        with col3:
            st.metric("Period Churn Impact", f"{period_churn_rate*100:.1f}%",
                    help="Fraction of forecasted revenue at risk due to churn in this period")
        with col4:
            avg_daily = future_fc["yhat"].mean()
            st.metric("Avg Daily Forecast", fmt_usd(avg_daily))

        st.markdown("---")

        # ── Plots ──
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overlapping View", "📈 Prophet Forecast", "📉 Churn-Adjusted", "📋 Trend Component"
        ])

        with tab1:
            st.plotly_chart(
                plot_overlapping(forecast, sales_df, period_churn_rate),
                use_container_width=True
            )

        with tab2:
            st.plotly_chart(
                plot_individual(forecast, sales_df,
                                "Sales Forecast (Prophet)", "#00CC96"),
                use_container_width=True
            )

        with tab3:
            st.plotly_chart(
                plot_individual(forecast, sales_df,
                                "Churn-Adjusted Forecast", "#EF553B",
                                show_churn=True, churn_rate=period_churn_rate),
                use_container_width=True
            )

        with tab4:
            st.plotly_chart(
                plot_components_custom(forecast),
                use_container_width=True
            )

        st.markdown("---")

        # ── Insights & Actions ──
        insights, actions = generate_insights(
            forecast, sales_df, df_orders, forecast_days, churn_stats, period_churn_rate
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("💡 Forecast Insights")
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        with col_right:
            st.subheader("🎯 Recommended Actions")
            for action in actions:
                st.markdown(f'<div class="action-box">{action}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Churn diagnostics ──
        with st.expander("🔍 Churn Calculation Details (click to expand)"):
            st.markdown("""
            **How churn is calculated (not the wrong way)**

            The dashboard avoids the common mistake of applying a *lifetime* churn rate to a
            short forecast window. Instead:

            1. It identifies each customer's last order date.
            2. Customers who haven't ordered in more than **2× their average purchase gap** are
            flagged as "at risk."
            3. The *revenue share* from at-risk customers is computed.
            4. That share is scaled by `period_days / 90` — meaning a 30-day window sees ~⅓ of the
            90-day churn impact, not the full lifetime churn rate.
            5. The result is capped at 50% to prevent negative forecasts.
            """)

            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.markdown(f"**Total unique customers:** {churn_stats['total_customers']}")
                st.markdown(f"**At-risk customers:** {churn_stats['at_risk_customers']}")
                st.markdown(f"**Avg purchase gap:** {churn_stats['avg_purchase_gap_days']:.0f} days" if isinstance(churn_stats['avg_purchase_gap_days'], float) else f"**Avg purchase gap:** {churn_stats['avg_purchase_gap_days']}")
            with diag_col2:
                st.markdown(f"**Churn threshold:** {churn_stats['churn_threshold_days']:.0f} days" if isinstance(churn_stats['churn_threshold_days'], float) else f"**Churn threshold:** {churn_stats['churn_threshold_days']}")
                st.markdown(f"**Revenue share at risk:** {churn_stats['at_risk_revenue_share_pct']:.1f}%")
                st.markdown(f"**Period churn rate applied:** {churn_stats['period_churn_rate']*100:.2f}%")

        # ── Raw forecast table ──
        with st.expander("📄 Forecast Data Table"):
            display_fc = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            display_fc["churn_adjusted_yhat"] = display_fc["yhat"] * (1 - period_churn_rate)
            display_fc.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound", "Churn-Adjusted"]
            for col in ["Forecast", "Lower Bound", "Upper Bound", "Churn-Adjusted"]:
                display_fc[col] = display_fc[col].round(2)
            st.dataframe(display_fc.reset_index(drop=True), use_container_width=True)
            csv = display_fc.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")


    if __name__ == "__main__":
        main()

else:
    st.info("Please upload a dataset to continue.")


