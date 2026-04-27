from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="bike",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRIMARY_COLOR = "#4C78A8"
MAX_COLOR = "#F58518"
MIN_COLOR = "#54A24B"
LINE_WORKDAY = "#4C78A8"
LINE_WEEKEND = "#E45756"


st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #5f6b7a;
            margin-bottom: 1.5rem;
        }
        .insight-box {
            background: #f5f8fc;
            border-left: 0.4rem solid #4c78a8;
            padding: 1rem 1.2rem;
            border-radius: 0.5rem;
            margin: 0.75rem 0 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    base_paths = [
        Path(__file__).parent.parent,
        Path(__file__).parent,
        Path("."),
    ]

    day_df = None
    hour_df = None

    for base in base_paths:
        day_path = base / "day.csv"
        hour_path = base / "hour.csv"
        if day_path.exists() and hour_path.exists():
            day_df = pd.read_csv(day_path)
            hour_df = pd.read_csv(hour_path)
            break

    if day_df is None or hour_df is None:
        st.error("File day.csv dan hour.csv tidak ditemukan.")
        st.stop()

    day_df["dteday"] = pd.to_datetime(day_df["dteday"])
    hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])

    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {1: "Cerah", 2: "Berawan", 3: "Hujan Ringan", 4: "Hujan Lebat"}
    weekday_map = {
        0: "Minggu",
        1: "Senin",
        2: "Selasa",
        3: "Rabu",
        4: "Kamis",
        5: "Jumat",
        6: "Sabtu",
    }
    month_map = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "Mei",
        6: "Jun",
        7: "Jul",
        8: "Ags",
        9: "Sep",
        10: "Okt",
        11: "Nov",
        12: "Des",
    }

    for df in (day_df, hour_df):
        df["season_label"] = df["season"].map(season_map)
        df["weather_label"] = df["weathersit"].map(weather_map)
        df["weekday_label"] = df["weekday"].map(weekday_map)
        df["month_label"] = df["mnth"].map(month_map)
        df["day_type"] = df["workingday"].map({0: "Hari Libur/Weekend", 1: "Hari Kerja"})
        df["temp_actual"] = df["temp"] * 47 - 8

    q1 = day_df["cnt"].quantile(0.25)
    q2 = day_df["cnt"].quantile(0.50)
    q3 = day_df["cnt"].quantile(0.75)
    bins = [0, q1, q2, q3, day_df["cnt"].max() + 1]
    labels_cluster = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    day_df["usage_cluster"] = pd.cut(
        day_df["cnt"], bins=bins, labels=labels_cluster, include_lowest=True
    )

    return day_df, hour_df


def make_highlight_colors(values):
    if len(values) == 0:
        return []

    colors = [PRIMARY_COLOR] * len(values)
    max_idx = values.idxmax()
    min_idx = values.idxmin()
    colors[max_idx] = MAX_COLOR
    if min_idx == max_idx:
        return colors
    colors[min_idx] = MIN_COLOR
    return colors


def add_bar_labels(ax, bars, fmt="{:,.0f}", offset=0):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


day_df, hour_df = load_data()


with st.sidebar:
    st.title("Bike Sharing Dashboard")
    st.markdown("Filter data berdasarkan rentang tanggal dan musim.")

    min_date = day_df["dteday"].min().date()
    max_date = day_df["dteday"].max().date()

    start_date = st.date_input(
        "Tanggal mulai",
        min_date,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.date_input(
        "Tanggal akhir",
        max_date,
        min_value=min_date,
        max_value=max_date,
    )

    season_options = ["Semua"] + sorted(day_df["season_label"].dropna().unique().tolist())
    selected_season = st.selectbox("Pilih musim", season_options)

    st.markdown("---")
    st.caption("Dataset: Bike Sharing Dataset")
    st.caption("Sumber: UCI Machine Learning Repository")


if start_date > end_date:
    st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
    st.stop()


filtered_day = day_df[
    (day_df["dteday"].dt.date >= start_date) & (day_df["dteday"].dt.date <= end_date)
].copy()
filtered_hour = hour_df[
    (hour_df["dteday"].dt.date >= start_date) & (hour_df["dteday"].dt.date <= end_date)
].copy()

if selected_season != "Semua":
    filtered_day = filtered_day[filtered_day["season_label"] == selected_season]
    filtered_hour = filtered_hour[filtered_hour["season_label"] == selected_season]

if filtered_day.empty or filtered_hour.empty:
    st.warning("Tidak ada data pada filter yang dipilih.")
    st.stop()


st.markdown('<div class="main-title">Bike Sharing Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Analisis interaktif untuk menjawab pertanyaan bisnis pada periode {start_date} sampai {end_date}.</div>',
    unsafe_allow_html=True,
)


total_rentals = filtered_day["cnt"].sum()
avg_daily = filtered_day["cnt"].mean()
total_casual = filtered_day["casual"].sum()
total_registered = filtered_day["registered"].sum()

metric_cols = st.columns(4)
metric_cols[0].metric("Total penyewaan", f"{total_rentals:,.0f}")
metric_cols[1].metric("Rata-rata harian", f"{avg_daily:,.0f}")
metric_cols[2].metric("Total casual", f"{total_casual:,.0f}")
metric_cols[3].metric("Total registered", f"{total_registered:,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Q1 Musim dan Cuaca",
        "Q2 Pola Penyewaan per Jam",
        "Tren Umum",
        "Analisis Cluster",
    ]
)


with tab1:
    st.subheader("Pertanyaan 1")
    st.write(
        "Selama 1 Januari 2011 sampai 31 Desember 2012, kategori musim dan kondisi cuaca "
        "apa yang dapat digunakan untuk mengelompokkan tingkat permintaan penyewaan sepeda "
        "harian ke dalam pola permintaan tinggi dan rendah?"
    )

    season_order = ["Spring", "Summer", "Fall", "Winter"]
    weather_order = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Lebat"]

    season_stats = (
        filtered_day.groupby("season_label", observed=True)["cnt"].mean().reindex(season_order).dropna()
    )
    weather_stats = (
        filtered_day.groupby("weather_label", observed=True)["cnt"].mean().reindex(weather_order).dropna()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = make_highlight_colors(season_stats.reset_index(drop=True))
        bars = ax.bar(season_stats.index, season_stats.values, color=colors, edgecolor="white")
        add_bar_labels(ax, bars, offset=max(season_stats.values) * 0.02)
        ax.set_title("Rata-rata penyewaan harian per musim")
        ax.set_xlabel("Musim")
        ax.set_ylabel("Rata-rata penyewaan")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = make_highlight_colors(weather_stats.reset_index(drop=True))
        bars = ax.bar(weather_stats.index, weather_stats.values, color=colors, edgecolor="white")
        add_bar_labels(ax, bars, offset=max(weather_stats.values) * 0.02)
        ax.set_title("Rata-rata penyewaan harian per kondisi cuaca")
        ax.set_xlabel("Kondisi cuaca")
        ax.set_ylabel("Rata-rata penyewaan")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    temp_corr = filtered_day[["temp_actual", "cnt"]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(
        filtered_day["temp_actual"],
        filtered_day["cnt"],
        c=filtered_day["cnt"],
        cmap="Blues",
        alpha=0.65,
        s=35,
        edgecolors="white",
        linewidth=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="Jumlah penyewaan")
    ax.set_title("Hubungan temperatur dan jumlah penyewaan")
    ax.set_xlabel("Temperatur aktual (C)")
    ax.set_ylabel("Jumlah penyewaan")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_season = season_stats.idxmax()
    low_season = season_stats.idxmin()
    top_weather = weather_stats.idxmax()
    low_weather = weather_stats.idxmin()

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Insight Q1.</b> Pada filter saat ini, rata-rata penyewaan tertinggi muncul pada <b>{top_season}</b>
        dan terendah pada <b>{low_season}</b>. Dari sisi cuaca, kategori <b>{top_weather}</b> menghasilkan
        rata-rata penyewaan tertinggi, sedangkan <b>{low_weather}</b> menjadi yang terendah.
        Korelasi temperatur dengan jumlah penyewaan bernilai <b>{temp_corr:.2f}</b>, yang menunjukkan bahwa
        hari yang lebih hangat cenderung diikuti penyewaan yang lebih tinggi.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab2:
    st.subheader("Pertanyaan 2")
    st.write(
        "Selama 1 Januari 2011 sampai 31 Desember 2012, bagaimana hubungan antara jenis hari "
        "hari kerja atau hari libur dan jam penyewaan terhadap perubahan rata-rata jumlah "
        "penyewaan sepeda, serta kapan puncak permintaan terjadi?"
    )

    hourly_pattern = (
        filtered_hour.groupby(["hr", "day_type"], observed=True)["cnt"].mean().reset_index()
    )
    hourly_detail = (
        filtered_hour.groupby(["hr", "day_type"], observed=True)
        .agg(avg_casual=("casual", "mean"), avg_registered=("registered", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for day_type, color in [("Hari Kerja", LINE_WORKDAY), ("Hari Libur/Weekend", LINE_WEEKEND)]:
        data = hourly_pattern[hourly_pattern["day_type"] == day_type]
        if not data.empty:
            ax.plot(
                data["hr"],
                data["cnt"],
                marker="o",
                linewidth=2.5,
                markersize=5,
                label=day_type,
                color=color,
            )
    ax.set_title("Rata-rata penyewaan per jam")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-rata penyewaan")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        workday_data = hourly_detail[hourly_detail["day_type"] == "Hari Kerja"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            workday_data["hr"],
            workday_data["avg_casual"],
            marker="o",
            linewidth=2,
            label="Casual",
            color=MIN_COLOR,
        )
        ax.plot(
            workday_data["hr"],
            workday_data["avg_registered"],
            marker="o",
            linewidth=2,
            label="Registered",
            color=PRIMARY_COLOR,
        )
        ax.set_title("Hari kerja: casual vs registered")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Rata-rata penyewaan")
        ax.set_xticks(range(24))
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        weekend_data = hourly_detail[hourly_detail["day_type"] == "Hari Libur/Weekend"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            weekend_data["hr"],
            weekend_data["avg_casual"],
            marker="o",
            linewidth=2,
            label="Casual",
            color=MIN_COLOR,
        )
        ax.plot(
            weekend_data["hr"],
            weekend_data["avg_registered"],
            marker="o",
            linewidth=2,
            label="Registered",
            color=PRIMARY_COLOR,
        )
        ax.set_title("Hari libur/weekend: casual vs registered")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Rata-rata penyewaan")
        ax.set_xticks(range(24))
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    weekday_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    heatmap_data = filtered_hour.pivot_table(
        values="cnt", index="weekday_label", columns="hr", aggfunc="mean"
    ).reindex([day for day in weekday_order if day in filtered_hour["weekday_label"].unique()])

    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.heatmap(heatmap_data, cmap="Blues", ax=ax, cbar_kws={"label": "Rata-rata penyewaan"})
    ax.set_title("Heatmap rata-rata penyewaan per jam dan hari")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Hari")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    workday_peak_hour = (
        hourly_pattern[hourly_pattern["day_type"] == "Hari Kerja"].sort_values("cnt", ascending=False).iloc[0]
    )
    weekend_peak_hour = (
        hourly_pattern[hourly_pattern["day_type"] == "Hari Libur/Weekend"]
        .sort_values("cnt", ascending=False)
        .iloc[0]
    )

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Insight Q2.</b> Pada hari kerja, puncak penyewaan terjadi pada jam <b>{int(workday_peak_hour["hr"]):02d}.00</b>
        dengan rata-rata sekitar <b>{workday_peak_hour["cnt"]:,.0f}</b> penyewaan. Pada hari libur/weekend,
        puncak bergeser ke jam <b>{int(weekend_peak_hour["hr"]):02d}.00</b> dengan rata-rata sekitar
        <b>{weekend_peak_hour["cnt"]:,.0f}</b> penyewaan. Pola ini menegaskan bahwa hari kerja lebih kuat
        dipengaruhi kebutuhan komuter, sedangkan hari libur lebih terkonsentrasi pada jam santai siang hari.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab3:
    st.subheader("Tren Umum Penyewaan")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(filtered_day["dteday"], filtered_day["cnt"], alpha=0.2, color=PRIMARY_COLOR)
    ax.plot(filtered_day["dteday"], filtered_day["cnt"], linewidth=1.3, color=PRIMARY_COLOR)
    ax.set_title("Tren harian penyewaan sepeda")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah penyewaan")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        monthly_avg = filtered_day.groupby("mnth", observed=True)["cnt"].mean().reset_index()
        month_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = make_highlight_colors(monthly_avg["cnt"].reset_index(drop=True))
        bars = ax.bar(monthly_avg["mnth"], monthly_avg["cnt"], color=colors, edgecolor="white")
        add_bar_labels(ax, bars, offset=max(monthly_avg["cnt"]) * 0.02)
        ax.set_title("Rata-rata penyewaan per bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Rata-rata penyewaan")
        ax.set_xticks(monthly_avg["mnth"])
        ax.set_xticklabels([month_labels[i - 1] for i in monthly_avg["mnth"]])
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            [total_casual, total_registered],
            labels=["Casual", "Registered"],
            colors=[MIN_COLOR, PRIMARY_COLOR],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.03, 0),
        )
        ax.set_title("Proporsi pengguna casual vs registered")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    monthly_peak = monthly_avg.loc[monthly_avg["cnt"].idxmax(), "mnth"]
    monthly_low = monthly_avg.loc[monthly_avg["cnt"].idxmin(), "mnth"]

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Insight tren umum.</b> Rata-rata bulanan tertinggi pada filter saat ini muncul di bulan
        <b>{month_labels[int(monthly_peak) - 1]}</b>, sedangkan yang terendah terjadi pada bulan
        <b>{month_labels[int(monthly_low) - 1]}</b>. Secara komposisi, pengguna <b>registered</b>
        tetap mendominasi total penyewaan dibanding pengguna casual.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab4:
    st.subheader("Analisis Lanjutan: Clustering Binning")
    st.write(
        "Hari dikelompokkan menjadi empat cluster berdasarkan jumlah penyewaan: "
        "Rendah, Sedang, Tinggi, dan Sangat Tinggi."
    )

    cluster_order = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

    cluster_counts = (
        filtered_day["usage_cluster"].value_counts().reindex(cluster_order).dropna()
    )
    cluster_temp = (
        filtered_day.groupby("usage_cluster", observed=True)["temp_actual"]
        .mean()
        .reindex(cluster_order)
        .dropna()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = make_highlight_colors(cluster_counts.reset_index(drop=True))
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor="white")
        add_bar_labels(ax, bars, fmt="{:,.0f}", offset=max(cluster_counts.values) * 0.02)
        ax.set_title("Distribusi jumlah hari per cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah hari")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = make_highlight_colors(cluster_temp.reset_index(drop=True))
        bars = ax.bar(cluster_temp.index, cluster_temp.values, color=colors, edgecolor="white")
        add_bar_labels(ax, bars, fmt="{:.1f}", offset=max(cluster_temp.values) * 0.02)
        ax.set_title("Rata-rata temperatur per cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Temperatur aktual (C)")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    cluster_palette = {
        "Rendah": "#72B7B2",
        "Sedang": "#54A24B",
        "Tinggi": "#ECA82C",
        "Sangat Tinggi": "#E45756",
    }
    for cluster in cluster_order:
        data = filtered_day[filtered_day["usage_cluster"] == cluster]
        if not data.empty:
            ax.scatter(
                data["temp_actual"],
                data["cnt"],
                label=cluster,
                alpha=0.65,
                s=38,
                color=cluster_palette[cluster],
                edgecolors="white",
                linewidth=0.4,
            )
    ax.set_title("Temperatur vs penyewaan berdasarkan cluster")
    ax.set_xlabel("Temperatur aktual (C)")
    ax.set_ylabel("Jumlah penyewaan")
    ax.grid(alpha=0.25)
    ax.legend(title="Cluster")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    cluster_stats = (
        filtered_day.groupby("usage_cluster", observed=True)
        .agg(
            jumlah_hari=("cnt", "count"),
            rata_rata_penyewaan=("cnt", "mean"),
            rata_rata_temperatur=("temp_actual", "mean"),
            rata_rata_kelembapan=("hum", "mean"),
            rata_rata_casual=("casual", "mean"),
            rata_rata_registered=("registered", "mean"),
        )
        .reindex(cluster_order)
        .dropna(how="all")
    )
    st.dataframe(cluster_stats.round(1), use_container_width=True)

    hottest_cluster = cluster_temp.idxmax()
    coldest_cluster = cluster_temp.idxmin()

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Insight cluster.</b> Cluster dengan temperatur rata-rata tertinggi adalah <b>{hottest_cluster}</b>,
        sedangkan yang terendah adalah <b>{coldest_cluster}</b>. Pola ini memperkuat hasil EDA utama bahwa
        temperatur yang lebih hangat cenderung terkait dengan tingkat penyewaan yang lebih tinggi.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption("Bike Sharing Analytics Dashboard - dibuat dengan Streamlit")
