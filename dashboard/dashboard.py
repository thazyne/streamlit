import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from pathlib import Path

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="🚲 Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1a73e8, #34a853, #fbbc04);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stSelectbox > div > div {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# DATA LOADING
# ========================
@st.cache_data
def load_data():
    """Load and preprocess bike sharing datasets."""
    # Try multiple paths for flexibility
    base_paths = [
        Path(__file__).parent.parent,  # Parent of dashboard/
        Path(__file__).parent,          # dashboard/ itself
        Path(".")                       # Current directory
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

    if day_df is None:
        st.error("❌ File day.csv dan hour.csv tidak ditemukan! Pastikan file berada di folder yang benar.")
        st.stop()

    # Preprocessing
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weather_map = {1: 'Cerah', 2: 'Berawan', 3: 'Hujan Ringan', 4: 'Hujan Lebat'}
    year_map = {0: 2011, 1: 2012}
    weekday_map = {0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'}
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                 7: 'Jul', 8: 'Ags', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'}

    for df in [day_df, hour_df]:
        df['season_label'] = df['season'].map(season_map)
        df['weather_label'] = df['weathersit'].map(weather_map)
        df['year'] = df['yr'].map(year_map)
        df['weekday_label'] = df['weekday'].map(weekday_map)
        df['month_label'] = df['mnth'].map(month_map)
        df['day_type'] = df['workingday'].map({0: 'Hari Libur/Weekend', 1: 'Hari Kerja'})
        df['temp_actual'] = df['temp'] * 47 - 8

    # Clustering by binning
    q1 = day_df['cnt'].quantile(0.25)
    q2 = day_df['cnt'].quantile(0.50)
    q3 = day_df['cnt'].quantile(0.75)
    bins = [0, q1, q2, q3, day_df['cnt'].max() + 1]
    labels_cluster = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    day_df['usage_cluster'] = pd.cut(day_df['cnt'], bins=bins, labels=labels_cluster, include_lowest=True)

    return day_df, hour_df

day_df, hour_df = load_data()

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Bike-sharing_in_Cracow%2C_1.jpg/640px-Bike-sharing_in_Cracow%2C_1.jpg", use_container_width=True)
    st.title("🚲 Bike Sharing")
    st.markdown("---")

    # Date filter
    min_date = day_df['dteday'].min().date()
    max_date = day_df['dteday'].max().date()

    start_date = st.date_input("📅 Tanggal Mulai", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("📅 Tanggal Akhir", max_date, min_value=min_date, max_value=max_date)

    # Season filter
    season_options = ['Semua'] + list(day_df['season_label'].unique())
    selected_season = st.selectbox("🌿 Pilih Musim", season_options)

    # Year filter
    year_options = ['Semua'] + sorted(day_df['year'].unique().tolist())
    selected_year = st.selectbox("📆 Pilih Tahun", year_options)

    st.markdown("---")
    st.markdown("**Dataset:** Bike Sharing Dataset")
    st.markdown("**Sumber:** UCI Machine Learning Repository")

# Apply filters
filtered_day = day_df[
    (day_df['dteday'].dt.date >= start_date) &
    (day_df['dteday'].dt.date <= end_date)
]
filtered_hour = hour_df[
    (hour_df['dteday'].dt.date >= start_date) &
    (hour_df['dteday'].dt.date <= end_date)
]

if selected_season != 'Semua':
    filtered_day = filtered_day[filtered_day['season_label'] == selected_season]
    filtered_hour = filtered_hour[filtered_hour['season_label'] == selected_season]

if selected_year != 'Semua':
    filtered_day = filtered_day[filtered_day['year'] == selected_year]
    filtered_hour = filtered_hour[filtered_hour['year'] == selected_year]

# ========================
# MAIN CONTENT
# ========================
st.markdown('<p class="main-header">🚲 Bike Sharing Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown("")

# --- KEY METRICS ---
col1, col2, col3, col4 = st.columns(4)

total_rentals = filtered_day['cnt'].sum()
avg_daily = filtered_day['cnt'].mean()
total_casual = filtered_day['casual'].sum()
total_registered = filtered_day['registered'].sum()

with col1:
    st.metric("📊 Total Penyewaan", f"{total_rentals:,.0f}")
with col2:
    st.metric("📈 Rata-rata Harian", f"{avg_daily:,.0f}")
with col3:
    st.metric("🚶 Pengguna Casual", f"{total_casual:,.0f}")
with col4:
    st.metric("🏢 Pengguna Registered", f"{total_registered:,.0f}")

st.markdown("---")

# ========================
# TAB LAYOUT
# ========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tren & Distribusi",
    "🌤️ Musim & Cuaca",
    "⏰ Pola Jam",
    "🔬 Analisis Cluster"
])

# ========================
# TAB 1: TREN & DISTRIBUSI
# ========================
with tab1:
    st.subheader("📊 Tren Penyewaan Sepeda")

    # Daily trend
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(filtered_day['dteday'], filtered_day['cnt'], alpha=0.3, color='#3498DB')
    ax.plot(filtered_day['dteday'], filtered_day['cnt'], linewidth=0.8, color='#2980B9')
    ax.set_xlabel('Tanggal', fontsize=12)
    ax.set_ylabel('Jumlah Penyewaan', fontsize=12)
    ax.set_title('Tren Harian Penyewaan Sepeda', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)

    with col_a:
        # Monthly average
        monthly_avg = filtered_day.groupby('mnth')['cnt'].mean().reset_index()
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_month = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(monthly_avg)))
        bars = ax.bar(monthly_avg['mnth'], monthly_avg['cnt'], color=colors_month, edgecolor='white')
        ax.set_xlabel('Bulan', fontsize=12)
        ax.set_ylabel('Rata-rata Penyewaan', fontsize=12)
        ax.set_title('Rata-rata Penyewaan per Bulan', fontsize=14, fontweight='bold')
        ax.set_xticks(monthly_avg['mnth'])
        ax.set_xticklabels([month_labels[i-1] for i in monthly_avg['mnth']])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        # User type distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [total_casual, total_registered]
        labels = ['Casual', 'Registered']
        colors = ['#FF6B6B', '#4ECDC4']
        explode = (0.05, 0)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 13})
        ax.set_title('Proporsi Pengguna Casual vs Registered', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ========================
# TAB 2: MUSIM & CUACA
# ========================
with tab2:
    st.subheader("🌤️ Analisis Berdasarkan Musim dan Cuaca")

    col_c, col_d = st.columns(2)

    with col_c:
        # Season analysis
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        season_stats = filtered_day.groupby('season_label')['cnt'].mean()
        season_stats = season_stats.reindex([s for s in season_order if s in season_stats.index])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_season = ['#2ECC71', '#F39C12', '#E74C3C', '#3498DB']
        available_seasons = season_stats.index.tolist()
        c_map = {s: c for s, c in zip(season_order, colors_season)}
        bar_colors = [c_map[s] for s in available_seasons]

        bars = ax.bar(available_seasons, season_stats.values, color=bar_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, season_stats.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                    f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xlabel('Musim', fontsize=13)
        ax.set_ylabel('Rata-rata Penyewaan Harian', fontsize=13)
        ax.set_title('Rata-rata Penyewaan per Musim', fontsize=15, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        # Weather analysis
        weather_order = ['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Lebat']
        weather_stats = filtered_day.groupby('weather_label')['cnt'].mean()
        weather_stats = weather_stats.reindex([w for w in weather_order if w in weather_stats.index])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_weather = ['#F1C40F', '#95A5A6', '#3498DB', '#2C3E50']
        available_weather = weather_stats.index.tolist()
        w_map = {w: c for w, c in zip(weather_order, colors_weather)}
        bar_colors_w = [w_map[w] for w in available_weather]

        bars = ax.bar(available_weather, weather_stats.values, color=bar_colors_w, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, weather_stats.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                    f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xlabel('Kondisi Cuaca', fontsize=13)
        ax.set_ylabel('Rata-rata Penyewaan Harian', fontsize=13)
        ax.set_title('Rata-rata Penyewaan per Cuaca', fontsize=15, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Temperature vs Rentals scatter
    st.markdown("#### 🌡️ Hubungan Temperatur dengan Penyewaan")
    fig, ax = plt.subplots(figsize=(14, 5))
    scatter = ax.scatter(filtered_day['temp_actual'], filtered_day['cnt'],
                         c=filtered_day['cnt'], cmap='RdYlGn', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, label='Jumlah Penyewaan', ax=ax)
    ax.set_xlabel('Temperatur (°C)', fontsize=13)
    ax.set_ylabel('Jumlah Penyewaan', fontsize=13)
    ax.set_title('Hubungan Temperatur dan Jumlah Penyewaan', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========================
# TAB 3: POLA JAM
# ========================
with tab3:
    st.subheader("⏰ Pola Penyewaan Berdasarkan Jam")

    # Hourly pattern by day type
    hourly_pattern = filtered_hour.groupby(['hr', 'day_type'])['cnt'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for day_type in ['Hari Kerja', 'Hari Libur/Weekend']:
        if day_type in hourly_pattern['day_type'].values:
            data = hourly_pattern[hourly_pattern['day_type'] == day_type]
            ax.plot(data['hr'], data['cnt'], marker='o', linewidth=2.5, markersize=6, label=day_type)
    ax.set_xlabel('Jam', fontsize=13)
    ax.set_ylabel('Rata-rata Penyewaan', fontsize=13)
    ax.set_title('Pola Penyewaan per Jam: Hari Kerja vs Libur/Weekend', fontsize=15, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_e, col_f = st.columns(2)

    with col_e:
        # Casual vs Registered by hour (workday)
        hourly_detail = filtered_hour.groupby(['hr', 'day_type']).agg(
            avg_casual=('casual', 'mean'),
            avg_registered=('registered', 'mean')
        ).reset_index()

        workday_data = hourly_detail[hourly_detail['day_type'] == 'Hari Kerja']
        if not workday_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(workday_data['hr'], workday_data['avg_casual'], marker='s', linewidth=2, label='Casual', color='#FF6B6B')
            ax.plot(workday_data['hr'], workday_data['avg_registered'], marker='^', linewidth=2, label='Registered', color='#4ECDC4')
            ax.set_xlabel('Jam', fontsize=12)
            ax.set_ylabel('Rata-rata Penyewaan', fontsize=12)
            ax.set_title('Casual vs Registered (Hari Kerja)', fontsize=14, fontweight='bold')
            ax.set_xticks(range(0, 24))
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_f:
        # Casual vs Registered by hour (weekend)
        weekend_data = hourly_detail[hourly_detail['day_type'] == 'Hari Libur/Weekend']
        if not weekend_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(weekend_data['hr'], weekend_data['avg_casual'], marker='s', linewidth=2, label='Casual', color='#FF6B6B')
            ax.plot(weekend_data['hr'], weekend_data['avg_registered'], marker='^', linewidth=2, label='Registered', color='#4ECDC4')
            ax.set_xlabel('Jam', fontsize=12)
            ax.set_ylabel('Rata-rata Penyewaan', fontsize=12)
            ax.set_title('Casual vs Registered (Hari Libur/Weekend)', fontsize=14, fontweight='bold')
            ax.set_xticks(range(0, 24))
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Heatmap
    st.markdown("#### 🗓️ Heatmap Penyewaan per Jam dan Hari")
    weekday_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    heatmap_data = filtered_hour.pivot_table(values='cnt', index='weekday_label', columns='hr', aggfunc='mean')
    heatmap_data = heatmap_data.reindex([w for w in weekday_order if w in heatmap_data.index])

    if not heatmap_data.empty:
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, ax=ax,
                    cbar_kws={'label': 'Rata-rata Penyewaan'})
        ax.set_xlabel('Jam', fontsize=13)
        ax.set_ylabel('Hari', fontsize=13)
        ax.set_title('Heatmap Rata-rata Penyewaan per Jam dan Hari', fontsize=15, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ========================
# TAB 4: ANALISIS CLUSTER
# ========================
with tab4:
    st.subheader("🔬 Analisis Lanjutan: Clustering dengan Binning")
    st.markdown("""
    Hari-hari dikelompokkan menjadi **4 cluster** berdasarkan jumlah penyewaan menggunakan teknik **binning berbasis kuartil**:
    - **Rendah**: ≤ Q1
    - **Sedang**: Q1 - Q2 (Median)
    - **Tinggi**: Q2 - Q3
    - **Sangat Tinggi**: > Q3
    """)

    cluster_order = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    colors_cluster = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

    col_g, col_h = st.columns(2)

    with col_g:
        # Cluster distribution
        cluster_counts = filtered_day['usage_cluster'].value_counts().reindex(
            [c for c in cluster_order if c in filtered_day['usage_cluster'].values]
        ).dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        cl_colors = [colors_cluster[cluster_order.index(c)] for c in cluster_counts.index]
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=cl_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, cluster_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xlabel('Cluster', fontsize=13)
        ax.set_ylabel('Jumlah Hari', fontsize=13)
        ax.set_title('Distribusi Hari per Cluster', fontsize=15, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_h:
        # Temperature per cluster
        cluster_temp = filtered_day.groupby('usage_cluster', observed=True)['temp_actual'].mean()
        cluster_temp = cluster_temp.reindex([c for c in cluster_order if c in cluster_temp.index])

        fig, ax = plt.subplots(figsize=(10, 6))
        cl_colors_t = [colors_cluster[cluster_order.index(c)] for c in cluster_temp.index]
        bars = ax.bar(cluster_temp.index, cluster_temp.values, color=cl_colors_t, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, cluster_temp.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}°C', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xlabel('Cluster', fontsize=13)
        ax.set_ylabel('Rata-rata Temperatur (°C)', fontsize=13)
        ax.set_title('Rata-rata Temperatur per Cluster', fontsize=15, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Scatter plot
    st.markdown("#### 🔍 Temperatur vs Penyewaan per Cluster")
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, cluster in enumerate(cluster_order):
        data = filtered_day[filtered_day['usage_cluster'] == cluster]
        if not data.empty:
            ax.scatter(data['temp_actual'], data['cnt'], c=colors_cluster[i],
                       label=cluster, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Temperatur (°C)', fontsize=13)
    ax.set_ylabel('Jumlah Penyewaan', fontsize=13)
    ax.set_title('Temperatur vs Penyewaan Berdasarkan Cluster', fontsize=15, fontweight='bold')
    ax.legend(title='Cluster', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cluster statistics table
    st.markdown("#### 📋 Statistik per Cluster")
    cluster_stats = filtered_day.groupby('usage_cluster', observed=True).agg(
        Jumlah_Hari=('cnt', 'count'),
        Rata_Rata_Penyewaan=('cnt', 'mean'),
        Rata_Rata_Temperatur=('temp_actual', 'mean'),
        Rata_Rata_Kelembapan=('hum', 'mean'),
        Rata_Rata_Casual=('casual', 'mean'),
        Rata_Rata_Registered=('registered', 'mean')
    ).reindex([c for c in cluster_order if c in filtered_day['usage_cluster'].values])

    st.dataframe(
        cluster_stats.round(1).style.background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9rem;'>"
    "🚲 Bike Sharing Analytics Dashboard | Data: UCI ML Repository | "
    "Dibuat dengan ❤️ menggunakan Streamlit"
    "</div>",
    unsafe_allow_html=True
)
