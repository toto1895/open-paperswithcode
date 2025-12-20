"""
Papers with Code - Streamlit Dashboard
Optimized for performance with 1-hour cache, fast search, and rich card display.
Uses st_files_connection for GCS access with Streamlit secrets.
"""

import os
import math
import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from st_files_connection import FilesConnection

st.set_page_config(
    page_title="Open papers with Code",
    page_icon="📚",
    layout="wide",
)

# ---------- Config ----------
BUCKET_NAME = "paperswithcode"
PREFIX = ""
CACHE_TTL = 3600  # 1 hour in seconds


def get_connection():
    """Get the GCS connection instance using service account from secrets."""
    # Parse service account JSON from secrets
    service_account_json = st.secrets.secrets.service_account_json
    service_account_info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    
    # Return connection with credentials
    return st.connection('gcs', type=FilesConnection)


def list_parquet_files() -> list[tuple[str, datetime]]:
    """List parquet files in the bucket with their update times."""
    conn = get_connection()
    # Use the underlying fs to list files
    fs = conn.fs
    files = fs.ls(f"{BUCKET_NAME}/{PREFIX}")
    
    parquet_files = []
    for f in files:
        if f.endswith(".parquet"):
            try:
                info = fs.info(f)
                updated = info.get('updated', info.get('timeCreated'))
                if updated:
                    if isinstance(updated, str):
                        updated = pd.to_datetime(updated)
                    parquet_files.append((f, updated))
            except Exception:
                pass
    
    return parquet_files


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading papers...")
def load_data() -> tuple[pd.DataFrame, str | None]:
    """Load latest parquet from GCS with 1-hour cache. Returns (df, parquet_filename)."""
    # List and find latest parquet file
    parquet_files = list_parquet_files()
    
    if not parquet_files:
        return pd.DataFrame(), None

    # Sort by update time and get latest
    latest_path, _ = max(parquet_files, key=lambda x: x[1])
    parquet_filename = os.path.basename(latest_path)
    
    # Read parquet using connection
    conn = get_connection()
    df = conn.read(latest_path, input_format="parquet", ttl=CACHE_TTL)

    # Normalize dates once at load time
    for col in ["created", "updated", "repo_updated_at", "repo_created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    # Pre-compute numeric columns
    for col in ["stars_total", "stars", "stars_last_7d", "stars_last_30d", "forks", "watchers", "open_issues"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Compute effective stars: use 'stars' if available and non-zero, else 'stars_total'
    if "stars" in df.columns and "stars_total" in df.columns:
        df["_effective_stars"] = df["stars"].where(df["stars"] > 0, df["stars_total"])
    elif "stars" in df.columns:
        df["_effective_stars"] = df["stars"]
    elif "stars_total" in df.columns:
        df["_effective_stars"] = df["stars_total"]
    else:
        df["_effective_stars"] = 0

    # Pre-compute lowercase search columns for fast filtering
    def safe_join(x):
        """Safely convert list/array/scalar to string."""
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " ".join(str(v) for v in x)
        try:
            if pd.isna(x):
                return ""
        except (ValueError, TypeError):
            pass
        return str(x)

    def get_col_as_str(col_name: str) -> pd.Series:
        """Get column as lowercase string series."""
        if col_name not in df.columns:
            return pd.Series("", index=df.index)
        return df[col_name].apply(safe_join).str.lower()

    df["_search"] = (
        get_col_as_str("title") + " " +
        get_col_as_str("abstract") + " " +
        get_col_as_str("keywords") + " " +
        get_col_as_str("authors") + " " +
        get_col_as_str("repo_description") + " " +
        get_col_as_str("categories_full")
    )

    return df, parquet_filename


def parse_ingest_time_from_filename(filename: str | None) -> str:
    """Parse ingest time from parquet filename in format %Y%m%d%H.parquet."""
    if not filename:
        return "N/A"
    
    basename = os.path.basename(filename)
    
    if basename.endswith(".parquet"):
        date_str = basename[:-8]
    else:
        return "N/A"
    
    try:
        dt = datetime.strptime(date_str, "%Y%m%d%H")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return "N/A"


def filter_data(df: pd.DataFrame, search: str, language: str, sort_by: str, date_range: tuple) -> pd.DataFrame:
    """Apply filters with optimized operations."""
    mask = pd.Series(True, index=df.index)

    # Language filter
    if language != "All" and "language" in df.columns:
        mask &= df["language"].fillna("") == language

    # Search filter - check all words exist
    if search:
        words = search.lower().split()
        for word in words:
            mask &= df["_search"].str.contains(word, regex=False, na=False)

    # Date filter
    if date_range and "created" in df.columns:
        start, end = date_range
        created = df["created"].dt.date
        mask &= (created >= start) & (created <= end)

    filtered = df.loc[mask]

    # Sort - use _effective_stars for star-based sorting
    sort_map = {
        "Stars (Total)": ("_effective_stars", False),
        "Stars (7 days)": ("stars_last_7d", False),
        "Stars (30 days)": ("stars_last_30d", False),
        "Newest First": ("created", False),
        "Oldest First": ("created", True),
        "Forks": ("forks", False),
        "Watchers": ("watchers", False),
        "Repo Updated (Recent)": ("repo_updated_at", False),
        "Repo Updated (Oldest)": ("repo_updated_at", True),
    }
    col, asc = sort_map.get(sort_by, ("_effective_stars", False))
    if col in filtered.columns:
        filtered = filtered.sort_values(col, ascending=asc, na_position="last")

    return filtered.reset_index(drop=True)


# ---------- Theme Toggle ----------
if "theme" not in st.session_state:
    st.session_state.theme = "light"


def get_theme_styles():
    """Return CSS styles based on current theme."""
    if st.session_state.theme == "dark":
        return {
            "main_bg": "linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%)",
            "sidebar_bg": "rgba(13, 17, 23, 0.95)",
            "sidebar_border": "rgba(56, 189, 248, 0.1)",
            "card_bg": "linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%)",
            "card_border": "rgba(56, 189, 248, 0.12)",
            "card_border_hover": "rgba(56, 189, 248, 0.5)",
            "card_shadow": "0 20px 50px rgba(56, 189, 248, 0.15), 0 0 30px rgba(56, 189, 248, 0.05)",
            "title_color": "#f1f5f9",
            "text_color": "#cbd5e1",
            "text_muted": "#94a3b8",
            "text_faint": "#64748b",
            "stat_value": "#e2e8f0",
            "header_bg": "linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)",
            "header_border": "rgba(56, 159, 228, 0.15)",
            "badge_lang_bg": "linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(56, 189, 248, 0.1))",
            "badge_lang_border": "rgba(56, 189, 248, 0.4)",
            "badge_lang_color": "#7dd3fc",
            "badge_cat_bg": "rgba(139, 92, 246, 0.15)",
            "badge_cat_border": "rgba(139, 92, 246, 0.3)",
            "badge_cat_color": "#c4b5fd",
            "keyword_bg": "rgba(251, 191, 36, 0.1)",
            "keyword_border": "rgba(251, 191, 36, 0.25)",
            "keyword_color": "#fcd34d",
            "link_paper_bg": "linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(56, 189, 248, 0.08))",
            "link_paper_border": "rgba(56, 189, 248, 0.3)",
            "link_paper_color": "#7dd3fc",
            "link_github_bg": "linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.08))",
            "link_github_border": "rgba(139, 92, 246, 0.3)",
            "link_github_color": "#c4b5fd",
            "ingest_bg": "rgba(34, 197, 94, 0.1)",
            "ingest_border": "rgba(34, 197, 94, 0.3)",
            "ingest_color": "#86efac",
            "license_bg": "rgba(34, 197, 94, 0.15)",
            "license_border": "rgba(34, 197, 94, 0.3)",
            "license_color": "#86efac",
        }
    else:
        return {
            "main_bg": "linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)",
            "sidebar_bg": "rgba(255, 255, 255, 0.98)",
            "sidebar_border": "rgba(59, 130, 246, 0.15)",
            "card_bg": "linear-gradient(145deg, rgba(255, 255, 255, 1) 0%, rgba(248, 250, 252, 0.95) 100%)",
            "card_border": "rgba(203, 213, 225, 0.6)",
            "card_border_hover": "rgba(59, 130, 246, 0.6)",
            "card_shadow": "0 20px 50px rgba(59, 130, 246, 0.15), 0 8px 25px rgba(0, 0, 0, 0.1)",
            "title_color": "#0f172a",
            "text_color": "#334155",
            "text_muted": "#475569",
            "text_faint": "#64748b",
            "stat_value": "#1e293b",
            "header_bg": "linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)",
            "header_border": "rgba(59, 130, 246, 0.25)",
            "badge_lang_bg": "linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(59, 130, 246, 0.1))",
            "badge_lang_border": "rgba(59, 130, 246, 0.5)",
            "badge_lang_color": "#1d4ed8",
            "badge_cat_bg": "rgba(139, 92, 246, 0.15)",
            "badge_cat_border": "rgba(139, 92, 246, 0.45)",
            "badge_cat_color": "#6d28d9",
            "keyword_bg": "rgba(245, 158, 11, 0.15)",
            "keyword_border": "rgba(245, 158, 11, 0.45)",
            "keyword_color": "#b45309",
            "link_paper_bg": "linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08))",
            "link_paper_border": "rgba(59, 130, 246, 0.45)",
            "link_paper_color": "#1d4ed8",
            "link_github_bg": "linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.08))",
            "link_github_border": "rgba(139, 92, 246, 0.45)",
            "link_github_color": "#6d28d9",
            "ingest_bg": "rgba(22, 163, 74, 0.12)",
            "ingest_border": "rgba(22, 163, 74, 0.4)",
            "ingest_color": "#15803d",
            "license_bg": "rgba(22, 163, 74, 0.12)",
            "license_border": "rgba(22, 163, 74, 0.4)",
            "license_color": "#15803d",
        }


# ---------- Styles ----------
theme = get_theme_styles()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500;600;700&display=swap');

.main {{ background: {theme['main_bg']}; }}
.stApp {{ background: transparent; }}

[data-testid="stSidebar"] {{
    background: {theme['sidebar_bg']};
    border-right: 1px solid {theme['sidebar_border']};
}}

.header-container {{
    background: {theme['header_bg']};
    border: 1px solid {theme['header_border']};
    border-radius: 0.75rem;
    padding: 0.6rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}}

.header-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}

.header-subtitle {{
    font-family: 'Outfit', sans-serif;
    color: {theme['text_muted']};
    font-size: 0.85rem;
    margin: 0;
}}

/* Mobile responsive styles */
@media (max-width: 768px) {{
    .header-container {{
        flex-direction: column;
        align-items: flex-start;
        padding: 0.8rem 1rem;
        gap: 0.3rem;
    }}
    
    .header-title {{
        font-size: 1.25rem;
    }}
    
    .header-subtitle {{
        font-size: 0.75rem;
    }}
    
    .card {{
        padding: 1rem;
        min-height: auto;
        border-radius: 0.75rem;
    }}
    
    .card-title {{
        font-size: 1rem;
        -webkit-line-clamp: 3;
    }}
    
    .card-stats {{
        gap: 0.5rem;
        flex-wrap: wrap;
    }}
    
    .stat {{
        font-size: 0.68rem;
    }}
    
    .card-abstract {{
        font-size: 0.8rem;
        -webkit-line-clamp: 3;
    }}
    
    .card-meta {{
        font-size: 0.7rem;
    }}
    
    .card-authors {{
        font-size: 0.7rem;
    }}
    
    .card-badges {{
        gap: 0.3rem;
    }}
    
    .badge {{
        font-size: 0.6rem;
        padding: 0.2rem 0.4rem;
    }}
    
    .card-keywords {{
        gap: 0.25rem;
    }}
    
    .keyword {{
        font-size: 0.58rem;
        padding: 0.15rem 0.4rem;
    }}
    
    .card-links {{
        flex-wrap: wrap;
        gap: 0.4rem;
    }}
    
    .card-link {{
        font-size: 0.68rem;
        padding: 0.35rem 0.7rem;
    }}
    
    .results-count {{
        font-size: 0.75rem;
    }}
    
    .ingest-badge {{
        font-size: 0.65rem;
        padding: 0.3rem 0.6rem;
    }}
}}

@media (max-width: 480px) {{
    .header-container {{
        padding: 0.6rem 0.8rem;
    }}
    
    .header-title {{
        font-size: 1.1rem;
    }}
    
    .header-subtitle {{
        font-size: 0.7rem;
    }}
    
    .card {{
        padding: 0.8rem;
    }}
    
    .card-title {{
        font-size: 0.95rem;
    }}
    
    .card-stats {{
        padding: 0.5rem 0;
    }}
    
    .stat {{
        font-size: 0.65rem;
    }}
    
    .card-abstract {{
        font-size: 0.78rem;
    }}
}}

.card {{
    background: {theme['card_bg']};
    border: 1px solid {theme['card_border']};
    border-radius: 1rem;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    min-height: 320px;
    display: flex;
    flex-direction: column;
}}

.card:hover {{
    border-color: {theme['card_border_hover']};
    box-shadow: {theme['card_shadow']};
    transform: translateY(-2px);
}}

.card-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.7rem;
}}

.badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.badge-lang {{
    background: {theme['badge_lang_bg']};
    border: 1px solid {theme['badge_lang_border']};
    color: {theme['badge_lang_color']};
}}

.badge-cat {{
    background: {theme['badge_cat_bg']};
    border: 1px solid {theme['badge_cat_border']};
    color: {theme['badge_cat_color']};
}}

.badge-license {{
    background: {theme['license_bg']};
    border: 1px solid {theme['license_border']};
    color: {theme['license_color']};
}}

.card-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: {theme['title_color']};
    line-height: 1.4;
    margin-bottom: 0.6rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.card-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
    padding: 0.6rem 0;
    border-top: 1px solid {theme['card_border']};
    border-bottom: 1px solid {theme['card_border']};
}}

.stat {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {theme['text_muted']};
    display: flex;
    align-items: center;
    gap: 0.3rem;
}}

.stat-value {{
    color: {theme['stat_value']};
    font-weight: 600;
}}

.stat-highlight {{
    color: #f59e0b;
}}

.card-meta {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    color: {theme['text_faint']};
    margin-bottom: 0.6rem;
}}

.card-abstract {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: {theme['text_color']};
    line-height: 1.55;
    flex-grow: 1;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.8rem;
}}

.card-authors {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    color: {theme['text_faint']};
    margin-bottom: 0.6rem;
    display: -webkit-box;
    -webkit-line-clamp: 1;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.card-keywords {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.7rem;
}}

.keyword {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: {theme['keyword_bg']};
    border: 1px solid {theme['keyword_border']};
    color: {theme['keyword_color']};
}}

.card-links {{
    display: flex;
    gap: 0.6rem;
    margin-top: auto;
    padding-top: 0.7rem;
}}

.card-link {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.4rem 0.9rem;
    border-radius: 6px;
    text-decoration: none;
    transition: all 0.2s ease;
}}

.link-paper {{
    background: {theme['link_paper_bg']};
    border: 1px solid {theme['link_paper_border']};
    color: {theme['link_paper_color']};
}}

.link-paper:hover {{
    filter: brightness(1.15);
}}

.link-github {{
    background: {theme['link_github_bg']};
    border: 1px solid {theme['link_github_border']};
    color: {theme['link_github_color']};
}}

.link-github:hover {{
    filter: brightness(1.15);
}}

.results-count {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {theme['text_faint']};
    padding: 0.5rem 0;
    margin-bottom: 1rem;
}}

.results-count span {{
    color: #2563eb;
    font-weight: 600;
}}

.ingest-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    background: {theme['ingest_bg']};
    border: 1px solid {theme['ingest_border']};
    color: {theme['ingest_color']};
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}}

/* Make columns fill width */
[data-testid="column"] > div {{
    width: 100%;
}}

/* ========== RESPONSIVE DESIGN FOR MOBILE ========== */
@media screen and (max-width: 768px) {{
    .header-container {{
        flex-direction: column;
        align-items: flex-start;
        padding: 0.8rem 1rem;
        gap: 0.3rem;
    }}
    
    .header-title {{
        font-size: 1.3rem;
    }}
    
    .header-subtitle {{
        font-size: 0.75rem;
    }}
    
    .card {{
        padding: 1rem;
        min-height: auto;
        border-radius: 0.75rem;
    }}
    
    .card-title {{
        font-size: 1rem;
        -webkit-line-clamp: 3;
    }}
    
    .card-stats {{
        gap: 0.5rem;
        padding: 0.5rem 0;
    }}
    
    .stat {{
        font-size: 0.68rem;
    }}
    
    .card-meta {{
        font-size: 0.7rem;
    }}
    
    .card-abstract {{
        font-size: 0.8rem;
        -webkit-line-clamp: 3;
    }}
    
    .card-authors {{
        font-size: 0.7rem;
    }}
    
    .card-badges {{
        gap: 0.3rem;
    }}
    
    .badge {{
        font-size: 0.58rem;
        padding: 0.2rem 0.45rem;
    }}
    
    .card-keywords {{
        gap: 0.25rem;
    }}
    
    .keyword {{
        font-size: 0.55rem;
        padding: 0.15rem 0.4rem;
    }}
    
    .card-links {{
        flex-wrap: wrap;
        gap: 0.5rem;
    }}
    
    .card-link {{
        font-size: 0.65rem;
        padding: 0.35rem 0.7rem;
        flex: 1;
        text-align: center;
        min-width: 80px;
    }}
    
    .results-count {{
        font-size: 0.75rem;
    }}
    
    .ingest-badge {{
        font-size: 0.65rem;
        padding: 0.3rem 0.6rem;
    }}
}}

@media screen and (max-width: 480px) {{
    .header-container {{
        padding: 0.6rem 0.8rem;
    }}
    
    .header-title {{
        font-size: 1.1rem;
    }}
    
    .header-subtitle {{
        font-size: 0.7rem;
    }}
    
    .card {{
        padding: 0.8rem;
        margin-bottom: 0.75rem;
    }}
    
    .card-title {{
        font-size: 0.95rem;
    }}
    
    .card-stats {{
        flex-wrap: wrap;
        gap: 0.4rem;
    }}
    
    .stat {{
        font-size: 0.62rem;
    }}
    
    .card-meta {{
        font-size: 0.65rem;
        line-height: 1.4;
    }}
    
    .card-abstract {{
        font-size: 0.75rem;
        -webkit-line-clamp: 2;
    }}
    
    .badge {{
        font-size: 0.52rem;
        padding: 0.15rem 0.35rem;
    }}
    
    .keyword {{
        font-size: 0.5rem;
        padding: 0.12rem 0.3rem;
    }}
    
    .card-link {{
        font-size: 0.6rem;
        padding: 0.3rem 0.5rem;
    }}
}}
</style>
""", unsafe_allow_html=True)


# ---------- Header ----------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">📚 Open papers with Code</h1>
    <p class="header-subtitle">Discover research papers with GitHub repositories</p>
</div>
""", unsafe_allow_html=True)


# ---------- Load Data ----------
df, parquet_filename = load_data()

if df.empty:
    st.error(f"No data found in gs://{BUCKET_NAME}/{PREFIX}")
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme_toggle = st.toggle(
        "Light Mode",
        value=(st.session_state.theme == "light"),
        help="Toggle between dark and light themes"
    )
    if theme_toggle and st.session_state.theme != "light":
        st.session_state.theme = "light"
        st.rerun()
    elif not theme_toggle and st.session_state.theme != "dark":
        st.session_state.theme = "dark"
        st.rerun()

    st.markdown("---")

    # Display latest ingest time from parquet filename
    ingest_str = parse_ingest_time_from_filename(parquet_filename)
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span class="ingest-badge">🔄 Last Ingest: {ingest_str}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Search & Filter")

    search = st.text_input(
        "Search",
        placeholder="Title, abstract, keywords, authors...",
        help="Search across title, abstract, keywords, authors, and repo description"
    )

    languages = ["All"] + sorted(df["language"].dropna().unique().tolist()) if "language" in df.columns else ["All"]
    language = st.selectbox("Language", languages)

    st.markdown("### 📊 Sort By")
    sort_by = st.selectbox(
        "Sort",
        ["Stars (Total)", "Newest First", "Oldest First", "Forks", "Watchers", "Repo Updated (Recent)", "Repo Updated (Oldest)"],
        label_visibility="collapsed"
    )

    st.markdown("### 📅 Date Range")
    if "created" in df.columns and df["created"].notna().any():
        date_min = df["created"].min().date()
        date_max = pd.Timestamp.today()
        default_start = max(date_min, (datetime.now() - timedelta(days=60)).date())
        date_range = st.date_input(
            "Created between",
            value=(default_start, date_max),
            min_value=date_min,
            max_value=date_max,
            label_visibility="collapsed"
        )
        date_range = tuple(date_range) if len(date_range) == 2 else None
    else:
        date_range = None


# ---------- Filter Data ----------
filtered = filter_data(df, search, language, sort_by, date_range)

st.markdown(f'<p class="results-count"><span>{len(filtered):,}</span> papers found</p>', unsafe_allow_html=True)

if filtered.empty:
    st.info("No papers match your filters. Try adjusting your search criteria.")
    st.stop()


# ---------- Pagination ----------
page_size = 48
total_pages = max(1, math.ceil(len(filtered) / page_size))

if "page" not in st.session_state:
    st.session_state.page = 1

page = st.session_state.page
start = (page - 1) * page_size
page_df = filtered.iloc[start:start + page_size]


# ---------- Render Cards ----------
def safe(val, default=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val[:5]) if val else default
    return str(val)


def render_card(row):
    title = safe(row.get("title"), "Untitled")
    abstract = safe(row.get("abstract"))
    abstract = (abstract[:280] + "...") if len(abstract) > 280 else abstract

    lang = safe(row.get("language"))
    categories = safe(row.get("categories_full"))
    license_val = safe(row.get("license"))
    authors = safe(row.get("authors"))
    keywords = row.get("keywords")

    stars = int(row.get("_effective_stars", 0) or 0)
    stars_7d = int(row.get("stars_last_7d", 0) or 0)
    stars_30d = int(row.get("stars_last_30d", 0) or 0)
    forks = int(row.get("forks", 0) or 0)
    watchers = int(row.get("watchers", 0) or 0)

    created = row.get("created")
    created_str = created.strftime("%Y-%m-%d") if pd.notna(created) else ""

    repo_created = row.get("repo_created_at")
    repo_created_str = repo_created.strftime("%Y-%m-%d") if pd.notna(repo_created) else ""

    repo_updated = row.get("repo_updated_at")
    repo_updated_str = repo_updated.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(repo_updated) else ""

    paper_url = safe(row.get("url"))
    github_url = safe(row.get("github_link"))
    arxiv_id = safe(row.get("id"))

    badges = []
    if lang:
        badges.append(f'<span class="badge badge-lang">{lang}</span>')
    if categories:
        for cat in categories.split(",")[:2]:
            badges.append(f'<span class="badge badge-cat">{cat.strip()}</span>')

    links = []
    if paper_url:
        links.append(f'<a href="{paper_url}" target="_blank" class="card-link link-paper">📄 Paper</a>')
    if github_url:
        links.append(f'<a href="{github_url}" target="_blank" class="card-link link-github">⚡ GitHub</a>')

    keywords_html = ""
    if keywords is not None:
        kw_list = []
        if isinstance(keywords, (list, tuple)):
            kw_list = [str(k) for k in keywords[:8]]
        else:
            try:
                if hasattr(keywords, '__iter__') and not isinstance(keywords, str):
                    kw_list = [str(k) for k in list(keywords)[:8]]
                elif not pd.isna(keywords):
                    kw_list = [str(keywords)]
            except (ValueError, TypeError):
                pass
        if kw_list:
            kw_tags = "".join(f'<span class="keyword">{kw}</span>' for kw in kw_list)
            keywords_html = f'<div class="card-keywords">{kw_tags}</div>'

    return f"""
<div class="card">
    <div class="card-badges">{"".join(badges)}</div>
    <div class="card-title">{title}</div>
    <div class="card-stats">
        <span class="stat"><span class="stat-highlight">⭐</span> <span class="stat-value">{stars:,}</span></span>
        <span class="stat">🍴 <span class="stat-value">{forks:,}</span></span>
        <span class="stat">👁 <span class="stat-value">{watchers:,}</span></span>
    </div>
    <div class="card-meta">
        {arxiv_id} · 📄 {created_str}
        {f' · 🚀 {repo_created_str}' if repo_created_str else ''}
        {f' · 🔄 {repo_updated_str}' if repo_updated_str else ''}
    </div>
    {f'<div class="card-authors">👤 {authors}</div>' if authors else ''}
    <div class="card-abstract">{abstract}</div>
    {keywords_html}
    <div class="card-links">{"".join(links)}</div>
</div>
"""


# Render grid
cols = st.columns(1, gap="medium")
for i, (_, row) in enumerate(page_df.iterrows()):
    with cols[i % 1]:
        st.markdown(render_card(row), unsafe_allow_html=True)


# ---------- Pagination Controls ----------
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.page,
        label_visibility="collapsed",
        key="page_input"
    )
    st.session_state.page = page


# Footer pagination info
st.markdown(f"""
<p style="text-align: center; color: {theme['text_faint']}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-top: 2rem;">
    Page {page} of {total_pages} · Showing {start + 1}-{min(start + page_size, len(filtered))} of {len(filtered):,} papers
</p>
""", unsafe_allow_html=True)