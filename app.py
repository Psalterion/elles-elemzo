import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings

# --- ALAP BEÁLLÍTÁSOK ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Nagy Dűr Farrowing Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- STÍLUS (CSS) ---
st.markdown("""
<style>
    .stDownloadButton > button {
        background-color: #f0f2f6;
        color: #31333F;
        border: 1px solid #d6d6d8;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐖 Nagy Dűr Farrowing Analyzer")
st.markdown("---")

# --- FÜGGVÉNYEK ---

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, sep=None, engine='python')
            except:
                file.seek(0)
                df = pd.read_csv(file, sep=';')
        else:
            df = pd.read_excel(file)

        if len(df.columns) > 0 and str(df.columns[0]).startswith('Unnamed'):
            df = df.iloc[:, 1:]

        df.columns = [str(c).strip() for c in df.columns]

        # Dátum keresés
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'dátum' in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Week'] = df[date_col].dt.isocalendar().week
        else:
            week_col = next((c for c in df.columns if 'week' in c.lower() or 'hét' in c.lower()), None)
            if week_col:
                df['Week'] = df[week_col]

        # Szótár
        mapping = {
            'Sow name': ['Sow name', 'Koca', 'Anya', 'Sow'],
            'Parity': ['Parity', 'Fialás', 'Ellés', 'Sorszám'],
            'Breed': ['Breed', 'Fajta', 'Genetika'],
            'Inseminator': ['Inseminator Name', 'Inszemináló', 'Rakó'],
            'Semen': ['Semen batches', 'Semen', 'Mag', 'Termékenyítőanyag', 'Batch'],
            'Liveborn': ['Liveborn', 'Élve', 'Live'],
            'Stillborn': ['Stillborn', 'Halva', 'Still'],
            'Totalborn': ['Totalborn', 'Összesen']
        }
        
        new_cols = {}
        for target, alternatives in mapping.items():
            for alt in alternatives:
                match = [c for c in df.columns if c.lower() == alt.lower()]
                if match:
                    new_cols[match[0]] = target
                    break
        
        return df.rename(columns=new_cols)

    except Exception as e:
        return None

def download_chart(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# --- FELIRATOZÓK ---
def force_text_on_bars(ax, bars, is_percent=False, color='black'):
    for bar in bars:
        height = bar.get_height()
        if height > 0.05: # Csak ha látható mennyiség
            label = f"{height:.1f}%" if is_percent else f"{height:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', color=color, fontweight='bold', fontsize=9)

def force_text_with_counts(ax, x_data, y_data, counts, color, position='top'):
    for x, y, c in zip(x_data, y_data, counts):
        if pd.notnull(y) and c > 0:
            label = f"{y:.1f}\n({int(c)})"
            xytext = (0, 15) if position == 'top' else (0, -30)
            va = 'bottom' if position == 'top' else 'top'
            ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, ha='center', va=va, color=color, fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2))

# --- FŐ PROGRAM ---
uploaded_files = st.sidebar.file_uploader("📂 Upload Files", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        data = load_data(f)
        if data is not None: dfs.append(data)
    
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        
        needed = ['Parity', 'Liveborn', 'Week']
        if all(c in df_all.columns for c in needed):
            
            # Adattisztítás
            for col in ['Parity', 'Liveborn', 'Stillborn', 'Week']:
                if col in df_all.columns:
                    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
            
            df_clean = df_all.dropna(subset=['Parity', 'Liveborn', 'Week']).copy()
            df_clean['Parity'] = df_clean['Parity'].astype(int)
            df_clean['Week'] = df_clean['Week'].astype(int)
            df_clean['Is_Gilt'] = df_clean['Parity'] == 1
            
            if 'Stillborn' in df_clean.columns:
                df_clean['Totalborn'] = df_clean['Liveborn'] + df_clean['Stillborn']
            else:
                df_clean['Totalborn'] = df_clean['Liveborn']
                df_clean['Stillborn'] = 0

            unique_weeks = df_clean['Week'].nunique()
            
            # --- KPI SÁV (ÚJ: Koca/Süldő bontás) ---
            # 5 oszlopot hozunk létre
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            
            # Koca/Süldő adatok kiszámolása
            sows_data = df_clean[~df_clean['Is_Gilt']]
            gilts_data = df_clean[df_clean['Is_Gilt']]
            
            avg_live_sow = sows_data['Liveborn'].mean() if not sows_data.empty else 0
            avg_live_gilt = gilts_data['Liveborn'].mean() if not gilts_data.empty else 0
            
            with kpi1:
                st.metric("Total Farrowings", f"{len(df_clean)}", delta_color="off")
            with kpi2:
                st.metric("Sow Avg Live", f"{avg_live_sow:.2f}", border=True)
            with kpi3:
                st.metric("Gilt Avg Live", f"{avg_live_gilt:.2f}", border=True)
            with kpi4:
                sb_rate = df_clean['Stillborn'].sum() / df_clean['Totalborn'].sum() * 100
                st.metric("Avg Stillborn Rate", f"{sb_rate:.1f}%", border=True)
            with kpi5:
                st.metric("Total Piglets (Live)", f"{int(df_clean['Liveborn'].sum())}", border=True)

            st.markdown("---")
            
            # --- TABOK ---
            tab1_name = "🔍 Weekly Deep Dive" if unique_weeks == 1 else "📈 Weekly Trends"
            tab1, tab2, tab3 = st.tabs([tab1_name, "📊 Production Curve", "🧬 Breed Analysis"])
            
            with tab1:
                if unique_weeks > 1:
                    # --- TREND ---
                    def calc(x):
                        sows = x[~x['Is_Gilt']]
                        gilts = x[x['Is_Gilt']]
                        sow_tb = sows['Totalborn'].sum()
                        gilt_tb = gilts['Totalborn'].sum()
                        return pd.Series({
                            'L_Sow': sows['Liveborn'].mean(), 'L_Gilt': gilts['Liveborn'].mean(),
                            'C_Sow': len(sows), 'C_Gilt': len(gilts),
                            'S_Sow': (sows['Stillborn'].sum()/sow_tb*100) if sow_tb > 0 else 0,
                            'S_Gilt': (gilts['Stillborn'].sum()/gilt_tb*100) if gilt_tb > 0 else 0
                        })
                    
                    weekly = df_clean.groupby('Week').apply(calc).reset_index()
                    weekly['Sort'] = weekly['Week'].apply(lambda w: w if w >= 50 else w + 100)
                    weekly = weekly.sort_values('Sort')
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                    x_lbl = weekly['Week'].astype(int).astype(str)
                    x_pos = range(len(weekly))

                    # 1. Grafikon: Élveszületés
                    ax1.plot(x_pos, weekly['L_Sow'], 'go-', label='Sow (2+)')
                    ax1.plot(x_pos, weekly['L_Gilt'], 's-', color='orange', label='Gilt (1)')
                    force_text_with_counts(ax1, x_pos, weekly['L_Sow'], weekly['C_Sow'], 'green', position='top')
                    force_text_with_counts(ax1, x_pos, weekly['L_Gilt'], weekly['C_Gilt'], 'darkorange', position='bottom')
                    # --- JAVÍTÁS KEZDETE ---
                    # 1. Kicseréljük a NaN (hiányzó) értékeket 0-ra a számítás idejére
                    max_sow = weekly['L_Sow'].fillna(0).max()
                    max_gilt = weekly['L_Gilt'].fillna(0).max()

                    # 2. Kiszámoljuk a maximumot (ha mindkettő 0, akkor adunk egy alap 10-es magasságot, hogy ne legyen hiba)
                    top_limit = max(max_sow, max_gilt)
                    if top_limit == 0 or pd.isna(top_limit):
                    top_limit = 10 # Alapértelmezett magasság, ha üres az adat

                    # 3. Beállítjuk a limitet
                    ax1.set_ylim(0, top_limit * 1.4)
                    # --- JAVÍTÁS VÉGE ---
                    ax1.set_xticks(x_pos); ax1.set_xticklabels(x_lbl); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
                    ax1.set_title("Weekly Liveborn Trend", fontsize=11)

                    # 2. Grafikon: Veszteség (%)
                    ax2.bar(x_pos, weekly['S_Sow'], color='darkred', alpha=0.6, label='Sow Loss %')
                    ax2.bar(x_pos, weekly['S_Gilt'], color='orange', alpha=0.6, label='Gilt Loss %', bottom=weekly['S_Sow'])
                    y_max_loss = (weekly['S_Sow'] + weekly['S_Gilt']).max()
                    ax2.set_ylim(0, y_max_loss * 1.3)
                    ax2.set_xticks(x_pos); ax2.set_xticklabels(x_lbl); ax2.legend(); ax2.set_title("Loss Trend (%)", fontsize=11)
                    
                    st.pyplot(fig, use_container_width=True)
                    st.download_button("📥 Download Chart (PNG)", download_chart(fig, "weekly_trend.png"), "weekly_trend.png", "image/png")

                else:
                    # --- MÉLYELEMZÉS (1 hét) ---
                    st.info(f"Detailed Analysis for Week {df_clean['Week'].iloc[0]}")
                    
                    # Összesítő Stacked Bar
                    categories = ['Sow (2+)', 'Gilt (1)']
                    live_vals = [sows_data['Liveborn'].mean() if not sows_data.empty else 0, gilts_data['Liveborn'].mean() if not gilts_data.empty else 0]
                    still_vals = [sows_data['Stillborn'].mean() if not sows_data.empty else 0, gilts_data['Stillborn'].mean() if not gilts_data.empty else 0]
                    total_vals = [l + s for l, s in zip(live_vals, still_vals)]
                    counts = [len(sows_data), len(gilts_data)]
                    
                    fig_main, ax_main = plt.subplots(figsize=(8, 6))
                    p1 = ax_main.bar(categories, live_vals, color='green', label='Live', width=0.5)
                    p2 = ax_main.bar(categories, still_vals, bottom=live_vals, color='salmon', label='Still', width=0.5)
                    
                    ax_main.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f')
                    for rect in p2:
                        height = rect.get_height()
                        if height > 0.1:
                            ax_main.text(rect.get_x() + rect.get_width()/2., rect.get_y() + height/2.,
                                        f"{height:.1f}", ha='center', va='center', color='black', fontsize=9)
                    
                    for i, total in enumerate(total_vals):
                        if total > 0:
                            ax_main.text(i, total + 0.1, f"Total: {total:.1f}\n({counts[i]})", 
                                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
                    ax_main.set_ylim(0, max(total_vals) * 1.35 if total_vals else 10)
                    ax_main.set_title("Results: Live + Still = Total Born", fontsize=12)
                    ax_main.legend(loc='upper right')
                    
                    st.pyplot(fig_main, use_container_width=True)
                    st.download_button("📥 Download Summary Chart", download_chart(fig_main, "summary_chart.png"), "summary_chart.png", "image/png")
                    
                    st.divider()

                    col1, col2 = st.columns(2)
                    
                    def plot_top_performance(data, group_col, title, ax):
                        stats = data.groupby(group_col).agg({'Liveborn':'mean', 'Parity':'count'}).reset_index()
                        stats = stats[stats['Parity'] >= 3].sort_values('Liveborn', ascending=True).tail(10)
                        
                        if not stats.empty:
                            bars = ax.barh(stats[group_col].astype(str), stats['Liveborn'], color='royalblue')
                            for i, (idx, row) in enumerate(stats.iterrows()):
                                ax.text(row['Liveborn'], i, f" {row['Liveborn']:.1f} ({int(row['Parity'])})", 
                                       va='center', fontweight='bold', fontsize=9)
                            ax.set_xlim(0, stats['Liveborn'].max() * 1.35) 
                            ax.set_title(title, fontsize=11)
                        else:
                            ax.text(0.5, 0.5, "Not enough data", ha='center')
                            ax.set_title(title)

                    with col1:
                        if 'Inseminator' in df_clean.columns:
                            fig_ins, ax_ins = plt.subplots(figsize=(6, 6))
                            plot_top_performance(df_clean, 'Inseminator', "🏆 Inseminator Ranking (Avg Live)", ax_ins)
                            st.pyplot(fig_ins, use_container_width=True)
                            st.download_button("📥 Download Inseminator Chart", download_chart(fig_ins, "inseminator_rank.png"), "inseminator_rank.png")

                    with col2:
                        if 'Semen' in df_clean.columns:
                            fig_sem, ax_sem = plt.subplots(figsize=(6, 6))
                            plot_top_performance(df_clean, 'Semen', "🧬 Semen Batch Ranking (Avg Live)", ax_sem)
                            st.pyplot(fig_sem, use_container_width=True)
                            st.download_button("📥 Download Semen Chart", download_chart(fig_sem, "semen_rank.png"), "semen_rank.png")

            with tab2:
                # --- GÖRBE (ANGOL - ÚJ: DARABSZÁM A %)
                df_clean['P_Group'] = df_clean['Parity'].apply(lambda x: x if x < 8 else 8)
                stat = df_clean.groupby('P_Group').agg({'Liveborn':'mean', 'Stillborn':'sum', 'Totalborn':'sum', 'Parity':'count'}).reset_index()
                
                # ÚJ SZÁMÍTÁS: Átlagos darabszám (Total Stillborn / Esetek száma)
                stat['Avg_SB_Count'] = stat['Stillborn'] / stat['Parity']
                
                fig2, ax = plt.subplots(figsize=(9, 6))
                
                # Zöld vonal (Élve)
                ax.plot(stat['P_Group'], stat['Liveborn'], 'go-', lw=2)
                force_text_with_counts(ax, stat['P_Group'], stat['Liveborn'], stat['Parity'], 'green', position='top')
                
                ax.set_ylim(0, stat['Liveborn'].max() * 1.3)
                ax.set_xticks(stat['P_Group'])
                ax.set_xticklabels([f"{int(p)}\n({c})" for p, c in zip(stat['P_Group'], stat['Parity'])])
                ax.set_title("Production Curve (by Parity)", fontsize=11); ax.grid(True, alpha=0.3)
                ax.set_ylabel("Liveborn (count)", color='green')

                # Piros oszlopok (MOST MÁR DARABSZÁM, NEM %)
                axr = ax.twinx()
                bars = axr.bar(stat['P_Group'], stat['Avg_SB_Count'], color='red', alpha=0.2)
                
                # Címkék (nem százalékos!)
                force_text_on_bars(axr, bars, is_percent=False, color='darkred')
                
                axr.set_ylim(0, stat['Avg_SB_Count'].max() * 1.5) # Bő ráhagyás
                axr.set_ylabel("Avg Stillborn (count)", color='red')
                
                st.pyplot(fig2, use_container_width=True)
                st.download_button("📥 Download Curve Chart", download_chart(fig2, "production_curve.png"), "production_curve.png", "image/png")
            
            with tab3:
                # --- FAJTA (ANGOL) ---
                if 'Breed' in df_clean.columns:
                    b_stat = df_clean.groupby('Breed').agg({'Liveborn':'mean', 'Stillborn':'mean', 'Parity':'count'}).reset_index()
                    b_stat = b_stat[b_stat['Parity'] >= 5].sort_values('Liveborn')
                    
                    if not b_stat.empty:
                        b_stat['Total_Avg'] = b_stat['Liveborn'] + b_stat['Stillborn']
                        fig3, ax3 = plt.subplots(figsize=(8, len(b_stat)*0.8 + 1))
                        
                        p1 = ax3.barh(b_stat['Breed'], b_stat['Liveborn'], color='teal', label='Live')
                        p2 = ax3.barh(b_stat['Breed'], b_stat['Stillborn'], left=b_stat['Liveborn'], color='salmon', label='Still')
                        
                        ax3.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f')
                        ax3.bar_label(p2, label_type='center', color='black', fontsize=9, fmt='%.1f')
                        
                        for i, (idx, row) in enumerate(b_stat.iterrows()):
                             text = f"Σ {row['Total_Avg']:.1f} ({int(row['Parity'])})"
                             ax3.text(row['Total_Avg'] + 0.2, i, text, va='center', color='black', fontsize=9, fontweight='bold')

                        ax3.set_xlim(0, b_stat['Total_Avg'].max() * 1.35)
                        ax3.set_title("Breed Performance", fontsize=11); ax3.legend(loc='lower right', fontsize=9)
                        
                        st.pyplot(fig3, use_container_width=True)
                        st.download_button("📥 Download Breed Chart", download_chart(fig3, "breed_analysis.png"), "breed_analysis.png", "image/png")
                    else:
                        st.warning("Not enough data (min. 5 farrowings per breed).")
                else:
                    st.info("No Breed data found.")

        else:
            st.error("Missing columns in the file.")

