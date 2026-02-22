import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Nagy Dűr Farrowing Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stDownloadButton > button { background-color: #0066cc; color: white; border: none; width: 100%; font-size: 18px; font-weight: bold; padding: 10px;}
    .stDownloadButton > button:hover { background-color: #004c99; color: white; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #0066cc; font-weight: bold; }
    [data-testid="stMetricLabel"] { font-size: 16px; font-weight: bold; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 36px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🐖 Nagy Dűr Farrowing Analyzer")
st.markdown("---")

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

        date_col = next((c for c in df.columns if 'date' in c.lower() or 'dátum' in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Farrow_Week'] = df[date_col].dt.isocalendar().week
            days_hu = {0: 'Hétfő', 1: 'Kedd', 2: 'Szerda', 3: 'Csütörtök', 4: 'Péntek', 5: 'Szombat', 6: 'Vasárnap'}
            df['Day_Name'] = df[date_col].dt.dayofweek.map(days_hu)
            df['Day_Num'] = df[date_col].dt.dayofweek
        else:
            week_col = next((c for c in df.columns if 'week' in c.lower() or 'hét' in c.lower()), None)
            if week_col:
                df['Farrow_Week'] = df[week_col]

        mapping = {
            'Sow name': ['Sow name', 'Koca', 'Anya', 'Sow'],
            'Parity': ['Parity', 'Fialás', 'Ellés', 'Sorszám'],
            'Breed': ['Breed', 'Fajta', 'Genetika'],
            'Inseminator': ['Inseminator Name', 'Inszemináló', 'Rakó'],
            'Semen': ['Semen batches', 'Semen', 'Mag', 'Termékenyítőanyag', 'Batch'],
            'Ins_Week': ['Group', 'Insemination Week', 'Rakás hete', 'Inszeminálás hete'],
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
        
        df = df.rename(columns=new_cols)
        
        if 'Ins_Week' not in df.columns and date_col:
            df['Ins_Week'] = (df[date_col] - pd.Timedelta(days=115)).dt.isocalendar().week
            
        return df

    except Exception as e:
        return None

def export_to_pdf(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in figs:
            fig.set_size_inches(11.69, 8.27)
            pdf.savefig(fig, orientation='landscape', bbox_inches='tight')
    buf.seek(0)
    return buf

def force_text_on_bars(ax, bars, is_percent=False, color='black'):
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:
            label = f"{height:.1f}%" if is_percent else f"{height:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom', color=color, fontweight='bold', fontsize=12)

def force_text_with_counts(ax, x_data, y_data, counts, color, position='top'):
    for x, y, c in zip(x_data, y_data, counts):
        if pd.notnull(y) and c > 0:
            label = f"{y:.1f}\n({int(c)})"
            xytext = (0, 15) if position == 'top' else (0, -35)
            va = 'bottom' if position == 'top' else 'top'
            ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, ha='center', va=va, color=color, fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))

# --- FŐ PROGRAM ---
uploaded_files = st.sidebar.file_uploader("📂 Upload Files", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        data = load_data(f)
        if data is not None: dfs.append(data)
    
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        
        needed = ['Parity', 'Liveborn', 'Farrow_Week']
        if all(c in df_all.columns for c in needed):
            
            for col in ['Parity', 'Liveborn', 'Stillborn', 'Farrow_Week', 'Ins_Week']:
                if col in df_all.columns:
                    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
            
            df_clean = df_all.dropna(subset=['Parity', 'Liveborn', 'Farrow_Week']).copy()
            df_clean['Parity'] = df_clean['Parity'].astype(int)
            df_clean['Is_Gilt'] = df_clean['Parity'] == 1
            
            if 'Stillborn' in df_clean.columns:
                df_clean['Totalborn'] = df_clean['Liveborn'] + df_clean['Stillborn']
            else:
                df_clean['Totalborn'] = df_clean['Liveborn']
                df_clean['Stillborn'] = 0

            # --- KPI SÁV ---
            sows_data = df_clean[~df_clean['Is_Gilt']]
            gilts_data = df_clean[df_clean['Is_Gilt']]
            
            overall_avg = df_clean['Liveborn'].mean()
            avg_live_sow = sows_data['Liveborn'].mean() if not sows_data.empty else 0
            avg_live_gilt = gilts_data['Liveborn'].mean() if not gilts_data.empty else 0
            sb_rate = df_clean['Stillborn'].sum() / df_clean['Totalborn'].sum() * 100
            
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Farrowings", f"{len(df_clean)}")
            k2.metric("Total AVG Live", f"{overall_avg:.2f}")
            k3.metric("Sow AVG Live", f"{avg_live_sow:.2f}")
            k4.metric("Gilt AVG Live", f"{avg_live_gilt:.2f}")
            k5.metric("Avg Loss Rate", f"{sb_rate:.1f}%")
            k6.metric("Total Piglets", f"{int(df_clean['Liveborn'].sum())}")
            
            st.markdown("---")
            
            figs_to_export = []

            # ROW 1
            col_a, col_b = st.columns(2)
            
            with col_a:
                if 'Ins_Week' in df_clean.columns:
                    def calc_ins(x):
                        sows = x[~x['Is_Gilt']]
                        gilts = x[x['Is_Gilt']]
                        return pd.Series({
                            'L_Total': x['Liveborn'].mean(), 'C_Total': len(x),
                            'L_Sow': sows['Liveborn'].mean() if len(sows)>0 else None,
                            'L_Gilt': gilts['Liveborn'].mean() if len(gilts)>0 else None,
                            'C_Sow': len(sows), 'C_Gilt': len(gilts)
                        })
                    
                    ins_stat = df_clean.groupby('Ins_Week').apply(calc_ins).reset_index().sort_values('Ins_Week')
                    
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    x_lbl = ins_stat['Ins_Week'].astype(int).astype(str)
                    x_pos = range(len(ins_stat))
                    
                    ax1.plot(x_pos, ins_stat['L_Total'], 'bo-', label='Overall Avg', lw=3, markersize=8)
                    ax1.plot(x_pos, ins_stat['L_Sow'], 'go--', label='Sow (2+)', lw=2)
                    ax1.plot(x_pos, ins_stat['L_Gilt'], 's--', color='orange', label='Gilt (1)', lw=2)
                    
                    # SZŰKÍTETT Y-TENGELY (Dinamikus minimum és maximum keresés)
                    min_val = ins_stat[['L_Total', 'L_Sow', 'L_Gilt']].min().min()
                    max_val = ins_stat[['L_Total', 'L_Sow', 'L_Gilt']].max().max()
                    ax1.set_ylim(min_val - 2.5, max_val + 3.0)
                    
                    # ÖSSZES PONT ADATAI
                    force_text_with_counts(ax1, x_pos, ins_stat['L_Total'], ins_stat['C_Total'], 'blue', position='top')
                    force_text_with_counts(ax1, x_pos, ins_stat['L_Sow'], ins_stat['C_Sow'], 'green', position='top')
                    force_text_with_counts(ax1, x_pos, ins_stat['L_Gilt'], ins_stat['C_Gilt'], 'darkorange', position='bottom')
                    
                    ax1.set_xticks(x_pos); ax1.set_xticklabels([f"Wk {w}" for w in x_lbl], fontsize=11)
                    ax1.set_title("Performance by Insemination Week (Avg Live)", fontsize=14, fontweight='bold')
                    ax1.legend(loc='lower right', fontsize=11); ax1.grid(True, alpha=0.3)
                    
                    st.pyplot(fig1, use_container_width=True)
                    figs_to_export.append(fig1)

            with col_b:
                if 'Day_Num' in df_clean.columns:
                    day_stat = df_clean.groupby(['Day_Num', 'Day_Name']).size().reset_index(name='Count')
                    all_days = pd.DataFrame({'Day_Num': range(7), 'Day_Name': ['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']})
                    day_stat = pd.merge(all_days, day_stat, on=['Day_Num', 'Day_Name'], how='left').fillna(0)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    bars = ax2.bar(day_stat['Day_Name'], day_stat['Count'], color='purple', alpha=0.7)
                    
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax2.text(bar.get_x() + bar.get_width()/2., height + (day_stat['Count'].max()*0.02),
                                     f"{int(height)}", ha='center', va='bottom', fontweight='bold', fontsize=14)
                    
                    ax2.set_ylim(0, day_stat['Count'].max() * 1.3)
                    ax2.set_title("Farrowings by Day of the Week", fontsize=14, fontweight='bold')
                    ax2.grid(axis='y', alpha=0.3)
                    
                    st.pyplot(fig2, use_container_width=True)
                    figs_to_export.append(fig2)

            st.markdown("---")
            
            # ROW 2
            col_c, col_d = st.columns(2)
            
            with col_c:
                categories = ['Sow (2+)', 'Gilt (1)', 'Overall (All)']
                live_vals = [
                    sows_data['Liveborn'].mean() if not sows_data.empty else 0, 
                    gilts_data['Liveborn'].mean() if not gilts_data.empty else 0,
                    df_clean['Liveborn'].mean() if not df_clean.empty else 0
                ]
                still_vals = [
                    sows_data['Stillborn'].mean() if not sows_data.empty else 0, 
                    gilts_data['Stillborn'].mean() if not gilts_data.empty else 0,
                    df_clean['Stillborn'].mean() if not df_clean.empty else 0
                ]
                total_vals = [l + s for l, s in zip(live_vals, still_vals)]
                counts = [len(sows_data), len(gilts_data), len(df_clean)]
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                p1 = ax3.bar(categories, live_vals, color='green', label='Live', width=0.6)
                p2 = ax3.bar(categories, still_vals, bottom=live_vals, color='salmon', label='Still', width=0.6)
                
                ax3.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f', fontsize=14)
                for rect in p2:
                    height = rect.get_height()
                    if height > 0.1:
                        ax3.text(rect.get_x() + rect.get_width()/2., rect.get_y() + height/2.,
                                 f"{height:.1f}", ha='center', va='center', color='black', fontsize=12)
                
                for i, total in enumerate(total_vals):
                    if total > 0:
                        ax3.text(i, total + 0.2, f"Total: {total:.1f}\n({counts[i]})", 
                                 ha='center', va='bottom', fontweight='bold', fontsize=14)
                
                ax3.set_ylim(0, max(total_vals) * 1.35 if total_vals else 10)
                ax3.set_title("Performance Breakdown (Live + Still)", fontsize=14, fontweight='bold')
                ax3.legend(loc='upper right', fontsize=11)
                
                st.pyplot(fig3, use_container_width=True)
                figs_to_export.append(fig3)

            with col_d:
                df_clean['P_Group'] = df_clean['Parity'].apply(lambda x: x if x < 8 else 8)
                stat = df_clean.groupby('P_Group').agg({'Liveborn':'mean', 'Stillborn':'sum', 'Totalborn':'sum', 'Parity':'count'}).reset_index()
                stat['Avg_SB_Count'] = stat['Stillborn'] / stat['Parity']
                
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.plot(stat['P_Group'], stat['Liveborn'], 'go-', lw=3, markersize=8)
                force_text_with_counts(ax4, stat['P_Group'], stat['Liveborn'], stat['Parity'], 'green', position='top')
                
                ax4.set_ylim(0, stat['Liveborn'].max() * 1.35)
                ax4.set_xticks(stat['P_Group'])
                ax4.set_xticklabels([f"{int(p)}\n({c})" for p, c in zip(stat['P_Group'], stat['Parity'])], fontsize=11)
                ax4.set_title("Production Curve by Parity (Liveborn vs Loss)", fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)

                axr = ax4.twinx()
                bars = axr.bar(stat['P_Group'], stat['Avg_SB_Count'], color='red', alpha=0.2, width=0.4)
                force_text_on_bars(axr, bars, is_percent=False, color='darkred')
                axr.set_ylim(0, stat['Avg_SB_Count'].max() * 1.5)
                
                st.pyplot(fig4, use_container_width=True)
                figs_to_export.append(fig4)

            st.markdown("---")
            
            # ROW 3
            col_e, col_f = st.columns(2)
            
            with col_e:
                if 'Breed' in df_clean.columns:
                    b_stat = df_clean.groupby('Breed').agg({'Liveborn':'mean', 'Stillborn':'mean', 'Parity':'count'}).reset_index()
                    b_stat = b_stat[b_stat['Parity'] >= 5].sort_values('Liveborn')
                    
                    if not b_stat.empty:
                        b_stat['Total_Avg'] = b_stat['Liveborn'] + b_stat['Stillborn']
                        fig5, ax5 = plt.subplots(figsize=(10, 6))
                        
                        p1 = ax5.barh(b_stat['Breed'], b_stat['Liveborn'], color='teal', label='Live')
                        p2 = ax5.barh(b_stat['Breed'], b_stat['Stillborn'], left=b_stat['Liveborn'], color='salmon', label='Still')
                        
                        ax5.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f', fontsize=12)
                        ax5.bar_label(p2, label_type='center', color='black', fontsize=11, fmt='%.1f')
                        
                        for i, (idx, row) in enumerate(b_stat.iterrows()):
                             text = f"Σ {row['Total_Avg']:.1f} ({int(row['Parity'])})"
                             ax5.text(row['Total_Avg'] + 0.2, i, text, va='center', color='black', fontsize=12, fontweight='bold')

                        ax5.set_xlim(0, b_stat['Total_Avg'].max() * 1.35)
                        ax5.set_title("Breed Performance", fontsize=14, fontweight='bold')
                        ax5.legend(loc='lower right', fontsize=11)
                        
                        st.pyplot(fig5, use_container_width=True)
                        figs_to_export.append(fig5)

            with col_f:
                if 'Inseminator' in df_clean.columns:
                    ins_stats = df_clean.groupby('Inseminator').agg({'Liveborn':'mean', 'Parity':'count'}).reset_index()
                    ins_stats = ins_stats[ins_stats['Parity'] >= 3].sort_values('Liveborn', ascending=True).tail(8)
                    
                    if not ins_stats.empty:
                        fig6, ax6 = plt.subplots(figsize=(10, 6))
                        bars = ax6.barh(ins_stats['Inseminator'].astype(str), ins_stats['Liveborn'], color='royalblue')
                        for i, (idx, row) in enumerate(ins_stats.iterrows()):
                            ax6.text(row['Liveborn'], i, f" {row['Liveborn']:.1f} ({int(row['Parity'])})", 
                                   va='center', fontweight='bold', fontsize=12)
                        
                        ax6.set_xlim(0, ins_stats['Liveborn'].max() * 1.35)
                        ax6.set_title("Top 8 Inseminators (Avg Live)", fontsize=14, fontweight='bold')
                        
                        st.pyplot(fig6, use_container_width=True)
                        figs_to_export.append(fig6)
                        
            st.markdown("---")
            
            if figs_to_export:
                pdf_buffer = export_to_pdf(figs_to_export)
                st.download_button(
                    label="📄 Download Full PDF Report (A4 Landscape)",
                    data=pdf_buffer,
                    file_name="Farrowing_Report.pdf",
                    mime="application/pdf"
                )
                        
        else:
            st.error("Missing columns.")
