import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Telepi Elemző", layout="wide")
st.title("🐖 Telepi Reprodukciós Műszerfal")

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
            df['Week'] = df[date_col].dt.isocalendar().week
        else:
            week_col = next((c for c in df.columns if 'week' in c.lower() or 'hét' in c.lower()), None)
            if week_col:
                df['Week'] = df[week_col]

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

# --- FELIRATOZÓK ---
def force_text_on_bars(ax, bars, is_percent=False, color='black'):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            label = f"{height:.1f}%" if is_percent else f"{height:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', color=color, fontweight='bold', fontsize=9)

def force_text_with_counts(ax, x_data, y_data, counts, color, position='top'):
    for x, y, c in zip(x_data, y_data, counts):
        if pd.notnull(y) and c > 0:
            label = f"{y:.1f}\n({int(c)})"
            xytext = (0, 15) if position == 'top' else (0, -30) # Megnövelt távolság
            va = 'bottom' if position == 'top' else 'top'
            ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, ha='center', va=va, color=color, fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2))

# --- FŐ PROGRAM ---
uploaded_files = st.sidebar.file_uploader("Fájlok feltöltése", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        data = load_data(f)
        if data is not None: dfs.append(data)
    
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        
        needed = ['Parity', 'Liveborn', 'Week']
        if all(c in df_all.columns for c in needed):
            
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
            st.success(f"✅ Adatok betöltve: {len(df_clean)} fialás, {unique_weeks} hét.")
            
            tab1_name = "🔍 Heti Mélyelemzés" if unique_weeks == 1 else "📈 Heti Trendek"
            tab1, tab2, tab3 = st.tabs([tab1_name, "📊 Termelési Görbe", "🧬 Fajta Elemzés"])
            
            with tab1:
                if unique_weeks > 1:
                    # --- TREND (Több hét) ---
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
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10)) # Magasabb
                    x_lbl = weekly['Week'].astype(int).astype(str)
                    x_pos = range(len(weekly))

                    # FELSŐ (Élve)
                    ax1.plot(x_pos, weekly['L_Sow'], 'go-', label='Koca')
                    ax1.plot(x_pos, weekly['L_Gilt'], 's-', color='orange', label='Süldő')
                    force_text_with_counts(ax1, x_pos, weekly['L_Sow'], weekly['C_Sow'], 'green', position='top')
                    force_text_with_counts(ax1, x_pos, weekly['L_Gilt'], weekly['C_Gilt'], 'darkorange', position='bottom')
                    
                    # MARGÓ NÖVELÉS (Y tengely)
                    y_max = max(weekly['L_Sow'].max(), weekly['L_Gilt'].max())
                    ax1.set_ylim(0, y_max * 1.4) # +40% hely fent
                    
                    ax1.set_xticks(x_pos); ax1.set_xticklabels(x_lbl); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
                    ax1.set_title("Heti Élveszületés Trend", fontsize=11)

                    # ALSÓ (Veszteség)
                    ax2.bar(x_pos, weekly['S_Sow'], color='darkred', alpha=0.6, label='Koca %')
                    ax2.bar(x_pos, weekly['S_Gilt'], color='orange', alpha=0.6, label='Süldő %', bottom=weekly['S_Sow'])
                    
                    # MARGÓ NÖVELÉS
                    y_max_loss = (weekly['S_Sow'] + weekly['S_Gilt']).max()
                    ax2.set_ylim(0, y_max_loss * 1.3)
                    
                    ax2.set_xticks(x_pos); ax2.set_xticklabels(x_lbl); ax2.legend(); ax2.set_title("Veszteség Trend (%)", fontsize=11)
                    st.pyplot(fig, use_container_width=True)

                else:
                    # --- MÉLYELEMZÉS (1 hét) ---
                    st.info(f"Heti Részletes Elemzés: {df_clean['Week'].iloc[0]}. hét")
                    
                    sows = df_clean[~df_clean['Is_Gilt']]
                    gilts = df_clean[df_clean['Is_Gilt']]
                    
                    categories = ['Koca (2+)', 'Süldő (1)']
                    live_vals = [sows['Liveborn'].mean() if not sows.empty else 0, gilts['Liveborn'].mean() if not gilts.empty else 0]
                    still_vals = [sows['Stillborn'].mean() if not sows.empty else 0, gilts['Stillborn'].mean() if not gilts.empty else 0]
                    total_vals = [l + s for l, s in zip(live_vals, still_vals)]
                    counts = [len(sows), len(gilts)]
                    
                    fig_main, ax_main = plt.subplots(figsize=(8, 6))
                    p1 = ax_main.bar(categories, live_vals, color='green', label='Élve', width=0.5)
                    p2 = ax_main.bar(categories, still_vals, bottom=live_vals, color='salmon', label='Halva', width=0.5)
                    
                    ax_main.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f')
                    for rect in p2:
                        height = rect.get_height()
                        if height > 0.1:
                            ax_main.text(rect.get_x() + rect.get_width()/2., rect.get_y() + height/2.,
                                        f"{height:.1f}", ha='center', va='center', color='black', fontsize=9)
                    
                    for i, total in enumerate(total_vals):
                        if total > 0:
                            ax_main.text(i, total + 0.1, f"Total: {total:.1f}\n(n={counts[i]})", 
                                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
                    # MARGÓ NÖVELÉS (Y)
                    ax_main.set_ylim(0, max(total_vals) * 1.35 if total_vals else 10) 
                    
                    ax_main.set_title("Eredmények: Élve + Halva = Összes született", fontsize=12)
                    ax_main.legend(loc='upper right')
                    st.pyplot(fig_main, use_container_width=True)
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
                            
                            # MARGÓ NÖVELÉS (X - Jobbra)
                            ax.set_xlim(0, stats['Liveborn'].max() * 1.35) 
                            ax.set_title(title, fontsize=11)
                        else:
                            ax.text(0.5, 0.5, "Nincs elég adat", ha='center')
                            ax.set_title(title)

                    with col1:
                        if 'Inseminator' in df_clean.columns:
                            fig_ins, ax_ins = plt.subplots(figsize=(6, 6))
                            plot_top_performance(df_clean, 'Inseminator', "🏆 Inszeminátor Toplista", ax_ins)
                            st.pyplot(fig_ins, use_container_width=True)

                    with col2:
                        if 'Semen' in df_clean.columns:
                            fig_sem, ax_sem = plt.subplots(figsize=(6, 6))
                            plot_top_performance(df_clean, 'Semen', "🧬 Mag (Batch) Toplista", ax_sem)
                            st.pyplot(fig_sem, use_container_width=True)

            with tab2:
                # --- GÖRBE ---
                df_clean['P_Group'] = df_clean['Parity'].apply(lambda x: x if x < 8 else 8)
                stat = df_clean.groupby('P_Group').agg({'Liveborn':'mean', 'Stillborn':'sum', 'Totalborn':'sum', 'Parity':'count'}).reset_index()
                stat['SB_Rate'] = stat['Stillborn'] / stat['Totalborn'] * 100
                
                fig2, ax = plt.subplots(figsize=(9, 6))
                ax.plot(stat['P_Group'], stat['Liveborn'], 'go-', lw=2)
                force_text_with_counts(ax, stat['P_Group'], stat['Liveborn'], stat['Parity'], 'green', position='top')
                
                # MARGÓ NÖVELÉS
                ax.set_ylim(0, stat['Liveborn'].max() * 1.3)

                ax.set_xticks(stat['P_Group'])
                ax.set_xticklabels([f"{int(p)}\n({c})" for p, c in zip(stat['P_Group'], stat['Parity'])])
                ax.set_title("Termelési Görbe", fontsize=11); ax.grid(True, alpha=0.3)
                
                axr = ax.twinx()
                bars = axr.bar(stat['P_Group'], stat['SB_Rate'], color='red', alpha=0.2)
                force_text_on_bars(axr, bars, is_percent=True, color='darkred')
                axr.set_ylim(0, stat['SB_Rate'].max() * 1.4) # Margó a %-nak is
                axr.set_ylabel("Veszteség %", color='red')
                st.pyplot(fig2, use_container_width=True)
            
            with tab3:
                # --- FAJTA ---
                if 'Breed' in df_clean.columns:
                    b_stat = df_clean.groupby('Breed').agg({'Liveborn':'mean', 'Stillborn':'mean', 'Parity':'count'}).reset_index()
                    b_stat = b_stat[b_stat['Parity'] >= 5].sort_values('Liveborn')
                    
                    if not b_stat.empty:
                        b_stat['Total_Avg'] = b_stat['Liveborn'] + b_stat['Stillborn']
                        fig3, ax3 = plt.subplots(figsize=(8, len(b_stat)*0.8 + 1))
                        
                        p1 = ax3.barh(b_stat['Breed'], b_stat['Liveborn'], color='teal', label='Élve')
                        p2 = ax3.barh(b_stat['Breed'], b_stat['Stillborn'], left=b_stat['Liveborn'], color='salmon', label='Halva')
                        
                        ax3.bar_label(p1, label_type='center', color='white', fontweight='bold', fmt='%.1f')
                        ax3.bar_label(p2, label_type='center', color='black', fontsize=9, fmt='%.1f')
                        
                        for i, (idx, row) in enumerate(b_stat.iterrows()):
                             text = f"Σ {row['Total_Avg']:.1f} ({int(row['Parity'])})"
                             ax3.text(row['Total_Avg'] + 0.2, i, text, va='center', color='black', fontsize=9, fontweight='bold')

                        # MARGÓ JOBBRA
                        ax3.set_xlim(0, b_stat['Total_Avg'].max() * 1.35)

                        ax3.set_title("Fajták Teljesítménye", fontsize=11); ax3.legend(loc='lower right', fontsize=9)
                        st.pyplot(fig3, use_container_width=True)
                    else:
                        st.warning("Nincs elég adat (min. 5 fialás).")
                else:
                    st.info("Nincs fajta adat.")

        else:
            st.error("Hiányzó oszlopok.")