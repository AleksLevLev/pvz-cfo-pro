import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. CONFIG & STYLE (DESIGN SYSTEM) ---
st.set_page_config(
    layout="wide",
    page_title="Финансы ПВЗ 📦",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

# Minimalist CSS for mobile-first feel
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem !important; font-weight: 700;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 600;}
    
    /* File Uploader Translation Hack */
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section::after {
        content: "Перетащите Excel-файл сюда (или несколько)";
        display: block;
        text-align: center;
        margin-top: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION (SECURITY) ---
def check_password():
    """Простая защита доступа"""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔒 Вход в систему")
            pwd = st.text_input("Введите пароль доступа", type="password")
            if st.button("Войти", type="primary", use_container_width=True):
                if pwd == "admin":  # ЗАДАЙ ПАРОЛЬ ЗДЕСЬ
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Неверный пароль")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. DATA ENGINE (LOGIC) ---
@st.cache_data
def get_mock_data():
    """Генерация демо-данных (кэшируем для скорости)"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30).tolist()
    random_dates = np.random.choice(dates, 500)
    
    data = {
        'date': random_dates,
        'operation_type': np.random.choice(['Выдача', 'Возврат', 'Приемка'], 500, p=[0.7, 0.1, 0.2]),
        'wb_reward': np.random.uniform(15.0, 120.0, 500).round(2),
        'penalty_amount': np.random.choice([0, 50, 100, 500], 500, p=[0.7, 0.15, 0.1, 0.05]),
        'penalty_reason': np.random.choice(
            ['Отсутствует', 'Подмена', 'Брак', 'Рейтинг', 'Утеря'], 
            500, 
            p=[0.7, 0.05, 0.1, 0.1, 0.05]
        )
    }
    df = pd.DataFrame(data)
    df.loc[df['penalty_amount'] == 0, 'penalty_reason'] = 'Отсутствует'
    return df

def load_single_file(uploaded_file):
    """Helper to load and normalize a single file"""
    df = None
    try:
        # 1. Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            try:
                # Attempt 1: Standard CSV
                df = pd.read_csv(uploaded_file)
                if df.shape[1] < 2:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')
            except:
                # Attempt 2: Russian Excel CSV
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, sep=';', encoding='cp1251')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        else:
            # Smart Sheet Search
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            target_columns = ['wb_reward', 'Вайлдберриз реализовал', 'Прибыль', 'Штрафы']
            
            found_sheet = False
            for sheet_name, sheet_df in xls.items():
                if any(col in sheet_df.columns for col in target_columns):
                    df = sheet_df
                    found_sheet = True
                    break
            
            if not found_sheet:
                df = list(xls.values())[0]

        # 2. Normalize columns
        df.columns = df.columns.astype(str).str.strip()
        
        column_mapping_standard = {
            'Вайлдберриз реализовал': 'wb_reward',
            'Штрафы': 'penalty_amount',
            'Тип операции': 'operation_type',
            'Обоснование штрафа': 'penalty_reason',
            'Вид начисления': 'operation_type',
            'Начислено': 'wb_reward',
            'Кол-во': 'quantity',
            'Баркод': 'barcode',
            'Дата': 'date',
            'date': 'date',
            'Время': 'date'
        }
        df = df.rename(columns=column_mapping_standard)
        
        # Format 2: Sales Report
        if 'wb_reward' not in df.columns and 'Прибыль' in df.columns:
            df['wb_reward'] = df['Прибыль']
            if 'Удержания' in df.columns:
                df['penalty_amount'] = df['Удержания']
            else:
                df['penalty_amount'] = 0
            df['operation_type'] = 'Выдача'
            df['penalty_reason'] = 'Прочее'

        # Fill NaNs
        if 'penalty_amount' in df.columns:
            df['penalty_amount'] = df['penalty_amount'].fillna(0)
        if 'wb_reward' in df.columns:
            df['wb_reward'] = df['wb_reward'].fillna(0)
            
        return df

    except Exception as e:
        st.error(f"Ошибка в файле {uploaded_file.name}: {e}")
        return None

# --- 4. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.header("⚙️ Настройки расходов")
    
    st.subheader("Загрузка данных")
    st.info("Загрузите один или несколько отчетов Wildberries.")
    uploaded_files = st.file_uploader("Перетащите файлы сюда", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    st.divider()
    
    st.subheader("Финансы (мес)")
    rent = st.number_input("Аренда", value=30000, step=1000)
    internet_security = st.number_input("Охрана/ПО", value=3000, step=500)
    consumables = st.number_input("Расходники", value=5000, step=500)
    amortization = st.number_input("Амортизация", value=2000, step=500)
    
    st.divider()
    
    st.subheader("Налоги")
    tax_rate = st.number_input("Налог УСН (%)", value=6.0, step=0.5)
    reserve_rate = st.number_input("% в Резерв", value=15.0, step=1.0)

# --- 5. MAIN INTERFACE ---
st.title("Финансы ПВЗ 📦")

# --- A. STAFF MANAGEMENT ---
with st.expander("👥 Управление сменами (ФОТ)", expanded=False):
    default_staff = pd.DataFrame([
        {"Сотрудник": "Иванов А.", "Кол-во смен": 3, "Ставка": 1500, "Бонус": 0},
        {"Сотрудник": "Петрова С.", "Кол-во смен": 4, "Ставка": 1500, "Бонус": 1000}
    ])
    
    edited_staff = st.data_editor(
        default_staff, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Ставка": st.column_config.NumberColumn(format="%d ₽"),
            "Бонус": st.column_config.NumberColumn(format="%d ₽")
        }
    )
    edited_staff['Total'] = (edited_staff['Кол-во смен'] * edited_staff['Ставка']) + edited_staff['Бонус']
    total_fot = edited_staff['Total'].sum()
    st.caption(f"Итого ФОТ за период: {total_fot:,.0f} ₽")

# --- B. DATA PROCESSING ---
main_df = pd.DataFrame()

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        df_temp = load_single_file(file)
        if df_temp is not None:
            df_temp['source_file'] = file.name
            all_dfs.append(df_temp)
    
    if all_dfs:
        main_df = pd.concat(all_dfs, ignore_index=True)
        main_df = main_df.drop_duplicates()
        st.toast(f"Загружено файлов: {len(uploaded_files)}", icon="📚")
else:
    main_df = get_mock_data()

# --- DATE FILTERING ---
report_period = "Весь период" # Default value
if not main_df.empty and 'date' in main_df.columns:
    # Convert dates
    main_df['date'] = pd.to_datetime(main_df['date'], dayfirst=True, errors='coerce')
    main_df = main_df.dropna(subset=['date']) # Drop rows with invalid dates
    
    if not main_df.empty: # Check again after dropping NaNs
        min_date_overall = main_df['date'].min().date()
        max_date_overall = main_df['date'].max().date()
        
        with st.sidebar:
            st.divider()
            st.subheader("📅 Период")
            date_range = st.date_input(
                "Выберите диапазон",
                value=(min_date_overall, max_date_overall),
                min_value=min_date_overall,
                max_value=max_date_overall,
                format="DD.MM.YYYY"
            )
        
        # Apply Filter
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            main_df = main_df[(main_df['date'].dt.date >= start_d) & (main_df['date'].dt.date <= end_d)]
            report_period = f"{start_d.strftime('%d.%m.%Y')} — {end_d.strftime('%d.%m.%Y')}"
        else: # If only one date is selected (e.g., by clicking on a single date)
            start_d = date_range[0]
            main_df = main_df[main_df['date'].dt.date == start_d]
            report_period = f"{start_d.strftime('%d.%m.%Y')}"
    else:
        st.warning("В загруженных данных нет корректных дат для фильтрации.")


# --- C. CALCULATIONS ---
if not main_df.empty:
    gross_income = main_df['wb_reward'].sum()
    total_penalties = main_df['penalty_amount'].sum()
    
    # Calculate weeks in selected period for averaging
    if 'date' in main_df.columns and not main_df['date'].empty:
        days_diff = (main_df['date'].max() - main_df['date'].min()).days
        num_weeks = max(days_diff / 7, 1) # Avoid division by zero, ensure at least 1 week
    else:
        num_weeks = 1

    # Adjust fixed costs to the selected period duration
    # Monthly costs / 4.3 * num_weeks
    period_fixed_costs = (rent + internet_security + consumables + amortization) / 4.3 * num_weeks
    
    tax_sum = gross_income * (tax_rate / 100)
    net_profit = gross_income - total_penalties - tax_sum - period_fixed_costs - total_fot
    dividends = net_profit * (1 - reserve_rate / 100)
    
    # Unit Economics
    issue_ops = main_df[main_df['operation_type'] == 'Выдача'].shape[0]
    total_expenses = total_penalties + tax_sum + period_fixed_costs + total_fot
    unit_cost = total_expenses / issue_ops if issue_ops > 0 else 0
    avg_revenue = (gross_income / issue_ops) if issue_ops > 0 else 0

    # Business Metrics
    margin_percent = (net_profit / gross_income * 100) if gross_income > 0 else 0
    avg_weekly_profit = net_profit / num_weeks

    # --- D. DASHBOARD LAYOUT ---

    # 1. Key Metrics
    st.subheader("Финансовый результат")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Выручка", f"{gross_income:,.0f} ₽".replace(',', ' '))
    c2.metric("Чистая Прибыль", f"{net_profit:,.0f} ₽".replace(',', ' '))
    
    # Margin with color logic
    margin_delta_str = ""
    margin_delta_color = "normal"
    if margin_percent < 10 and margin_percent >= 0:
        margin_delta_str = "Низкая"
        margin_delta_color = "inverse"
    elif margin_percent < 0:
        margin_delta_str = "Отрицательная"
        margin_delta_color = "inverse"
    else:
        margin_delta_str = "Норма"
        margin_delta_color = "normal"

    c3.metric("Рентабельность", f"{margin_percent:.1f}%", delta=margin_delta_str, delta_color=margin_delta_color)
    c4.metric("Прибыль в неделю (ср.)", f"{avg_weekly_profit:,.0f} ₽".replace(',', ' '))

    st.markdown("---")

    # 2. Verdict Block
    st.subheader("🤖 Анализ ситуации")
    if net_profit > 0:
        st.success(f"✅ **Отличная работа!**\n\nТочка в плюсе на **{net_profit:,.0f} ₽**. Рентабельность: **{margin_percent:.1f}%**.\n\n📅 *Период отчета: {report_period}*")
    else:
        st.error(f"🚨 **Внимание! Убыток {abs(net_profit):,.0f} ₽.**\n\nРасходы превышают доходы. Рентабельность отрицательная.\n\n📅 *Период отчета: {report_period}*")

    st.markdown("---")

    # 3. Charts
    col_main, col_side = st.columns([2, 1])

    with col_main:
        if num_weeks <= 1.5:
            # --- SCENARIO 1: SHORT TERM (WATERFALL) ---
            fig_waterfall = go.Figure(go.Waterfall(
                name="Cashflow", orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "total"],
                x=["Выручка", "Штрафы", "Налоги", "Аренда/Fix", "ФОТ", "Прибыль"],
                textposition="outside",
                text=[f"{x:,.0f}" for x in [gross_income, -total_penalties, -tax_sum, -period_fixed_costs, -total_fot, net_profit]],
                y=[gross_income, -total_penalties, -tax_sum, -period_fixed_costs, -total_fot, net_profit],
                connector={"line": {"color": "rgb(200, 200, 200)"}},
                decreasing={"marker": {"color": "#E74C3C"}},
                increasing={"marker": {"color": "#2ECC71"}},
                totals={"marker": {"color": "#333333"}},
                hovertemplate='%{label}: %{value:,.0f} ₽<extra></extra>'
            ))
            fig_waterfall.update_layout(
                title="Движение средств (Waterfall)", 
                margin=dict(l=0, r=0, t=40, b=0),
                height=400,
                showlegend=False,
                separators=" ."
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            # --- SCENARIO 2: LONG TERM (DYNAMICS) ---
            # Bar Chart (Dynamics by Date/Week)
            weekly_data = main_df.groupby(pd.Grouper(key='date', freq='W-MON')).agg(
                wb_reward=('wb_reward', 'sum'),
                penalty_amount=('penalty_amount', 'sum')
            ).reset_index()
            
            weekly_data['net_result_approx'] = weekly_data['wb_reward'] - weekly_data['penalty_amount']
            
            # Color logic (Soft colors)
            weekly_data['color'] = np.where(weekly_data['net_result_approx'] < 0, '#E74C3C', '#2ECC71')
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=weekly_data['date'],
                y=weekly_data['net_result_approx'],
                marker_color=weekly_data['color'],
                hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Результат: %{y:,.0f} ₽<extra></extra>'
            ))
            
            fig_bar.update_layout(
                title="Динамика результата (по неделям)",
                margin=dict(l=0, r=0, t=40, b=0),
                height=400,
                showlegend=False,
                separators=" ."
            )
            fig_bar.update_xaxes(tickformat="%d.%m")
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_side:
        if num_weeks <= 1.5:
            # --- SCENARIO 1: EXPENSES STRUCTURE (DONUT) ---
            expenses_data = pd.DataFrame([
                {"Category": "Штрафы", "Amount": total_penalties},
                {"Category": "Налоги", "Amount": tax_sum},
                {"Category": "Аренда/Fix", "Amount": period_fixed_costs},
                {"Category": "ФОТ", "Amount": total_fot}
            ])
            # Filter out zero expenses
            expenses_data = expenses_data[expenses_data["Amount"] > 0]
            
            if not expenses_data.empty:
                fig_donut = px.pie(
                    expenses_data, 
                    values='Amount', 
                    names='Category', 
                    hole=0.6,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_donut.update_traces(
                    textinfo='percent', 
                    hovertemplate='<b>%{label}</b><br>Сумма: %{value:,.0f} ₽<extra></extra>'
                )
                
                # Center text with total expenses
                total_exp = expenses_data['Amount'].sum()
                fig_donut.update_layout(
                    title="Структура расходов",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    annotations=[dict(text=f"{total_exp/1000:.0f}k", x=0.5, y=0.5, font_size=24, showarrow=False)],
                    separators=" ."
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.container(height=400, border=True).write("✅ Расходов нет")
        else:
            # --- SCENARIO 2: PENALTIES (HORIZONTAL BAR) ---
            penalties_df = main_df[main_df['penalty_amount'] > 0]
            if not penalties_df.empty:
                reason_group = penalties_df.groupby('penalty_reason')['penalty_amount'].sum().reset_index()
                reason_group = reason_group.sort_values(by='penalty_amount', ascending=True) # Sort for horizontal bar
                
                fig_h_bar = go.Figure(go.Bar(
                    x=reason_group['penalty_amount'],
                    y=reason_group['penalty_reason'],
                    orientation='h',
                    marker_color='#E74C3C',
                    hovertemplate='<b>%{y}</b><br>Сумма: %{x:,.0f} ₽<extra></extra>'
                ))
                
                fig_h_bar.update_layout(
                    title="Топ причин штрафов",
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=400,
                    showlegend=False,
                    separators=" ."
                )
                st.plotly_chart(fig_h_bar, use_container_width=True)
            else:
                st.container(height=400, border=True).write("🎉 Штрафов нет")

    # 3. Unit Economics Insights
    st.markdown("### 🧠 Unit-экономика")
    e1, e2 = st.columns(2)
    with e1:
        st.info(f"Себестоимость выдачи: **{unit_cost:.1f} ₽** / шт")
    with e2:
        margin = avg_revenue - unit_cost
        if margin > 0:
            st.success(f"Заработок с 1 выдачи: **{margin:.1f} ₽**")
        else:
            st.error(f"Убыток с 1 выдачи: **{margin:.1f} ₽**")

    # 4. Anti-Penalty Module
    if total_penalties > 0:
        st.markdown("---")
        with st.expander("⚖️ Помощник по оспариванию штрафов (Нажмите, чтобы развернуть)", expanded=False):
            p_col1, p_col2 = st.columns([1, 1])
            
            # Filter penalties
            penalties_df = main_df[main_df['penalty_amount'] > 0].copy()
            # Format date for display
            if 'date' in penalties_df.columns:
                penalties_df['date_str'] = penalties_df['date'].dt.strftime('%d.%m.%Y')
            
            with p_col1:
                st.subheader("📋 Список нарушений")
                # Show simplified table
                display_cols = ['date_str', 'penalty_amount', 'penalty_reason', 'operation_type']
                # Rename for display
                display_df = penalties_df[display_cols].rename(columns={
                    'date_str': 'Дата',
                    'penalty_amount': 'Штраф',
                    'penalty_reason': 'Причина',
                    'operation_type': 'Тип'
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with p_col2:
                st.subheader("📝 Текст претензии")
                
                # Generate claim text
                claim_lines = []
                for index, row in penalties_df.iterrows():
                    d = row['date_str']
                    s = row['penalty_amount']
                    r = row['penalty_reason']
                    claim_lines.append(f"{d} — {s} руб. — {r}")
                
                details_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(claim_lines)])
                
                today_str = pd.Timestamp.today().strftime('%d.%m.%Y')
                
                claim_text = f"""В поддержку Wildberries
От партнера (ID точки: [ВАШ ID])

ПРЕТЕНЗИЯ О НЕСОГЛАСИИ С УДЕРЖАНИЯМИ

За период с {report_period} были произведены удержания на общую сумму {total_penalties:,.0f} руб.
Считаю данные удержания необоснованными, так как товары были приняты и выданы корректно, видеозаписи имеются.

Детализация спорных операций:
{details_text}

Прошу отменить данные удержания и произвести перерасчет в ближайшем отчете.
Дата: {today_str}"""
                
                st.code(claim_text, language='text')
                st.caption("👆 Нажмите кнопку копирования в углу и вставьте в обращение на портале WB.")

else:
    st.info("Загрузите данные для отображения аналитики.")
