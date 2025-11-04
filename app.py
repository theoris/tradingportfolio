import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import numpy as np
from pathlib import Path
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ============ Streamlit Config ============
st.set_page_config(page_title="Trading Portfolio", layout="wide", page_icon="📈")

# --- Global CSS Fixes for AgGrid, Scrollbar, Header, Popup ---
st.markdown("""
<style>
/* General Dark Mode */
body, .main, [data-testid="stAppViewContainer"] {
    background-color: #111 !important;
    color: #EEE !important;
}
[data-testid="stHeader"] {background: none;}

/* AgGrid Dark Theme Tweaks */
.ag-theme-balham {
    --ag-background-color: #1e1e1e;
    --ag-odd-row-background-color: #2a2a2a;
    --ag-header-background-color: #222;
    --ag-font-color: #00008B;
    --ag-borders: #444;
}

/* Scrollbar always visible */
.ag-root-wrapper-body.ag-layout-normal {
    overflow: auto !important;
}
.ag-body-viewport {
    overflow-y: auto !important;
    overflow-x: auto !important;
}

/* Sticky Header */
.ag-header {
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
    background-color: #222 !important;
}
.ag-header-cell-label {
    color: #fff !important;
    font-weight: bold !important;
}

/* Prevent clipping of popups */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.block-container {
    overflow: visible !important;
}

/* Popup visibility */
.ag-popup {
    z-index: 9999 !important;
}
</style>
""", unsafe_allow_html=True)

# ============ Helper Functions ============

def load_data():
    base = Path(__file__).resolve().parent
    trade_path = base / "data" / "trading.json"
    update_path = base / "data" / "update.json"

    trade_df, update_df = pd.DataFrame(), pd.DataFrame()
    if trade_path.exists():
        with open(trade_path, "r", encoding="utf-8") as f:
            trade_df = pd.DataFrame(json.load(f))
    if update_path.exists():
        with open(update_path, "r", encoding="utf-8") as f:
            update_df = pd.DataFrame(json.load(f))
    return trade_df, update_df


def normalize_selected(selected):
    if selected is None:
        return []
    if isinstance(selected, pd.DataFrame):
        return selected.to_dict("records")
    if isinstance(selected, dict):
        return [selected]
    if isinstance(selected, list):
        return selected
    return []


def validate_trading_data(df):
    errors = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        s = f"Row {i}: "
        if not str(row.Series).strip():
            errors.append(s + "Series is required.")
        try:
            float(row.Qty)
        except Exception:
            errors.append(s + "Qty must be numeric.")
        if row.Type not in ['Stock', 'Future', 'Option']:
            errors.append(s + "Type must be one of: Stock, Future, Option.")
        if row.status not in ['Open', 'Closed', 'Expired']:
            errors.append(s + "status must be one of: Open, Closed, Expired.")
        # if row.Type == 'Option' and (not getattr(row, 'Strike price', 0)):
        #     errors.append(s + "Strike price required for Option.")
        if row.status == 'Expired' and (not getattr(row, 'Underlying price', 0)):
            errors.append(s + "Underlying price required for Expired trade.")
    return errors


def validate_update_data(df):
    errors = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        s = f"Row {i}: "
        if not str(row.Series).strip():
            errors.append(s + "Series is required.")
        try:
            float(row._asdict().get('Update Price', 0))
        except Exception:
            errors.append(s + "Update Price must be numeric.")
    return errors


def calculate_option_pl(df):
    df['payoff'] = np.where(
        df['Type'] == "Call",
        np.maximum(df['Underlying price'] - df['Strike price'], 0),
        np.maximum(df['Strike price'] - df['Underlying price'], 0)
    )
    df['P&L'] = (df['payoff'] - df['Entry price']) * df['Qty'] * df['MULTIPLER']
    return df


def calculate_metrics(df_trades, df_updates):
    if df_trades.empty:
        return pd.DataFrame()

    merged = pd.merge(df_trades, df_updates, on='Series', how='left')
    for c in ['Qty','Entry price','Exit price','Strike price','Update Price','fee','MULTIPLER','Underlying price']:
        merged[c] = pd.to_numeric(merged.get(c, 0), errors='coerce').fillna(0)

    merged['P&L'] = 0
    # Closed
    closed = merged[merged['status'] == 'Closed']
    merged.loc[closed.index, 'P&L'] = (closed['Exit price'] - closed['Entry price']) * closed['Qty'] * closed['MULTIPLER']
    # Open
    openpos = merged[merged['status'] == 'Open']
    merged.loc[openpos.index, 'P&L'] = (openpos['Update Price'] - openpos['Entry price']) * openpos['Qty'] * openpos['MULTIPLER']
    # Expired
    exp = merged[merged['status'] == 'Expired']
    if not exp.empty:
        exp = calculate_option_pl(exp)
        merged.loc[exp.index, 'P&L'] = exp['P&L']

    return merged


# ============ Streamlit App ============
def app():
    st.title("📊 Trading Portfolio Dashboard")
    currency_unit = st.sidebar.text_input("Currency Unit", value="THB")

    trade_df, update_df = load_data()
    if "trading_df" not in st.session_state:
        st.session_state["trading_df"] = trade_df.copy()
    if "update_df" not in st.session_state:
        st.session_state["update_df"] = update_df.copy()

    tab1, tab2 = st.tabs(["Trading Data", "Update Data"])

    # --- Trading Data Grid ---
    with tab1:
        df = st.session_state["trading_df"]
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True, sortable=True, filter=True, resizable=True)
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_grid_options(domLayout='normal', headerHeight=32)
        grid_response = AgGrid(df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            theme="balham",
            height=450,
            allow_unsafe_jscode=True
        )
        edited_df = pd.DataFrame(grid_response["data"])
        selected = normalize_selected(grid_response.get("selected_rows"))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("➕ Add Row"):
                new_row = {col: "" for col in df.columns}
                edited_df = pd.concat([edited_df, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state["trading_df"] = edited_df
                st.rerun()
        with c2:
            if selected and st.button("🗑 Delete Selected"):
                idx = selected[0].get("_selectedRowNodeInfo", {}).get("nodeRowIndex", None)
                if idx is not None:
                    edited_df = edited_df.drop(index=int(idx)).reset_index(drop=True)
                st.session_state["trading_df"] = edited_df
                st.rerun()
        with c3:
            if selected and st.button("📋 Clone Selected"):
                sel = selected[0].copy()
                sel.pop("_selectedRowNodeInfo", None)
                edited_df = pd.concat([edited_df, pd.DataFrame([sel])], ignore_index=True)
                st.session_state["trading_df"] = edited_df
                st.rerun()
        with c4:
            if st.button("💾 Save Trading JSON"):
                errors = validate_trading_data(edited_df)
                if errors:
                    st.error("⚠️ Please fix before saving:")
                    for e in errors:
                        st.markdown(f"- {e}")
                else:
                    path = Path("data/trading.json")
                    os.makedirs(path.parent, exist_ok=True)
                    edited_df.to_json(path, orient='records', indent=2, force_ascii=False)
                    st.success("✅ Saved trading.json")
                    st.session_state["trading_df"] = edited_df
                    st.rerun()

    # --- Update Data Grid ---
    with tab2:
        df2 = st.session_state["update_df"]
        gb2 = GridOptionsBuilder.from_dataframe(df2)
        gb2.configure_default_column(editable=True, sortable=True, filter=True, resizable=True)
        gb2.configure_selection("single", use_checkbox=True)
        gb2.configure_grid_options(domLayout='normal', headerHeight=32)
        grid_response2 = AgGrid(df2,
            gridOptions=gb2.build(),
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            theme="balham",
            height=350,
            allow_unsafe_jscode=True
        )
        edited_df2 = pd.DataFrame(grid_response2["data"])
        selected2 = normalize_selected(grid_response2.get("selected_rows"))

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("➕ Add Row (Update)"):
                new_row = {col: "" for col in df2.columns}
                edited_df2 = pd.concat([edited_df2, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state["update_df"] = edited_df2
                st.rerun()
        with c2:
            if selected2 and st.button("🗑 Delete Selected (Update)"):
                idx = selected2[0].get("_selectedRowNodeInfo", {}).get("nodeRowIndex", None)
                if idx is not None:
                    edited_df2 = edited_df2.drop(index=int(idx)).reset_index(drop=True)
                st.session_state["update_df"] = edited_df2
                st.rerun()
        with c3:
            if st.button("💾 Save Update JSON"):
                errors = validate_update_data(edited_df2)
                if errors:
                    st.error("⚠️ Please fix before saving:")
                    for e in errors:
                        st.markdown(f"- {e}")
                else:
                    path = Path("data/update.json")
                    os.makedirs(path.parent, exist_ok=True)
                    edited_df2.to_json(path, orient='records', indent=2, force_ascii=False)
                    st.success("✅ Saved update.json")
                    st.session_state["update_df"] = edited_df2
                    st.rerun()

    st.divider()

    # --- Portfolio Summary ---
    edited_df = st.session_state["trading_df"]
    edited_df2 = st.session_state["update_df"]
    processed = calculate_metrics(edited_df, edited_df2)

    if not processed.empty:
        st.subheader("Portfolio Performance Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Realized P&L", f"{processed[processed['status']=='Closed']['P&L'].sum():,.2f} {currency_unit}")
        col2.metric("Unrealized P&L", f"{processed[processed['status']=='Open']['P&L'].sum():,.2f} {currency_unit}")
        col3.metric("Total Trades", len(processed))

        st.subheader("P&L by Series")
        pl_series = processed.groupby("Series")["P&L"].sum().reset_index()
        st.plotly_chart(px.bar(pl_series, x="Series", y="P&L", color="P&L", color_continuous_scale="Tealrose"))

        st.subheader("Cumulative P&L Over Time")
        processed["Entry date"] = pd.to_datetime(processed.get("Entry date"), errors="coerce")
        processed = processed.sort_values("Entry date")
        processed["Cumulative P&L"] = processed["P&L"].cumsum()
        st.plotly_chart(px.line(processed, x="Entry date", y="Cumulative P&L"))
    else:
        st.info("No valid trading data loaded yet.")

    # --- Detailed Report Section (with P&L) ---
    processed_df = processed
    st.header("Detailed Trading Report")

    display_columns = ['Type','status','Cumulative P&L','P&L','Series', 'Qty', 'Entry price', 'Exit price', 'Update Price','Underlying price', 'fee','payoff', 'Comment','Strike price','Entry date']
    formatted_df = processed_df.copy() if not processed_df.empty else pd.DataFrame()
    if not formatted_df.empty:
        formatted_df['Entry date'] = pd.to_datetime(formatted_df['Entry date'], errors='coerce').dt.strftime('%d/%m/%Y')
        formatted_df['Exit date'] = pd.to_datetime(formatted_df.get('Exit date'), errors='coerce').dt.strftime('%d/%m/%Y')
        existing_columns = [col for col in display_columns if col in formatted_df.columns]

        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'

        st.dataframe(formatted_df[existing_columns].style.map(lambda v: 'color: green' if isinstance(v, (int,float)) and v > 0 else ('color: red' if isinstance(v, (int,float)) and v < 0 else 'color: black'), subset=['P&L']).format({
            'Qty': '{:.0f}',
            'Entry price': '{:,.2f}',
            'Exit price': '{:,.2f}',
            'Strike price': '{:,.2f}',
            'fee': '{:,.2f}',
            'Update Price': '{:,.2f}',
            'Underlying price': '{:,.2f}',
            'P&L': '{:,.2f}',
            'payoff': '{:,.2f}',
            'Cumulative P&L': '{:,.2f}'
        }))
    else:
        st.info("No detailed trades to display.")

if __name__ == "__main__":
    app()
