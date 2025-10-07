import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
# from io import StringIO
# Add parent directory (app/) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Load Data ---
def load_data():
    """Load and process trading and update data from JSON files."""
    try:
        BASE_DIR = Path(__file__).resolve().parent
        DEFAULT_TRADING_PATH = BASE_DIR / "data" / "trading.json"
        DEFAULT_UPDATE_PATH = BASE_DIR / "data" / "update.json"

        with open(DEFAULT_TRADING_PATH, 'r', encoding='utf-8') as f:
            trading_data = json.load(f)
        trading_df = pd.DataFrame(trading_data)

        with open(DEFAULT_UPDATE_PATH, 'r', encoding='utf-8') as f:
            update_data = json.load(f)
        update_df = pd.DataFrame(update_data)
        # if trading_df is None:
        #     trading_df = pd.DataFrame() 
        # if update_df is None:
        #     update_df == pd.DataFrame() 

        # Merge dataframes to have a single source of truth
        # Note: We don't need to merge here anymore, as it will be done inside calculate_metrics
        
        return trading_df, update_df
    except FileNotFoundError:
        st.error("Make sure 'trading.json' and 'update.json' files are in the Data directory.")
        return None, None
    
#option


def calculate_option_pl(df: pd.DataFrame, at_expiry=True):
    """
    คำนวณ Payoff และ P&L ของ Options
    
    Parameters
    ----------
    df : pd.DataFrame
        ต้องมีคอลัมน์:
        ['Type', 'Underlying price', 'Strike price', 'Entry price', 
         'Exit price', 'Qty', 'MULTIPLER']
    at_expiry : bool
        ถ้า True = คำนวณตอนหมดอายุ (Expiry) โดยใช้ Underlying price
        ถ้า False = คำนวณตอนปิดก่อนหมดอายุ (Close before expiry)
    
    Returns
    -------
    pd.DataFrame
        DataFrame พร้อม payoff และ P&L
    """
    
    processed_df = df.copy()
    
    if at_expiry:
        # Payoff (intrinsic value)
        processed_df['payoff'] = np.where(
            processed_df['Type'] == "Call",
            np.maximum(processed_df['Underlying price'] - processed_df['Strike price'], 0),
            np.maximum(processed_df['Strike price'] - processed_df['Underlying price'], 0)
        )
        
        # P&L = (Payoff - Entry premium) * Qty * Multiplier
        processed_df['P&L'] = (processed_df['payoff'] - processed_df['Entry price']) * processed_df['Qty'] * processed_df['MULTIPLER']
    
    else:
        # Close before expiry → P&L from premium difference
        processed_df['payoff'] = np.nan  # ไม่มี payoff จริง
        processed_df['P&L'] = (processed_df['Exit price'] - processed_df['Entry price']) * processed_df['Qty'] * processed_df['MULTIPLER']
    
    return processed_df


# --- Calculations ---
def calculate_metrics(df_trades, df_updates):
    """Calculate P&L for each trade and overall portfolio metrics."""
    # Re-merge the updated trading data with the current prices
    processed_df = pd.merge(df_trades, df_updates, on='Series', how='left')
    # Convert relevant columns to numeric, including the new 'fee' column
    for col in ['Qty', 'Entry price', 'Exit price', 'Strike price', 'Update Price', 'fee','MULTIPLER','Underlying price']:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    # Calculate P&L for all trades
    processed_df['P&L'] = 0.0
    processed_df['payoff'] = 0.0

    # Calculate P&L for 'Closed' trades
    closed_trades = processed_df[processed_df['status'] == 'Closed'].copy()
    long_closed = closed_trades[closed_trades['Qty'] > 0]
    short_closed = closed_trades[closed_trades['Qty'] < 0]
    processed_df.loc[long_closed.index, 'P&L'] = (long_closed['Exit price'] - long_closed['Entry price']) * long_closed['Qty']* long_closed['MULTIPLER']
    processed_df.loc[short_closed.index, 'P&L'] = (short_closed['Entry price'] - short_closed['Exit price']) * abs(short_closed['Qty'])* short_closed['MULTIPLER']
    # Apply round-trip fee for closed trades
    processed_df.loc[closed_trades.index, 'P&L'] -= (closed_trades['fee'] * abs(closed_trades['Qty'])) * 2

    # Calculate P&L for 'Open' trades using update price
    open_trades = processed_df[processed_df['status'] == 'Open'].copy()
    long_open = open_trades[open_trades['Qty'] > 0]
    short_open = open_trades[open_trades['Qty'] < 0]
    processed_df.loc[long_open.index, 'P&L'] = (long_open['Update Price'] - long_open['Entry price']) * long_open['Qty']* long_open['MULTIPLER']
    processed_df.loc[short_open.index, 'P&L'] = (short_open['Entry price'] - short_open['Update Price']) * abs(short_open['Qty'])* short_open['MULTIPLER']
    # Apply one-way fee for open trades
    processed_df.loc[open_trades.index, 'P&L'] -= open_trades['fee'] * abs(open_trades['Qty'])

    # Calculate P&L for 'Expired' trades
    expired_trades = processed_df[processed_df['status'] == 'Expired'].copy()

    #ITM strike<underlying
    #OTM strike>underlying
    # call_expired = expired_trades[expired_trades['Type'] == "Call"]
    # put_expired = expired_trades[expired_trades['Type'] == "Put"]
    # # Call options payoff
    # processed_df.loc[call_expired.index, 'payoff'] = (call_expired['Underlying price'] - call_expired['Strike price']).clip(lower=0)
    # # Put options payoff
    # processed_df.loc[put_expired.index, 'payoff'] = (put_expired['Strike price'] - put_expired['Underlying price']).clip(lower=0)
    # # P&L calculation
    # processed_df.loc[call_expired.index, 'P&L'] = (processed_df.loc[call_expired.index, 'payoff'] - call_expired['Entry price']) * call_expired['Qty'] * call_expired['MULTIPLER']
    # processed_df.loc[put_expired.index, 'P&L'] = (processed_df.loc[put_expired.index, 'payoff'] - put_expired['Entry price']) * put_expired['Qty'] * put_expired['MULTIPLER']
    expired_trades = calculate_option_pl(expired_trades, at_expiry=True)
    # print(expired_trades)
    processed_df.loc[expired_trades.index, 'payoff'] = expired_trades['payoff']
    processed_df.loc[expired_trades.index, 'P&L'] = expired_trades['P&L']
    # print(processed_df)
    # pl = (payoff - premium) * qty * multiplier

    # Apply one-way fee for expired trades (since it's a one-way transaction)
    processed_df.loc[expired_trades.index, 'P&L'] -= expired_trades['fee'] * abs(expired_trades['Qty'])

    return processed_df

# --- Streamlit App Layout ---
def app():
    st.set_page_config(layout="wide", page_title="Professional Trading Portfolio Report",
                       page_icon="📈")
    # Apply custom branding
    st.markdown("""
        <style>
            .st-emotion-cache-18ni7ap { font-family: Kanit; font-weight: bold; }
            .st-emotion-cache-18ni7ap { color: #000080; }
            .st-emotion-cache-1f19z12 { color: #008080; }
            body { background-color: #F5F5F5; }
            .stDataFrame, .stTable, .st-emotion-cache-13ln4j2, .st-emotion-cache-e1y03s2 { font-family: Sarabun; }
            }
        </style>
    """, unsafe_allow_html=True)
        
    st.title("Performance")
    st.sidebar.title("Trading Portfolio")
    st.sidebar.header("(stock,future,option)")
    st.sidebar.caption("Made by [THEO]")
    st.sidebar.caption("Check out ")
    st.sidebar.caption("My website simplifies the financial statements of Thailand listed companies(SET index), making them easier to understand. [here](https://setviz.pages.dev/).")
    st.sidebar.caption("Tweets  [here](https://x.com/theoris/).")
    st.sidebar.divider()
    st.sidebar.header("settings")
    currency_unit = st.sidebar.text_input("curreny unit", value= "THB")
                
    trading_df, update_df = load_data()
    st.markdown("""
        - Let's upload your file or use server file for guide first or edit and save to your local.
        - please read Data entry guide as below ( trading/update file)
        - anything would like to know more please feel free to contact me
        """)

    st.subheader("Upload file")
    tab1, tab2 = st.tabs(["Trading data", "Update data"])
    with tab1:
        uploaded_file = st.file_uploader("Choose a trading file", type=["json"])
        trading_df0 = pd.DataFrame([{
            "Entry date": "2025-05-16",
            "Series": "S50H26",
            "Qty": 1,
            "Entry price": 765.0,
            "Exit date": None,
            "Exit price": "",
            "Strike price": None,
            "fee": 83.57,
            "MULTIPLER": 200,
            "Underlying price": None,
            "Type": "Future",
            "Comment": None,
            "status": "Open"
        }])
        # Convert the first row to JSON (just as a schema/sample)
        sample_json = trading_df0.head(1).to_dict(orient="records")
        pretty_json = json.dumps(sample_json, indent=2, ensure_ascii=False)
        with st.expander("📘 Data Entry Guide of trading.json"):
            st.markdown("""
            - Stock → `Qty`: only positive      
            - Option,Future → `Qty`: positive = Long, negative = Short  
            - `Type`: Stock, Future, Option  
            - `status`: Open, Closed, Expiry  
            - If `Type` = Option → must fill Strike price  
            - If Option expired → set status = Expiry + fill Underlying price  
            - `fee`: including tax per 1 Qty
            - If you edit  Data on table don't forget to click to Save Changes to JSON file.(/data/trading.json,/data/update.json)
                        
            """)
            st.write("Structure of trading.json file:")
            st.code(pretty_json, language="json")            
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".json"):
                    trading_df2 = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith(".csv"):
                    trading_df2 = pd.read_csv(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload JSON.")
                    trading_df2 = None
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                trading_df2 = None

            # If we successfully loaded a new DataFrame, replace trading_df
            if trading_df2 is not None and not trading_df2.empty:
                trading_df = trading_df2
                st.success("✅ Uploaded file loaded successfully!")
    with tab2:
        uploaded_updatefile = st.file_uploader("Choose a update file", type=["json"])
        update_df0 = pd.DataFrame([{
            "Series":"S50H26",
            "Update Price":822.0
        }])
        sample_json1 = update_df0.head(1).to_dict(orient="records")
        pretty_json1 = json.dumps(sample_json1, indent=2, ensure_ascii=False)        
        with st.expander("📘 Data Entry Guide of update.json"):
            st.markdown("""
            - `Series`: All of series,symbol that you have on trading file      
            - `Update Price`: lastest price 
            """)
            st.write("Structure of update.json file:")
            st.code(pretty_json1, language="json")            
        if uploaded_updatefile is not None:
            try:
                if uploaded_updatefile.name.endswith(".json"):
                    update_df2 = pd.read_json(uploaded_updatefile)
                elif uploaded_updatefile.name.endswith(".csv"):
                    update_df2 = pd.read_csv(uploaded_updatefile)
                else:
                    st.error("Unsupported file type. Please upload JSON.")
                    update_df2 = None
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                update_df2 = None

            # If we successfully loaded a new DataFrame, replace trading_df
            if update_df2 is not None and not update_df2.empty:
                update_df = update_df2
                st.success("✅ Uploaded update file loaded successfully!")

    # Insert containers separated into tabs:
    st.subheader("Edit/Save/Download file")
    tab1, tab2 = st.tabs(["Trading data", "Update data"])
    with tab1:
        with st.expander("TradingData"):
            if trading_df is not None:
                # --- Editable Data Table Section ---
                st.header("1. Edit Trading Data")
                # Display the data editor and capture changes
                edited_df = st.data_editor(
                    trading_df[['Entry date', 'Series', 'Qty', 'Entry price', 'Exit date', 'Exit price', 'Strike price', 'fee', 'MULTIPLER','Underlying price','Type', 'Comment', 'status']],
                    hide_index=True,
                    num_rows="dynamic"
                )
            # Button to save changes back to the JSON file
            if st.button("Save Changes: Trading data JSON"):
                edited_df.to_json("data/trading.json", orient='records', indent=2)
                st.success("Changes saved successfully to trading.json!")
                st.rerun()
            st.download_button("Download trading.json", data=edited_df.to_json(index=False).encode(), file_name="trading.json", disabled=False)
    with tab2:
        with st.expander("Update Data"):
            if update_df is not None:
                st.header("2. Update Data")
                # Display the data editor and capture changes
                edited_df2 = st.data_editor(
                    update_df[['Series','Update Price']],
                    hide_index=True,
                    num_rows="dynamic"
                )
            # Button to save changes back to the JSON file
            if st.button("Save Changes: Update data JSON"):
                edited_df2.to_json("data/update.json", orient='records', indent=2)
                st.success("Changes saved successfully to update.json!")
                st.rerun()
            st.download_button("Download update.json", data=edited_df2.to_json(index=False).encode(), file_name="update.json", disabled=False)

    st.markdown("---")

    if trading_df is not None and update_df is not None:
        # --- Recalculate and Display Metrics with the edited data ---
        processed_df = calculate_metrics(edited_df, update_df)

        st.header("Performance Summary")
        col1, col2, col3, col4 = st.columns(4)

        total_realized_pl = processed_df[processed_df['status'] == 'Closed']['P&L'].sum()
        total_unrealized_pl = processed_df[processed_df['status'] == 'Open']['P&L'].sum()
        total_portfolio_pl = processed_df['P&L'].sum()
        num_trades = len(processed_df)

        col1.metric("Total Realized P&L", f"{total_realized_pl:,.2f} {currency_unit}", "profit" if total_realized_pl > 0 else "loss")
        col2.metric("Total Unrealized P&L", f"{total_unrealized_pl:,.2f} {currency_unit}", "profit" if total_unrealized_pl > 0 else "loss")
        col3.metric("Total Portfolio P&L", f"{total_portfolio_pl:,.2f} {currency_unit}", "profit" if total_portfolio_pl > 0 else "loss")
        col4.metric("Number of Trades", num_trades)

        st.markdown("---")

        # --- Charts Section ---
        st.header("Portfolio Visualization")

        st.subheader("P&L by Series")
        pl_by_series = processed_df.groupby('Series')['P&L'].sum().reset_index()
        fig_series = px.bar(pl_by_series, x='Series', y='P&L', 
                            title='P&L Distribution Across Assets',
                            color='P&L',
                            color_continuous_scale=px.colors.diverging.Tealrose)
        st.plotly_chart(fig_series)#, use_container_width=True)

        st.subheader("Cumulative Portfolio Value Over Time")
        processed_df['Entry date'] = pd.to_datetime(processed_df['Entry date'], errors='coerce')
        processed_df = processed_df.sort_values(by='Entry date')
        processed_df['Cumulative P&L'] = processed_df['P&L'].cumsum()
        fig_time = px.line(processed_df.dropna(subset=['Entry date']), x='Entry date', y='Cumulative P&L',
                            title='Portfolio P&L Over Time')
        st.plotly_chart(fig_time)#, use_container_width=True)

        st.markdown("---")

        # --- Detailed Report Section (with P&L) ---
        st.header("Detailed Trading Report")

        # Define the desired column order
        display_columns = ['Type','status','Cumulative P&L','P&L','Series', 'Qty', 'Entry price', 'Exit price', 'Update Price','Underlying price', 'fee','payoff', 'Comment','Strike price','Entry date']
        formatted_df = processed_df.copy()
        formatted_df['Entry date'] = formatted_df['Entry date'].dt.strftime('%d/%m/%Y')
        formatted_df['Exit date'] = pd.to_datetime(formatted_df['Exit date'], errors='coerce').dt.strftime('%d/%m/%Y')
        existing_columns = [col for col in display_columns if col in formatted_df.columns]
        # Function to color P&L values based on sign
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'

        st.dataframe(formatted_df[existing_columns].style.applymap(color_pnl, subset=['P&L']).format({
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
        }))#, use_container_width=True)

        # def color_pnl(val):
        #     color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        #     return f'color: {color}'

        # styler = (
        #     formatted_df[existing_columns]
        #     .style
        #     .applymap(color_pnl, subset=['P&L'])
        #     .format({
        #         'Qty': '{:.0f}',
        #         'Entry price': '{:,.2f}',
        #         'Exit price': '{:,.2f}',
        #         'Strike price': '{:,.2f}',
        #         'fee': '{:,.2f}',
        #         'Update Price': '{:,.2f}',
        #         'Underlying price': '{:,.2f}',
        #         'P&L': '{:,.2f}',
        #         'payoff': '{:,.2f}',
        #         'Cumulative P&L': '{:,.2f}'
        #     })
        #     .set_table_styles(
        #         {col: [{"selector": "th", "props": "text-align: center;"}] for col in existing_columns},
        #         overwrite=False
        #     )
        #     .hide(axis="index")   # optional: hides row index
        # )

        # st.table(styler)


if __name__ == "__main__":
    app()
