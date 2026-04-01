import streamlit as st
import sqlite3
import pandas as pd

# Path connected via docker volume
DB_PATH = "/data/escalation_queue.db"

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM escalated_queries ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty if db doesn't exist yet

def update_status(query_id, new_status):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE escalated_queries SET status = ? WHERE id = ?", (new_status, query_id))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error updating: {e}")

st.set_page_config(page_title="Amharic Farmer Advisor Dashboard", layout="wide")
st.title("🌾 Amharic Farmer Advisor - Admin Dashboard")
st.markdown("Monitor unanswerable queries and manage the escalation queue.")

df = load_data()

if df.empty:
    st.info("No data available yet in the escalation queue.")
else:
    st.subheader("Escalation Queue")
    
    # Filter by status
    status_filter = st.selectbox("Filter by Status", ["All", "pending", "resolved"])
    if status_filter != "All":
        display_df = df[df["status"] == status_filter]
    else:
        display_df = df
        
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Resolve a Ticket")
    col1, col2 = st.columns(2)
    with col1:
        ticket_id = st.number_input("Enter ID of query to mark resolved", min_value=1, step=1)
    with col2:
        st.write("")
        st.write("")
        if st.button("Mark as Resolved"):
            update_status(ticket_id, "resolved")
            st.success(f"Ticket {ticket_id} updated! Refreshing...")
            st.rerun()
