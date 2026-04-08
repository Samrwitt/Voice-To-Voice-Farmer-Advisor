import streamlit as st
import sqlite3
import pandas as pd
import bcrypt
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "/data/advisor.db"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="/data/chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection = chroma_client.get_or_create_collection(name="agronomy_kb", embedding_function=sentence_transformer_ef)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def setup_admin():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM admin_users")
        if c.fetchone()[0] == 0:
            pw_hash = bcrypt.hashpw("admin".encode('utf-8'), bcrypt.gensalt())
            c.execute("INSERT INTO admin_users (username, password_hash, role) VALUES (?, ?, ?)", 
                      ("admin", pw_hash.decode('utf-8'), "admin"))
            conn.commit()
    except Exception as e:
        pass
    finally:
        conn.close()

def login_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM admin_users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        stored_hash = row[0].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return row[1]
    return None

st.set_page_config(page_title="Farmer Advisor Dashboard", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = ""

setup_admin()

if not st.session_state.logged_in:
    st.title("Login to Farmer Advisor Admin/Expert Portal")
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Invalid credentials")
else:
    st.sidebar.title("Navigation")
    pages = ["Escalation Queue", "Market Prices", "Knowledge Base", "Alerts & Forecasts"]
    choice = st.sidebar.radio("Go to", pages)
    st.sidebar.write(f"Logged in as: **{st.session_state.role}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if choice == "Escalation Queue":
        st.title("Escalation Queue")
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM escalated_queries ORDER BY timestamp DESC", conn)
        conn.close()
        
        if df.empty:
            st.info("No escalations found.")
        else:
            st.dataframe(df, use_container_width=True)
            ticket_id = st.number_input("Resolve Ticket ID:", min_value=1, step=1)
            if st.button("Mark Resolved"):
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("UPDATE escalated_queries SET status='resolved' WHERE id=?", (ticket_id,))
                conn.commit()
                conn.close()
                st.success(f"Ticket {ticket_id} resolved!")
                st.rerun()

    elif choice == "Market Prices":
        st.title("Manage Expected Market Prices")
        with st.form("Add Market Price"):
            crop = st.text_input("Crop Name (e.g., Teff)")
            region = st.text_input("Region (e.g., Addis Ababa)")
            price = st.number_input("Price")
            unit = st.text_input("Unit (e.g., Kg)")
            submit = st.form_submit_button("Submit")
            if submit and crop:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("INSERT INTO market_prices (crop_name, region, price, unit) VALUES (?, ?, ?, ?)",
                          (crop, region, price, unit))
                conn.commit()
                conn.close()
                st.success(f"Added price for {crop}.")
                
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM market_prices ORDER BY updated_at DESC", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)

    elif choice == "Knowledge Base":
        st.title("Manage ChromaDB Knowledge Base")
        if st.session_state.role != "admin":
            st.error("Permission Denied: Only Admins can modify the KB.")
        else:
            with st.form("Add KB Entry"):
                intent = st.text_input("Intent name (e.g., crop_disease)")
                response = st.text_area("Amharic Response text")
                submit = st.form_submit_button("Add to Vector Store")
                if submit and intent and response:
                    doc_id = f"kb_{collection.count() + 1}"
                    collection.add(
                        documents=[response],
                        metadatas=[{"intent": intent}],
                        ids=[doc_id]
                    )
                    st.success(f"Successfully added document ID: {doc_id}")

    elif choice == "Alerts & Forecasts":
        st.title("Predictive Summary & Alerts Broadcast")
        if st.session_state.role != "admin":
            st.error("Permission Denied: Only Admins can broadcast alerts.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Broadcast New Alert")
                with st.form("Add Alert"):
                    target_region = st.selectbox("Target Region", ["all", "Addis Ababa", "Oromia", "Amhara", "SNNPR", "Tigray"])
                    alert_message = st.text_area("Alert Message (Amharic)")
                    severity = st.selectbox("Severity", ["info", "warning", "critical"])
                    submit_alert = st.form_submit_button("Broadcast")
                    
                    if submit_alert and alert_message:
                        conn = get_db_connection()
                        c = conn.cursor()
                        c.execute("INSERT INTO alerts (target_region, alert_message, severity) VALUES (?, ?, ?)",
                                  (target_region, alert_message, severity))
                        conn.commit()
                        conn.close()
                        st.success("Alert Broadcasted Successfully!")

            with col2:
                st.subheader("Predictive Analytics")
                conn = get_db_connection()
                try:
                    df_alerts = pd.read_sql_query("SELECT target_region, severity, count(*) as count FROM alerts GROUP BY target_region, severity", conn)
                    if not df_alerts.empty:
                        st.bar_chart(df_alerts.pivot(index='target_region', columns='severity', values='count').fillna(0))
                    else:
                        st.info("No active alerts.")
                except Exception as e:
                    pass
                conn.close()
                st.markdown("**Simulated Reach:** ~45,000 farmers in active warning zones.")
                
        st.subheader("Recent Broadcasts")
        conn = get_db_connection()
        try:
            df = pd.read_sql_query("SELECT id, target_region, alert_message, severity, created_at FROM alerts ORDER BY created_at DESC", conn)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.warning("Could not load alerts. Make sure the database schema is updated.")
        conn.close()
