import streamlit as st
import pandas as pd
import json
from log_parser import parse_logs  # make sure this exists and works

st.set_page_config(page_title="Advanced Log Parser", layout="wide")
st.title("🧠 Log Parser with Project Pulse")

uploaded_file = st.file_uploader("📄 Upload your log file", type=["txt"])

# Load and parse logs
if uploaded_file:
    if "log_data" not in st.session_state:
        log_text = uploaded_file.read().decode("utf-8")
        structured_logs = parse_logs(log_text)
        if not structured_logs:
            st.warning("No valid log lines parsed.")
        else:
            st.session_state.log_data = pd.DataFrame(structured_logs)

# Continue only if logs were parsed
if "log_data" in st.session_state:
    df = st.session_state.log_data

    # Sidebar filtering
    st.sidebar.header("🔍 Filter")
    levels = df["level"].dropna().unique().tolist()
    selected_levels = st.sidebar.multiselect("Select log levels", levels, default=levels)
    filtered_df = df[df["level"].isin(selected_levels)]

    st.subheader("📋 Log Entries")

    for idx in filtered_df.index:
        row = df.loc[idx]

        with st.expander(f"[{row['level']}] {row['timestamp']}"):

            st.code(row["raw"], language="text")

            st.markdown(f"""
            - 🧱 **Template**: `{row["template"]}`
            - 🙍 **User**: `{row.get("username", "N/A")}`
            - 🌐 **IP Address**: `{row.get("ip", "N/A")}`
            - 🕒 **Timestamp**: `{row.get("timestamp", "N/A")}`
            - 🧩 **Parameters**: `{", ".join(row["parameters"]) if row["parameters"] else "N/A"}`
            """)

            # Editable template and parameters
            template_input = st.text_area("✏️ Edit Template", row["template"], key=f"tpl_{row['log_id']}")
            params_input = st.text_area("✏️ Edit Parameters (comma-separated)", ", ".join(row["parameters"]), key=f"prm_{row['log_id']}")

            df.at[idx, "template"] = template_input
            df.at[idx, "parameters"] = [x.strip() for x in params_input.split(",")]

            # Action dropdown (Approve / Reject / Pending)
            action = st.selectbox(
                "✅ Action",
                ["Pending", "Approve", "Reject"],
                index=["Pending", "Approve", "Reject"].index(row["status"]),
                key=f"st_{row['log_id']}"
            )
            df.at[idx, "status"] = action

    # Export section
    approved_logs = df[df["status"] == "Approve"]
    rejected_logs = df[df["status"] == "Reject"]

    st.subheader("📤 Export Logs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Approved Logs")
        if not approved_logs.empty:
            st.download_button(
                "⬇️ Download Approved JSON",
                data=json.dumps(approved_logs.to_dict(orient="records"), indent=2),
                file_name="approved_logs.json",
                mime="application/json"
            )
        else:
            st.info("No approved logs yet.")

    with col2:
        st.markdown("### ❌ Rejected Logs")
        if not rejected_logs.empty:
            st.download_button(
                "⬇️ Download Rejected JSON",
                data=json.dumps(rejected_logs.to_dict(orient="records"), indent=2),
                file_name="rejected_logs.json",
                mime="application/json"
            )
        else:
            st.info("No rejected logs yet.")
