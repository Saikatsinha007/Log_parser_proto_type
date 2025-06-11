# 🧠 Advanced Log Parser with Streamlit + Drain3

A powerful and interactive log parsing application that allows users to upload raw log files, extract structured fields like timestamp, IP address, username, log level, and parameters using the Drain3 algorithm, and interactively approve or reject entries via an intuitive Streamlit UI.

---

## 🚀 Features

- 📄 Upload and parse `.txt` log files
- 🧠 Intelligent log parsing using [Drain3](https://github.com/logpai/Drain3)
- 🔍 Extract key fields: timestamp, log level, IP address, username, message template, and parameters
- ✅ Interactive review with approve/reject/pending actions
- ✏️ Edit extracted templates and parameters
- 🔄 Filter logs by log level
- 📤 Export approved and rejected logs as JSON files

---

## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Log Parsing**: [Drain3](https://github.com/logpai/Drain3)
- **Data Handling**: Pandas
- **Programming Language**: Python 3.8+

---

## 📁 Project Structure

```

log\_parser\_app/
├── app.py               # Main Streamlit app UI
├── log\_parser.py        # Log parsing logic using Drain3
├── drain3\_state.json    # Drain3 persistence file
├── example\_logs.txt     # Sample log file (optional)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

````

---

## 🔧 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Saikatsinha007/Log_parser_proto_type.git
cd Log_parser_proto_type
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📄 Example Log Format

Supported logs should follow this approximate pattern:

```
[2025-06-11 10:42:13] INFO: User 'john_doe' logged in from IP 192.168.1.10
```

---

## 💡 How to Use

1. Upload a `.txt` log file.
2. The parser will extract:

   * Timestamp
   * Log Level
   * IP Address
   * Username
   * Message Template
   * Parameters
3. Review each log entry inside expandable sections.
4. Edit templates/parameters and select **Approve**, **Reject**, or leave as **Pending**.
5. Use the buttons to download:

   * ✅ `approved_logs.json`
   * ❌ `rejected_logs.json`

---

## 📤 Output Format

Each exported JSON log includes:

```json
{
  "log_id": "uuid",
  "raw": "[timestamp] LEVEL: message...",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "level": "INFO",
  "template": "User <*> logged in from IP <*>",
  "parameters": ["john_doe", "192.168.1.10"],
  "username": "john_doe",
  "ip": "192.168.1.10",
  "status": "Approve"
}
```

---

## 👥 Contributors

* [👩‍💻 Oppangi Poojita](https://github.com/)
* [👨‍💻 Saikat Sinha](https://github.com/Saikatsinha007)

We collaboratively designed and developed this tool to streamline log analysis and classification workflows in a user-friendly and intelligent manner.
