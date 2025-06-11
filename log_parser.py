import uuid
import re
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence

# Setup Drain3
persistence = FilePersistence("drain3_state.json")
template_miner = TemplateMiner(persistence)

# Regex helpers
def extract_ip(text):
    match = re.search(r"(?:\d{1,3}\.){3}\d{1,3}", text)
    return match.group() if match else None

def extract_username(text):
    match = re.search(r"User\s'([^']+)'", text)
    return match.group(1) if match else None

def extract_timestamp(text):
    match = re.match(r"\[(.*?)\]", text)
    return match.group(1) if match else None

def parse_logs(log_text: str):
    structured_logs = []
    lines = log_text.strip().splitlines()

    for line in lines:
        result = template_miner.add_log_message(line)
        if not result or "template_mined" not in result:
            continue

        structured_logs.append({
            "log_id": str(uuid.uuid4()),
            "raw": line,
            "timestamp": extract_timestamp(line),
            "level": line.split("]")[1].split(":")[0].strip() if "]" in line and ":" in line else "UNKNOWN",
            "template": result.get("template_mined", ""),
            "parameters": result.get("parameter_list", []),
            "username": extract_username(line),
            "ip": extract_ip(line),
            "status": "Pending"
        })

    return structured_logs
