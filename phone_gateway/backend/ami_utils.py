import os


def sip_endpoint_for_alert(phone_number: str) -> str:
    raw = (phone_number or "").strip()
    default_endpoint = os.getenv("SIP_ALERT_DEFAULT_ENDPOINT", "farmeruhamayohannes").strip()
    if raw.startswith("sip:"):
        raw = raw[4:]
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    raw = raw.strip().strip("/")
    return raw or default_endpoint


def ami_action(headers: dict[str, str]) -> bytes:
    lines = [f"{key}: {value}" for key, value in headers.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
