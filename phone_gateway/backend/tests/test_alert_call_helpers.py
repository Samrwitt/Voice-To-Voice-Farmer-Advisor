import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami_utils import ami_action, sip_endpoint_for_alert
from tts_chunking import chunk_tts_text


def test_sip_endpoint_for_alert_strips_sip_uri():
    assert sip_endpoint_for_alert("sip:farmeruhamayohannes@localhost") == "farmeruhamayohannes"


def test_ami_action_uses_crlf_terminator():
    payload = ami_action({"Action": "Ping"})

    assert payload == b"Action: Ping\r\n\r\n"


def test_chunk_tts_text_splits_long_expert_message():
    text = (
        "የባለሙያ መልስ ዝግጁ ነው። በማሳዎ ላይ የተባይ ምልክት ካዩ መጀመሪያ "
        "ተጎዳውን ቅጠል ያስወግዱ ከዚያም በአቅራቢያዎ ያለውን የግብርና ባለሙያ ያነጋግሩ።"
    )
    chunks = chunk_tts_text(text, max_chars=70)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
