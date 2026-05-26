import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami_utils import ami_action, sip_endpoint_for_alert


def test_sip_endpoint_for_alert_strips_sip_uri():
    assert sip_endpoint_for_alert("sip:farmeruhamayohannes@localhost") == "farmeruhamayohannes"


def test_ami_action_uses_crlf_terminator():
    payload = ami_action({"Action": "Ping"})

    assert payload == b"Action: Ping\r\n\r\n"
