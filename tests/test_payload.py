
import requests, io

def test_contract():
    # This test is illustrative (won't run in CI without GPUs)
    url = "http://127.0.0.1:8093/generate_video/"
    resp = requests.post(url, data={"prompt": "pink bicycle"})
    assert resp.status_code in (200, 204)
    if resp.status_code == 200:
        assert resp.headers.get("content-type","").startswith("video/")
