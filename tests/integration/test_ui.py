def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "TechnicalDoc AI Assistant" in r.text


def test_static_assets_are_served(client):
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "/query/stream" in js.text
    css = client.get("/static/style.css")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]


def test_static_missing_asset_404(client):
    assert client.get("/static/nope.js").status_code == 404


def test_api_routes_still_work_with_ui_mounted(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
