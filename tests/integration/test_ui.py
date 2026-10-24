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


def test_ui_has_conversations_sidebar_and_docs_modal(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert 'id="new-chat-btn"' in html
    assert 'id="conversations-list"' in html
    assert 'id="docs-modal"' in html
    assert 'id="open-docs-modal-btn"' in html
    assert 'id="upload-form"' in html
    assert 'id="docs-list"' in html


def test_app_js_handles_conversations_api(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    js = r.text
    assert "/conversations" in js
    assert "activeConversationId" in js


def test_ui_has_download_conversation_button(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert 'id="download-btn"' in html
    assert "Descargar conversación (.txt)" in html


def test_app_js_downloads_full_conversation_with_user_and_bot_prefixes(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    js = r.text
    assert "formatConversationForDownload" in js
    assert "USUARIO:" in js
    assert "BOT:" in js
    assert "currentConversationMessages" in js


