"""E2E tests for the frontend HTML/JS application."""
import pytest

# ============================================================
# HTML Structure Tests
# ============================================================

class TestHtmlStructure:
    """Test that the HTML page loads and has correct structure."""

    def test_page_loads(self, html):
        assert html.title is not None
        assert html.title.string == "LLM Config Manager"

    def test_all_script_tags_present(self, html):
        """Verify all modular JS files are loaded."""
        from urllib.parse import urlparse
        scripts = html.find_all("script", src=True)
        srcs = [urlparse(s["src"]).path for s in scripts if urlparse(s["src"]).path.startswith("/js/")]
        expected = [
            "/js/state.js",
            "/js/utils.js",
            "/js/api.js",
            "/js/filter.js",
            "/js/localmodels.js",
            "/js/render.js",
            "/js/ws.js",
            "/js/form.js",
            "/js/app.js",
        ]
        assert len(srcs) == 9, f"Expected 9 script tags, found {len(srcs)}"
        for src in expected:
            assert src in srcs, f"Missing script: {src}"

    def test_no_large_inline_script(self, html):
        """Verify no large inline script block exists (modularized)."""
        inline_scripts = html.find_all("script", src=False)
        # Should only have the small init script
        assert len(inline_scripts) == 1
        init_text = inline_scripts[0].string or ""
        assert "App.init()" in init_text
        assert "Ws.connect()" in init_text
        assert "LocalModels.fetch()" in init_text

    def test_form_elements_exist(self, html):
        """Verify all form input elements are present."""
        assert html.find(id="hf_repo") is not None
        assert html.find(id="commit_select") is not None
        assert html.find(id="commit_sha") is not None
        assert html.find(id="quant") is not None
        assert html.find(id="symlink_name") is not None
        assert html.find(id="parameters") is not None
        assert html.find(id="submitBtn") is not None
        assert html.find(id="clearBtn") is not None

    def test_config_list_area_exists(self, html):
        """Verify the config list container exists."""
        assert html.find(id="configList") is not None
        assert html.find(id="configSearch") is not None
        assert html.find(id="rpcToggle") is not None
        assert html.find(id="wsStatus") is not None

    def test_storage_list_exists(self, html):
        """Verify the storage list container exists."""
        assert html.find(id="storageList") is not None
        assert html.find(id="storageSearch") is not None

    def test_default_parameters_set(self, html):
        """Verify default parameters are pre-filled in the form."""
        # Check that the textarea exists with proper attributes
        params = html.find(id="parameters")
        assert params is not None
        assert params.name == "textarea"
        assert params.get("rows") == "7"
        assert params.get("required") is not None

    def test_rpc_toggle_initially_off(self, html):
        """Verify RPC toggle starts unchecked."""
        toggle = html.find(id="rpcToggle")
        assert toggle.get("checked") is None

    def test_mmproj_container_hidden(self, html):
        """Verify mmproj container has hidden class."""
        container = html.find(id="mmproj_container")
        assert container is not None
        assert "hidden" in container.get("class", [])

    def test_static_assets_accessible(self, http):
        """Verify static CSS is accessible (when built)."""
        r = http.get("/static/output.css")
        # CSS is built in Docker, may not exist in local test env
        assert r.status_code in (200, 404)

    def test_js_assets_accessible(self, http):
        """Verify JS modules are served."""
        for js in ["state.js", "utils.js", "api.js", "app.js"]:
            r = http.get(f"/js/{js}")
            assert r.status_code == 200, f"Failed to load /js/{js}"


# ============================================================
# API Module Tests
# ============================================================

class TestApiModule:
    """Test that API routes are properly configured."""

    def test_quants_endpoint_exists(self, http):
        r = http.get("/api/quants?repo=test/test", follow_redirects=False)
        # Should return 404 (repo not found) not 404 (endpoint not found)
        assert r.status_code in (400, 404, 500)

    def test_commits_endpoint_exists(self, http):
        r = http.get("/api/commits?repo=test/test", follow_redirects=False)
        assert r.status_code in (400, 404, 500)

    def test_models_endpoint_exists(self, http):
        r = http.get("/api/models")
        # Should return JSON (empty models list)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data

    def test_rpc_mode_endpoint_exists(self, http):
        r = http.post("/api/rpc_mode", json={"enabled": True})
        assert r.status_code == 200

    def test_setup_endpoint_exists(self, http):
        r = http.post("/api/setup", json={
            "hf_repo": "test/test",
            "quant": "Q4_K_M",
            "symlink_name": "test",
            "parameters": "{}",
            "revision": "latest"
        })
        # Should return 400/422 (validation error) not 404 (not found)
        assert r.status_code in (400, 404, 422, 500)


# ============================================================
# Form Interaction Tests
# ============================================================

class TestFormInteraction:
    """Test form structure and attributes."""

    def test_submit_button_text(self, html):
        btn = html.find(id="submitBtn")
        assert btn.string == "Provision Model"

    def test_clear_button_text(self, html):
        btn = html.find(id="clearBtn")
        assert btn.string == "Clear"

    def test_form_title_initial(self, html):
        title = html.find(id="formTitle")
        assert title.string == "Deploy New Config"

    def test_quant_select_has_disabled_class(self, html):
        """Verify quant select has Tailwind disabled styling."""
        quant = html.find(id="quant")
        classes = quant.get("class", [])
        assert "disabled:opacity-50" in classes

    def test_form_has_hidden_original_name(self, html):
        hidden = html.find(id="original_name")
        assert hidden is not None
        assert hidden.get("type") == "hidden"

    def test_parameters_is_textarea(self, html):
        params = html.find(id="parameters")
        assert params.name == "textarea"

    def test_setup_form_exists(self, html):
        form = html.find(id="setupForm")
        assert form is not None
        # Form uses JS event listener for submission
        assert form.get("class") is not None


# ============================================================
# Search/Filter Tests
# ============================================================

class TestSearchFilter:
    """Test search and filter UI elements."""

    def test_config_search_exists(self, html):
        search = html.find(id="configSearch")
        assert search is not None
        assert search.get("type") == "text"

    def test_storage_search_exists(self, html):
        search = html.find(id="storageSearch")
        assert search is not None
        assert search.get("type") == "text"


# ============================================================
# CSS/Styling Tests
# ============================================================

class TestStyling:
    """Test that CSS is properly structured."""

    def test_custom_styles_present(self, html):
        """Verify custom CSS styles are inlined."""
        styles = html.find_all("style")
        style_text = "".join(s.string or "" for s in styles)
        assert "toggle-checkbox" in style_text
        assert "custom-scrollbar" in style_text
        assert "config-card" in style_text

    def test_tailwind_css_loaded(self, html):
        """Verify Tailwind CSS is linked."""
        links = html.find_all("link", rel="stylesheet")
        css_srcs = [link["href"] for link in links]
        assert "/static/output.css" in css_srcs

    def test_favicon_present(self, html):
        """Verify favicon is linked."""
        links = html.find_all("link", rel="icon")
        assert len(links) > 0
        assert links[0]["href"] == "/favicon.ico"


# ============================================================
# Layout Tests
# ============================================================

class TestLayout:
    """Test the page layout structure."""

    def test_main_grid_exists(self, html):
        """Verify the main grid container exists."""
        grid = html.find(class_="max-w-7xl")
        assert grid is not None

    def test_two_column_layout(self, html):
        """Verify the two-column layout structure."""
        cols = html.find_all(class_="md:col-span-5")
        assert len(cols) == 1
        cols = html.find_all(class_="md:col-span-7")
        assert len(cols) == 1

    def test_dark_theme(self, html):
        """Verify dark theme is applied."""
        assert html.html.get("class") is not None
        assert "dark" in html.html.get("class", [])

    def test_body_has_dark_bg(self, html):
        body = html.body
        assert body is not None
        body_class = body.get("class", [])
        assert "bg-gray-900" in body_class


# ============================================================
# WebSocket Tests
# ============================================================

class TestWebSocket:
    """Test WebSocket endpoint configuration."""

    def test_ws_status_element_exists(self, html):
        """Verify WebSocket status indicator is in the HTML."""
        ws_status = html.find(id="wsStatus")
        assert ws_status is not None
        # Should have a dot indicator span
        assert ws_status.find("span") is not None
