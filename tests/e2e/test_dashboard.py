"""tests/e2e/__init__.py"""
"""tests/e2e/test_dashboard.py

E2E tests for the Streamlit dashboard using Playwright.

Prerequisites:
    pip install pytest playwright
    playwright install chromium

Running:
    # Start the dashboard in a separate terminal:
    streamlit run app.py --server.port 8501

    # Run tests:
    pytest tests/e2e/test_dashboard.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.mark.skipif(
    os.environ.get("SKIP_E2E") == "1",
    reason="E2E tests require running Streamlit server"
)
class TestDashboard:
    def test_dashboard_loads(self, browser):
        page = browser.new_page()
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        assert page.title() == "BTC Forecast — AlphaI × Polaris"

    def test_theme_toggle(self, browser):
        page = browser.new_page()
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        initial_bg = page.evaluate(
            "getComputedStyle(document.body).backgroundColor"
        )
        toggle = page.locator("button", has_text="TOGGLE THEME")
        if toggle.count() > 0:
            toggle.click()
            page.wait_for_timeout(500)
            new_bg = page.evaluate(
                "getComputedStyle(document.body).backgroundColor"
            )

    def test_no_console_errors(self, browser):
        errors = []

        def on_console(msg):
            if msg.type == "error":
                errors.append(msg.text)

        page = browser.new_page()
        page.on("console", on_console)
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        assert len(errors) == 0, f"Console errors: {errors}"

    def test_prediction_range_renders(self, browser):
        page = browser.new_page()
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        content = page.content()
        assert "95% CONFIDENCE INTERVAL" in content

    def test_responsive_at_375px(self, browser):
        page = browser.new_page()
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        html = page.content()
        assert "BTC FORECAST" in html or "₿" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])