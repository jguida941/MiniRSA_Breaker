import pytest

PyQt6 = pytest.importorskip("PyQt6")

from rsa import RSAMainWindow  # noqa: E402


@pytest.mark.gui
def test_main_window_initializes(qtbot) -> None:
    window = RSAMainWindow()
    qtbot.addWidget(window)
    window.show()

    assert window.n_label.text() == "33"
    assert window.phi_label.text() == "20"
    assert window.e_spin.value() == 3

    window.calculate_keys()
    assert "Keys calculated successfully" in window.statusBar().currentMessage()
