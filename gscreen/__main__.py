import sys

from PySide6.QtWidgets import QApplication

from gscreen.gscreen import GaussScreen

app = QApplication([])

g_screen = GaussScreen()
g_screen.resize(1200, 800)
g_screen.show()

sys.exit(app.exec())
