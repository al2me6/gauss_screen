from PySide6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtWidgets import QSpinBox, QStyledItemDelegate, QStyleOptionViewItem, QWidget


class SpinBoxDelegate(QStyledItemDelegate):
    """https://doc.qt.io/qtforpython-6/examples/example_widgets_itemviews_spinboxdelegate.html"""

    def __init__(self, /, parent: QObject | None = None):  # noqa: F821
        super().__init__(parent)

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSpinBox:
        editor = QSpinBox(parent)
        editor.setMinimum(1)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex | QPersistentModelIndex):
        assert isinstance(editor, QSpinBox)
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if isinstance(value, int):
            editor.setValue(value)
        else:
            editor.setValue(editor.valueFromText(value))

    def setModelData(
        self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex | QPersistentModelIndex
    ):
        assert isinstance(editor, QSpinBox)
        editor.interpretText()
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self,
        editor: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ):
        rect: QRect = option.rect  # type: ignore
        decoration: QSize = option.decorationSize  # type: ignore
        rect.adjust(max(decoration.width(), 0), 0, 0, 0)
        editor.setGeometry(rect)
