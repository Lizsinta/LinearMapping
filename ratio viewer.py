import os

import numpy as np
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QSize, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QColor, QPen, QCursor
from PyQt6.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QSizePolicy, \
    QTextBrowser, QGraphicsRectItem, QSlider, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, \
    QMessageBox, QRadioButton, QSpacerItem, QMenu
from PyQt6.QtCore import Qt

import pyqtgraph as pg
from qtgraph import line, scatter, bar


class Ui_LrMap(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1280, 720))
        MainWindow.setSizeIncrement(QtCore.QSize(0, 0))
        MainWindow.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1280, 720))
        self.centralwidget.setSizeIncrement(QtCore.QSize(0, 0))
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setObjectName("centralwidget")
        self.mainLayout = QGridLayout(self.centralwidget)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName("mainLayout")
        self.mainLayout.setContentsMargins(1, 1, 1, 1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


class ImgGraphicsView(QGraphicsView):
    sigMouseMovePoint = pyqtSignal(QPoint)
    sigCal = pyqtSignal(np.ndarray)
    sigRangeCancel = pyqtSignal(int)
    sigSave = pyqtSignal(int)

    def __init__(self, parent=None, scrollbar_x=True, scrollbar_y=True, range_select=True):
        super(ImgGraphicsView, self).__init__(parent)
        self.dragStartPos = QPointF(0, 0)
        self.selectStartPos = QPointF(0, 0)
        self.selectCurrentPos = QPointF(0, 0)

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn if scrollbar_x else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn if scrollbar_y else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # self.setDragMode(QGraphicsView.DragMode.)

        self.range_select = range_select
        if self.range_select:
            self.selectPlot = QGraphicsRectItem()
            pen = QPen()
            color = QColor()
            color.setNamedColor('red')
            pen.setColor(color)
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.SolidLine)
            self.selectPlot.setPen(pen)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragStartPos = event.pos()
            menu = QMenu()
            action = menu.addAction('save')
            action.triggered.connect(self.save_action)
            menu.exec(QCursor.pos())
        elif event.button() == Qt.MouseButton.RightButton:
            if self.range_select:# and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                self.selectStartPos = self.mapToScene(event.pos())
                self.selectCurrentPos = self.mapToScene(event.pos())
                self.selectPlot.setRect(self.selectStartPos.x(), self.selectStartPos.y(), 0, 0)
                self.scene().addItem(self.selectPlot)
            else:
                # delta = self.selectStartPos - self.selectCurrentPos
                # if delta.y() == 0 and delta.x() == 0:
                #     return
                pass
        return

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pass
        elif event.button() == Qt.MouseButton.RightButton:
            pass
        return

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pass
        if event.button() == Qt.MouseButton.RightButton:
            if self.range_select:
                #if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                delta = self.selectStartPos - self.selectCurrentPos
                if delta.y() == 0 and delta.x() == 0:
                    self.sigRangeCancel.emit(0)
                    self.selectStartPos = QPointF(0, 0)
                    self.selectCurrentPos = QPointF(0, 0)
                    self.scene().removeItem(self.selectPlot)
                self.sigCal.emit(np.array([self.selectStartPos.y(), self.selectCurrentPos.y(),
                                            self.selectStartPos.x(), self.selectCurrentPos.x()]).astype(int))
        return

    def mouseMoveEvent(self, event):
        mpt = event.pos()
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            return
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = mpt - self.dragStartPos
            self.dragStartPos = mpt

            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        elif event.buttons() == Qt.MouseButton.RightButton:
            if self.range_select:
                #if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                    delta = self.mapToScene(mpt) - self.selectStartPos
                    self.selectCurrentPos = self.mapToScene(mpt)
                    self.selectPlot.setRect(self.selectStartPos.x(), self.selectStartPos.y(), delta.x(), delta.y())

        self.sigMouseMovePoint.emit(mpt)
        return

    def save_action(self):
        self.sigSave.emit(0)


class MainWindow(QMainWindow, Ui_LrMap):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("LrMapping")
        self.mainLayout.setRowStretch(0, 1)
        self.mainLayout.setRowStretch(1, 1)
        self.mainLayout.setRowStretch(2, 1)
        self.mainLayout.setRowStretch(3, 1)
        self.mainLayout.setRowStretch(4, 1)
        self.mainLayout.setRowStretch(5, 1)
        self.mainLayout.setColumnStretch(0, 2)
        self.mainLayout.setColumnStretch(1, 1)
        self.mainLayout.setColumnStretch(2, 1)

        self.ut0 = np.array([])
        self.ut = np.array([])
        self.energy = np.array([])
        self.e0 = np.array([])
        self.amp = np.array([])

        # histogram
        self.barView = pg.PlotWidget(background=[255, 255, 255, 255])
        self.barView.setObjectName('barView')
        self.mainLayout.addWidget(self.barView, 1, 1, 2, 2)
        self.barView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.barItemT = pg.BarGraphItem(x=np.array([]), height=np.array([]), width=0.3)
        self.barItemF = pg.BarGraphItem(x=np.array([]), height=np.array([]), width=0.3)

        # 2d plot
        self.g2d = pg.PlotWidget(background=[255, 255, 255, 255])
        self.mainLayout.addWidget(self.g2d, 1, 0, 2, 1)
        self.fitLine = pg.PlotDataItem()
        self.scatter = pg.ScatterPlotItem()

        # img
        self.imgView = ImgGraphicsView(scrollbar_x=False, scrollbar_y=False, range_select=False)
        self.imgView.setObjectName('imgView')
        self.mainLayout.addWidget(self.imgView, 3, 0, 2, 2)
        self.pixitem_img = QGraphicsPixmapItem()
        self.scene_img = QGraphicsScene()
        self.x_img, self.y_img = 0, 0
        self.mouse_xy_img = QPoint(0, 0)
        self.imgView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.imgView.sigSave.connect(self.imgSaveEvent)

        self.img = np.array([])
        self.pic_img = QPixmap()
        self.currect_img = 0

        # browser
        self.textBrowser = QTextBrowser()
        self.mainLayout.addWidget(self.textBrowser, 3, 2, 2, 1)

        # slider
        # self.sliderLayout = QHBoxLayout()
        # self.sliderLayout.setObjectName("sliderLayout")
        # self.mainLayout.addLayout(self.sliderLayout, 5, 1, 1, 2)
        # self.sliderLabel = QLabel(parent=self.centralwidget)
        # self.sliderLabel.setObjectName("sliderLabel")
        # self.sliderLabel.setText(f"R filter [{0:>5}]  ")
        # self.sliderLayout.addWidget(self.sliderLabel)
        # self.slider = QSlider(Qt.Orientation.Horizontal)
        # self.slider.setRange(0, 10000)
        # self.slider.setTickInterval(1000)
        # self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.slider.valueChanged.connect(self.slicerChange)
        # self.sliderLayout.addWidget(self.slider)
        # self.slicePosi = 0

        # img select
        # self.imgButtonLayout = QHBoxLayout()
        # self.sliderLayout.setObjectName("imgButtonLayout")
        # self.mainLayout.addLayout(self.imgButtonLayout, 5, 0, 1, 1)
        # self.imgButtonLayout.addWidget(QLabel('image control: '))
        # img_name = ['Sum ', 'Co ', 'Fe ', 'Mn ']
        # self.imgButton = np.array([QRadioButton(img_name[i], parent=self.centralwidget) for i in range(4)], dtype=QRadioButton)
        # self.imgButton[0].setChecked(True)
        # for i in range(4):
        #     self.imgButton[i].setObjectName(img_name[i])
        #     self.imgButtonLayout.addWidget(self.imgButton[i])
        #     self.imgButton[i].setEnabled(False)
        #     self.imgButton[i].clicked.connect(self.img_select)
        # spacer = QSpacerItem(40, 20, hPolicy=QSizePolicy.Policy.Expanding, vPolicy=QSizePolicy.Policy.Minimum)
        # self.imgButtonLayout.addItem(spacer)

        # r map select
        # self.imgButtonLayout.addWidget(QLabel('r map control: '))
        # r_name = ['rmax ', 'points ', 'weight ']
        # self.rButton = np.array([QRadioButton(r_name[i], parent=self.centralwidget) for i in range(3)],
        #                           dtype=QRadioButton)
        # self.rButton[0].setChecked(True)
        # for i in range(3):
        #     self.rButton[i].setObjectName(r_name[i])
        #     self.imgButtonLayout.addWidget(self.rButton[i])
        #     self.rButton[i].setEnabled(False)
        #     self.rButton[i].clicked.connect(self.rmap_select)
        # self.current_rmap = 0
        # self.imgButtonLayout.addWidget(QLabel('   ||    '))

        # file reader
        self.fileLayout = QGridLayout()
        self.fileLayout.setObjectName("fileLayout")
        self.mainLayout.addLayout(self.fileLayout, 0, 0, 1, 1)
        self.fileLabel = QLabel('file ', parent=self.centralwidget)
        self.fileLayout.addWidget(self.fileLabel, 0, 0, 1, 1)

        self.fileButton = QPushButton('...', parent=self.centralwidget)
        self.fileLayout.setObjectName("file")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.fileButton.setSizePolicy(sizePolicy)
        self.fileLayout.addWidget(self.fileButton, 0, 1, 1, 1)
        self.fileButton.clicked.connect(self.read_name)

        self.f_name = r'D:\data\20250321\Co\Co-CoO-Co3O4_bg\co_ut.npy'
        self.fileLine = QLineEdit(self.f_name, parent=self.centralwidget)
        self.fileLayout.addWidget(self.fileLine, 0, 2, 1, 1)
        self.fileLine.setReadOnly(True)

        self.runButton = QPushButton('Run', parent=self.centralwidget)
        self.runButton.setSizePolicy(sizePolicy)
        self.fileLayout.addWidget(self.runButton, 0, 3, 1, 1)
        self.runButton.clicked.connect(self.load_dat)

    def read_name(self):
        file_dialog = QFileDialog(self, "select dat files...", os.path.dirname(self.fileLine), "DAT Files (*_ut.npy)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
        else:
            return
        self.f_name = file_paths[0].split('_ut.npy')[0]
        self.fileLine.setText(file_paths[0].split('_ut.npy')[0])

    def load_dat(self):
        self.f_name = self.f_name.split('_ut.npy')[0]
        self.ut0 = np.load(self.f_name + '_ut.npy').reshape(30, 64, -1)
        ut0 = self.ut0.reshape(-1, self.ut0.shape[-1])
        self.energy = np.load(os.path.dirname(self.f_name) + r'\energy0.npy')
        self.e0 = np.load(self.f_name + '_e0.npy').reshape(30, 64)
        self.amp = np.load(self.f_name + '_amp.npy').reshape(30, 64)
        print(self.amp.min(), self.amp.max())
        eref = self.energy[self.e0].mean()
        en, ep = 20, 200
        flatten = False
        print(eref)

        # std = np.array([0])
        # if eref < 7200:
        #     species = np.array(['Fe', 'FeO', 'Fe2O3a', 'Fe2O3y', 'Fe3O4'])
        #     ref_folder = r'D:/data/Fe-ref_align'
        # elif eref < 7800:
        #     species = np.array(['Co', 'CoO', 'Co3O4'])
        #     ref_folder = r'D:/data/Co-ref'
        # else:
        #     species = np.array(['Cu', 'Cu2O', 'CuO'])
        #     ref_folder = r'D:\data\20240524'  # r'D:/data/Cu-ref_align'# os.getcwd() + '/data'#

        print(self.ut0.shape)
        efit = np.where((self.energy > (eref - en)) & (self.energy < (eref + ep)))[0]
        pre_edge_i = np.where(self.energy < (eref - 20))[0]
        post_edge_i = np.where(self.energy > (eref + 50))[0]

        poly_pre = np.polyfit(self.energy[pre_edge_i], ut0[:, pre_edge_i].T, 1)
        poly_post = np.polyfit(self.energy[post_edge_i], ut0[:, post_edge_i].T, 1)
        half_height = ((eref * (poly_post[0] + poly_pre[0]) + poly_post[1] + poly_pre[1]) / 2).reshape(
            poly_post[0].size, 1)
        e0 = np.argmin(np.abs(ut0[:, efit] - half_height), axis=1) + int(efit[0])
        pre_edge_i = np.where(self.energy < (eref - 50))[0]
        post_edge_i = np.where(self.energy > (eref + 150))[0]
        poly_pre = np.polyfit(self.energy[pre_edge_i], ut0[:, pre_edge_i].T, 1)
        poly_post = np.polyfit(self.energy[post_edge_i], ut0[:, post_edge_i].T, 1)
        self.ut = ut0 - (self.energy.reshape(-1, 1) * poly_pre[0] + poly_pre[1]).T
        # ut0 = ut0 - np.min(ut0, axis=1)[:, None]
        # ut0 = ut0 - (energy.reshape(-1, 1) * poly_pre[0] + poly_pre[1]).T
        if flatten:
            et_map = np.repeat(self.energy[None, :], e0.size, 0)
            for i in range(e0.size):
                et_map[i, :e0[i]] = self.energy[e0[i]]
            self.ut += (self.energy[e0, None] ** 2 - et_map ** 2) * poly_post[0][:, None]
            self.ut += (self.energy[e0, None] - et_map) * (poly_post[1] - poly_pre[0])[:, None]
        self.ut = self.ut.reshape(*self.ut0.shape)

        # fit graph
        self.upper = True
        _amp = self.amp.reshape(-1)
        absorb = _amp / (poly_pre[0] * self.energy[e0] + poly_pre[1])
        degree = np.arange(2)
        threshold = 0.6
        fit = (_amp + 0.2) * 0.9  # post 02
        dist = np.abs(absorb - fit)
        ci = dist
        if self.upper:
            ci = (ci - threshold) / (ci.max() - threshold)
        else:
            ci = (ci - ci.min()) / (threshold - ci.min())
        cm = plt.get_cmap('rainbow')
        color = (cm(ci.clip(0, 1))[:, :3]*255).astype(int)
        color[dist < threshold if self.upper else dist > threshold] = np.array([0, 0, 0])
        # color[_amp < 1.2] = np.array([0, 0, 0, 1])
        brushes = [pg.mkBrush(*rgb) for rgb in color]
        self.fitLine.setData(x=_amp, y=fit, color=(255, 0, 0))
        self.g2d.addItem(self.fitLine)
        self.scatter.addPoints([{'x': _amp[i], 'y': absorb[i], 'brush':brushes[i]} for i in range(_amp.shape[0])])
        self.g2d.addItem(self.scatter)
        # ax1.plot(_amp, fit, c='r', label='fitting line')
        # ax1.set_title('pre line: A$x^{2}$+Bx+C' if degree[1] == 2 else 'post line: Ax+B')
        # ax1.set_xlabel('Δμt', fontsize=15)
        # ax1.set_ylabel('A$E_{0}$+B', fontsize=15)
        # ax1.legend()

        # bar graph
        rdf = np.unique(np.round(dist, 2), return_counts=True)
        rdf_select = (rdf[0] >= threshold) if self.upper else (rdf[0] <= threshold)
        ci = (rdf[0][rdf_select] - rdf[0][rdf_select].min()) / (rdf[0][rdf_select].max() - rdf[0][rdf_select].min())
        color = (cm(ci.clip(0, 1))[:, :3] * 255).astype(int)
        brushes = [pg.mkBrush(*rgb) for rgb in color]
        self.barItemT.setData(x=rdf[0][rdf_select], height=rdf[1][rdf_select], width=0.005, brush=brushes)
        self.barItemF.setData(x=rdf[0][~rdf_select], hieght=rdf[1][~rdf_select], width=0.005, brush='k')
        self.barView.addItem(self.barItemT)
        self.barView.addItem(self.barItemF)

        # init_3dplot(self.g3d, grid=False, background=[0, 0, 0], alpha=1.0, view=50000, title='example',
        #             size=[int(self.darray.max()), int(self.darray.max()), int(self.darray.max())])
        # self.scatter = gl.GLScatterPlotItem(pos=self.darray, color=(1, 1, 1, 1), size=5)
        # self.g3d.addItem(self.scatter)

        # img
        # self.img0 = ((self.amp - self.amp.min()) / (self.amp.max() - self.amp.min()) * 255
        #             ).astype(np.uint8)[:, :, None].repeat(3, 2)
        # self.img = self.img0.copy()
        # self.pic_img = QPixmap.fromImage(
        #     QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
        #            QImage.Format.Format_RGB888))
        # self.pixitem_img.setPixmap(self.pic_img)
        # self.scene_img.addItem(self.pixitem_img)
        # self.imgView.setScene(self.scene_img)
        # self.imgView.fitInView(self.scene_img.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # self.imgView.sigMouseMovePoint.connect(self.mouselocation_img)
        # scale = max((self.pic_img.width()) / (self.imgView.width() - 13),
        #             (self.pic_img.height()) / (self.imgView.height() - 13))
        #self.imgView.scale(scale, scale)

        # self.slider.setRange(0, int(self.darray_r.max()))
        # self.slider.setTickInterval(int(self.darray_r.max()) // 10)

        # self.runButton.setEnabled(False)
        # for i in range(4):
        #     self.imgButton[i].setEnabled(True)
        # for i in range(3):
        #     self.rButton[i].setEnabled(True)

    def analysis(self, absorb, image0, degree=2, upper=False):
        _amp = image0.reshape(-1)
        degree = np.arange(2) + degree - 1
        threshold = 0.6
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        fit = (_amp + 0.2) * 0.9  # post 02
        dist = np.abs(absorb - fit)
        ci = dist
        if upper:
            ci = (ci - threshold) / (ci.max() - threshold)
        else:
            ci = (ci - ci.min()) / (threshold - ci.min())
        cm = plt.get_cmap('rainbow')
        color = cm(ci.clip(0, 1))
        color[dist < threshold if upper else dist > threshold] = np.array([0, 0, 0, 1])
        # color[_amp < 1.2] = np.array([0, 0, 0, 1])
        ax1.scatter(_amp, absorb, c=color)
        ax1.plot(_amp, fit, c='r', label='fitting line')
        ax1.set_title('pre line: A$x^{2}$+Bx+C' if degree[1] == 2 else 'post line: Ax+B')
        ax1.set_xlabel('Δμt', fontsize=15)
        ax1.set_ylabel('A$E_{0}$+B', fontsize=15)
        ax1.legend()
        ax2 = fig.add_subplot(132)
        rdf = np.unique(np.round(dist, 2), return_counts=True)
        rdf_select = (rdf[0] >= threshold) if upper else (rdf[0] <= threshold)
        ci = (rdf[0][rdf_select] - rdf[0][rdf_select].min()) / (rdf[0][rdf_select].max() - rdf[0][rdf_select].min())
        ax2.bar(rdf[0][rdf_select], rdf[1][rdf_select], width=0.005, color=cm(ci))
        ax2.bar(rdf[0][~rdf_select], rdf[1][~rdf_select], width=0.005, color='k')
        ax2.set_title('|difference of line to point(Δμt, A$E_{0}$+B)|')
        ax2.set_xlabel('difference', fontsize=15)
        ax2.set_ylabel('count', fontsize=15)
        # ax2.set_yscale('log')
        rdf = np.unique(np.round(dist, 2), return_inverse=True)
        select_i = np.where(
            rdf[1] >= np.where(rdf[0] >= threshold)[0][0] if upper else rdf[1] < np.where(rdf[0] >= threshold)[0][0])[0]
        ax3 = fig.add_subplot(133)
        ax3.imshow(image0, 'gray')
        color = np.zeros(dist.size)
        color[select_i] = rdf[1][select_i]
        color = color / color.max()
        color = color.reshape(image0.shape)
        if not upper:
            color = 1 - color
        alpha = np.zeros(_amp.size)
        alpha[select_i] += 1
        # alpha[_amp<1.2] = 0
        alpha = alpha.reshape(image0.shape)
        ax3.imshow(color, cmap='rainbow', alpha=alpha)

    def updatePixmap(self, select: np.ndarray):
        self.select = select
        if self.select[1] < self.select[0]:
            self.select[1], self.select[0] = self.select[0], self.select[1]
        if self.select[3] < self.select[2]:
            self.select[3], self.select[2] = self.select[2], self.select[3]
        self.select[1] = min(self.select[1], self.rmap.shape[0])
        self.select[3] = min(self.select[3], self.rmap.shape[1])

        self.img = self.img0.copy()
        if (self.select[1] + self.select[3] - self.select[0] - self.select[2]) > 0:
            target_index = np.array([], dtype=int)
            for i in np.arange(self.select[0], self.select[1]):
                for j in np.arange(self.select[2], self.select[3]):
                    temp = self.point_list[int(i*self.rmap.shape[0]+j)]
                    if not len(temp) == 0:
                        temp = temp[np.where(self.darray_r[temp] >= self.slicePosi)[0]]
                        target_index = np.append(target_index, temp)

            self.mean_ratio = self.darray[target_index].mean(0)
            self.mean_ratio /= self.mean_ratio.sum()

            # color = np.zeros((self.darray.shape[0], 4)) + 1
            # color.T[1][target_index] = 0
            # color.T[2][target_index] = 0
            # self.scatter.setData(color=color)

            target_index = np.unravel_index(target_index, self.co.shape)
            self.img.transpose(2, 0, 1)[0][target_index] = 255
            self.img.transpose(2, 0, 1)[1][target_index] = 0
            self.img.transpose(2, 0, 1)[2][target_index] = 0
        self.pic_img = QPixmap.fromImage(QImage(self.img.data, self.img.shape[1], self.img.shape[0],
                                                self.img.shape[1] * 3, QImage.Format.Format_RGB888))
        self.pixitem_img.setPixmap(self.pic_img)

    def img_select(self):
        name = self.sender().objectName()
        name_list = {'Sum ': 0, 'Co ': 1, 'Fe ': 2, 'Mn ': 3}
        target_img = name_list[name]
        if not self.currect_img == target_img:
            for i in range(4):
                if not i == target_img:
                    self.imgButton[i].setChecked(False)
            if target_img == 0:
                img = self.imgarray
            elif target_img == 1:
                img = self.co
            elif target_img == 2:
                img = self.fe
            else:
                img = self.mn
            self.img0 = ((img - img.min()) / (img.max() - img.min()) * 255
                         ).astype(np.uint8)[:, :, None].repeat(3, 2)
        self.currect_img = target_img
        self.updatePixmap(self.select)

    def rmap_select(self):
        name = self.sender().objectName()
        name_list = {'rmax ': 0, 'points ': 1, 'weight ': 2}
        target_rmap = name_list[name]
        if not self.current_rmap == target_rmap:
            for i in range(3):
                if not i == target_rmap:
                    self.imgButton[i].setChecked(False)
            if target_rmap == 0:
                self.rmap = self.r_distri
            elif target_rmap == 1:
                self.rmap = self.point
            else:
                self.rmap = self.r_distri * self.point
            r_map = ((self.rmap - self.rmap.min()) / (self.rmap.max() - self.rmap.min()) * 255
                         ).astype(np.uint8)[:, :, None].repeat(3, 2)
            self.pic_r = QPixmap.fromImage(QImage(r_map.data, r_map.shape[1], r_map.shape[0], r_map.shape[1] * 3,
                                                  QImage.Format.Format_RGB888))
            self.pixitem_r.setPixmap(self.pic_r)
        self.current_rmap = target_rmap

    def slicerChange(self, x):
        self.slicePosi = x
        self.sliderLabel.setText(f"R filter [{self.slicePosi:>5}]  ")
        if self.select.size > 0:
            self.updatePixmap(self.select)
            self.locationUpdate()

    def mouselocation_img(self, mpt):
        sc_xy = self.imgView.mapToScene(mpt.x(), mpt.y())
        self.mouse_xy_img = self.pixitem_img.mapFromScene(sc_xy).toPoint()
        if self.mouse_xy_img.y() >= self.pic_img.height():
            self.mouse_xy_img.setY(int(self.pic_img.height() - 1))
        elif self.mouse_xy_img.y() <= 0:
            self.mouse_xy_img.setY(0)
        else:
            pass
        if self.mouse_xy_img.x() >= self.pic_img.width():
            self.mouse_xy_img.setX(int(self.pic_img.width() - 1))
        elif self.mouse_xy_img.x() <= 0:
            self.mouse_xy_img.setX(0)
        else:
            pass
        self.x_img, self.y_img = self.mouse_xy_img.x(), self.mouse_xy_img.y()
        self.locationUpdate()

    def mouselocation_r(self, mpt):
        sc_xy = self.rView.mapToScene(mpt.x(), mpt.y())
        self.mouse_xy_r = self.pixitem_r.mapFromScene(sc_xy).toPoint()
        if self.mouse_xy_r.y() >= self.pic_r.height():
            self.mouse_xy_r.setY(int(self.pic_r.height() - 1))
        elif self.mouse_xy_r.y() <= 0:
            self.mouse_xy_r.setY(0)
        else:
            pass
        if self.mouse_xy_r.x() >= self.pic_r.width():
            self.mouse_xy_r.setX(int(self.pic_r.width() - 1))
        elif self.mouse_xy_r.x() <= 0:
            self.mouse_xy_r.setX(0)
        else:
            pass
        self.x_r, self.y_r = self.mouse_xy_r.x(), self.mouse_xy_r.y()
        self.locationUpdate()

    def locationUpdate(self):
        img_index = int(self.y_img*self.co.shape[0] + self.x_img)
        img_ratio = self.darray[img_index].astype(int)
        text = (f'img\nx:{self.x_img}  y:{self.y_img}\n'
                f'Pointed data:\n'
                f'Co:{img_ratio[0]}\n'
                f'Fe:{img_ratio[1]}\n'
                f'Mn:{img_ratio[2]}\n\n'
                f'sphere map\nx:{self.x_r}  y:{self.y_r}\n'
                f'Pointed ratio:\n'
                f'Co:{self.ratio_co[self.y_r][self.x_r]:.2f}\n'
                f'Fe:{self.ratio_fe[self.y_r][self.x_r]:.2f}\n'
                f'Mn:{self.ratio_mn[self.y_r]:.2f}\n\n'
                f'Area Average:\n'
                f'Co:{self.mean_ratio[0]:.2f}\n'
                f'Fe:{self.mean_ratio[1]:.2f}\n'
                f'Mn:{self.mean_ratio[2]:.2f}\n')
        self.textBrowser.setText(text)

    def imgSaveEvent(self, a0):
        if self.sender().objectName() == 'imgView':
            address = os.path.dirname(self.f_list[0]) + r'/image.png'
            name = QFileDialog.getSaveFileName(self, 'select save path...', address)[0]
            if name == '':
                return
            plt.imsave(name, self.img)
        else:
            address = os.path.dirname(self.f_list[0]) + r'/r distribution.png'
            name = QFileDialog.getSaveFileName(self, 'select save path...', address)[0]
            if name == '':
                return
            img = ((self.rmap - self.rmap.min()) / (
                    self.rmap.max() - self.rmap.min()) * 255
                   ).astype(np.uint8)[:, :, None].repeat(3, 2)
            if (self.select[1] + self.select[3] - self.select[0] - self.select[2]) > 0:
                red = np.array([255, 0, 0])
                img[self.select[0], self.select[2]:self.select[3]+1] = red
                img[self.select[1], self.select[2]:self.select[3]+1] = red
                img[self.select[0]:self.select[1]+1, self.select[2]] = red
                img[self.select[0]:self.select[1]+1, self.select[3]] = red
            plt.imsave(name, img)

    def warning_window(self, massage):
        QMessageBox.critical(self, 'Error', massage)



# rdf = np.unique(co//100*100, return_counts=True)
# plt.bar(rdf[0]-25, rdf[1], width=50)
# rdf = np.unique(fe//100*100, return_counts=True)
# plt.bar(rdf[0], rdf[1], width=50)
# rdf = np.unique(mn//100*100, return_counts=True)
# plt.bar(rdf[0]+25, rdf[1], width=50)
# plt.subplot(311)
# plt.scatter(co, fe)
# plt.xlabel('Co')
# plt.ylabel('Fe')
# plt.subplot(312)
# plt.scatter(fe, mn)
# plt.xlabel('Fe')
# plt.ylabel('Mn')
# plt.subplot(313)
# plt.scatter(mn, co)
# plt.xlabel('Mn')
# plt.ylabel('Co')
if __name__ == '__main__':
    from sys import argv, exit
    #file = r'D:\data\H017_Co_1x1_0x0-0-0.dat'
    # file = r'D:\data\H054_Co_1x1_0x0_zoom-0.dat'
    # f_co = r'D:\data\H054_Co_1x1_0x0_zoom-0.dat'
    # f_fe = r'D:\data\H054_Fe_1x1_0x0_zoom-0.dat'
    # f_mn = r'D:\data\H054_Mn_1x1_0x0_zoom-0.dat'
    # f_co = r'D:\data\H017_Co_1x1_0x0-0-0.dat'
    # f_fe = r'D:\data\H017_Fe_1x1_0x0-0-0.dat'
    # f_mn = r'D:\data\H017_Mn_1x1_0x0-0-0.dat'
    # with open(f_co, 'r') as f:
    #     while True:
    #         temp = f.readline()
    #         if not temp.find('X-NUM') == -1:
    #             width = int(temp.split()[2]) - 1
    #             height = int(f.readline().split()[2]) - 1
    #             break
    #         if not temp:
    #             break
    # co = np.loadtxt(f_co, usecols=2, dtype=float, delimiter=',').reshape(height, width)
    # co[co > 5000] = 0
    # fe = np.loadtxt(f_fe, usecols=2, dtype=float, delimiter=',').reshape(height, width)
    # fe[fe > 10000] = 0
    # mn = np.loadtxt(f_mn, usecols=2, dtype=float, delimiter=',').reshape(height, width)
    # darray = np.vstack((co.flatten(), fe.flatten(), mn.flatten())).T
    # darray_r = np.sqrt((darray**2).sum(1))
    # darray_ele = np.zeros_like(darray_r)
    # darray_ele[darray_r>0] = np.arccos(darray[darray_r > 0, 2]/darray_r[darray_r > 0])/np.pi*180
    # darray_azi = np.zeros_like(darray_r)
    # darray_azi[(darray_r > 0)&(darray[:, 0]>0)] = np.arctan(darray[(darray_r > 0)&(darray[:, 0]>0), 1] / darray[(darray_r > 0)&(darray[:, 0]>0), 0])/np.pi*180
    # darray_map = np.unique(np.round(np.vstack((darray_azi, darray_ele)).T, 0).astype(int), axis=0, return_inverse=True)
    # # darray_map[1]: index sequence in darray size
    # r_distri = np.zeros((91, 91))
    # point = np.zeros((91, 91))
    # point_list = [np.array([]), ] * point.size
    # print(np.unique(darray_map[1]).size)
    # for i in np.unique(darray_map[1]):
    #     locate = darray_map[0][i]
    #     point_list[int(locate[0]*point.shape[0] + locate[1])] = np.where(darray_map[1] == i)[0]
    #     r_distri[locate[0]][locate[1]] = darray_r[np.where(darray_map[1] == i)[0]].max()
    #     point[locate[0]][locate[1]] = np.where(darray_map[1] == i)[0].size
    # point[0] = 0
    # point[-1] = 0
    # point[:, 0] = 0
    # point[:, -1] = 0
    # weight_r = r_distri#*point
    # target_max = np.unravel_index(np.argmax(weight_r), weight_r.shape)
    # # target_index = np.array([], dtype=int)
    # # for i in np.arange(target_max[0]-5, target_max[0]+5):
    # #     for j in np.arange(target_max[1] - 5, target_max[1] + 5):
    # #         target_index = np.append(target_index, point_list[int(i*weight_r.shape[0]+j)])
    # target_index = point_list[np.argmax(weight_r)]
    # target_index = np.unravel_index(target_index, co.shape)
    #
    # #plt.subplot(211)
    # plt.imshow(r_distri, 'gray')
    # plt.ylabel('Φ', fontsize=15)
    # plt.xlabel('θ', fontsize=15)
    # # plt.subplot(212)
    # # alpha = np.zeros_like(co)
    # # alpha[target_index] = 1
    # # plt.imshow(alpha, 'Reds', alpha=alpha)
    # # plt.imshow(co+fe+mn, 'gray', alpha=1-alpha)
    # # plt.ylabel('azi')
    # # plt.xlabel('ele')
    # azi = np.arange(90)/180*np.pi
    # ele = np.arange(90)/180*np.pi
    # #cartesian = np.array([np.sin(ele) * np.cos(azi), np.sin(ele) * np.sin(azi), np.cos(ele)])
    # rmax_i = np.unravel_index(np.argmax(weight_r), weight_r.shape)
    # ele, azi = rmax_i[0]/180*np.pi, rmax_i[1]/180*np.pi
    # print(rmax_i, np.sin(ele) * np.cos(azi), np.sin(ele) * np.sin(azi), np.cos(ele))
    # plt.show()



    app = QApplication(argv)
    main = MainWindow()
    main.show()
    exit(app.exec())