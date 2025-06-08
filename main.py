"""
Aile 3D interactive : dessiner ou générer deux profils (racine & saumon),
produire et visualiser le maillage 3D avec OpenGL.

Fonctionnalités :
- Saisie de profils 2D au clic ou génération par code NACA.
- Rééchantillonnage uniforme des profils selon leur longueur.
- Interpolation linéaire + twist entre racine et saumon.
- Rendu accéléré via pyqtgraph.opengl.
- Cases à cocher pour afficher/masquer le fil de fer (mesh) et la surface (texture).
- Exporter le maillage 3D au format OBJ.
- Thème Clair/Obscur basculable.
- Centre de rotation centré sur l'aile.
- Correction du bug wireframe sans surface.
"""

import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QTabWidget, QCheckBox,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D, QPalette, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ### Générateur NACA 4 chiffres ###
def naca4(code, n=200, chord=1.0):
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Code NACA '{code}' invalide, ex. '2412'.")
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    beta = np.linspace(0, math.pi, n//2)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (
        0.2969 * np.sqrt(x) -
        0.1260 * x -
        0.3516 * x**2 +
        0.2843 * x**3 -
        0.1015 * x**4
    )
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dy = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2*p*x - x**2),
            m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2)
        )
        dy = np.where(
            x < p,
            2*m / p**2 * (p - x),
            2*m / (1-p)**2 * (p - x)
        )
    theta = np.arctan(dy)
    xu = x - yt * np.sin(theta)
    zu = yc + yt * np.cos(theta)
    xl = x[::-1] + yt[::-1] * np.sin(theta[::-1])
    zl = yc[::-1] - yt[::-1] * np.cos(theta[::-1])
    pts = np.vstack([
        np.concatenate([xu, xl]),
        np.concatenate([zu, zl])
    ]).T * chord
    return pts

# ### Rééchantillonnage par longueur ###
def resample_profile(points, num=200):
    pts = np.array(points)
    deltas = np.diff(pts, axis=0)
    dist = np.hypot(deltas[:,0], deltas[:,1])
    cum = np.concatenate(([0], np.cumsum(dist)))
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (num,1))
    t = cum / total
    ti = np.linspace(0,1,num)
    xi = np.interp(ti, t, pts[:,0])
    zi = np.interp(ti, t, pts[:,1])
    return np.vstack([xi, zi]).T

# ### Mesh 3D linéaire entre profils ###
def generate_mesh(root, tip, span, twist_tip, num_chord=200, num_span=40):
    r2 = resample_profile(root, num_chord)
    t2 = resample_profile(tip, num_chord)
    mesh = np.zeros((num_span, num_chord, 3))
    for j, v in enumerate(np.linspace(0,1,num_span)):
        angle = math.radians(v * twist_tip)
        ca, sa = math.cos(angle), math.sin(angle)
        Xr, Zr = r2[:,0], r2[:,1]
        Xt, Zt = t2[:,0], t2[:,1]
        Yr = np.zeros_like(Xr)
        Yt = np.full_like(Xt, span)
        Xr3 = Xr*ca + Zr*sa
        Zr3 = -Xr*sa + Zr*ca
        Xt3 = Xt*ca + Zt*sa
        Zt3 = -Xt*sa + Zt*ca
        mesh[j,:,0] = (1-v)*Xr3 + v*Xt3
        mesh[j,:,1] = (1-v)*Yr  + v*Yt
        mesh[j,:,2] = (1-v)*Zr3 + v*Zt3
    verts = mesh.reshape(-1,3)
    faces = []
    for j in range(num_span-1):
        for i in range(num_chord-1):
            idx = j*num_chord + i
            a, b = idx, idx+1
            c, d = idx+num_chord, idx+num_chord+1
            faces.append([a,b,c])
            faces.append([b,d,c])
    return verts, np.array(faces)

# ### Éditeur de profil 2D ###
class ProfileEditor(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAspectLocked(True)
        self.showGrid(x=True,y=True,alpha=0.3)
        self.data=[]
        self.curve=self.plot([],[],pen=pg.mkPen('b',width=2))
        self.proxy=pg.SignalProxy(self.scene().sigMouseClicked, slot=self.on_click)
        self.setLabel('bottom','x')
        self.setLabel('left','z')

    def on_click(self, evt):
        ev=evt[0]
        if ev.button()!=Qt.LeftButton: return
        x,z=self.plotItem.vb.mapSceneToView(ev.scenePos())
        self.data.append((x,z))
        xs,zs=zip(*self.data)
        self.plotItem.clear()
        self.curve=self.plot(xs,zs,pen=pg.mkPen('b',width=2))
        self.plotItem.vb.autoRange()

    def clear(self):
        self.data.clear()
        self.plotItem.clear()
        self.curve=self.plot([],[],pen=pg.mkPen('b',width=2))
        self.plotItem.vb.autoRange()

    def set_profile(self, pts):
        self.data=[(float(x),float(z)) for x,z in pts]
        xs,zs=zip(*self.data)
        self.plotItem.clear()
        self.curve=self.plot(xs,zs,pen=pg.mkPen('b',width=2))
        self.plotItem.vb.autoRange()

    def get_profile(self):
        return np.array(self.data) if self.data else None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Concepteur d'aile 3D")

        # Configuration de l'interface
        central=QWidget(); self.setCentralWidget(central)
        hbox=QHBoxLayout(central)

        panel=QWidget(); vpanel=QVBoxLayout(panel)

        form=QFormLayout()
        self.root_code=QLineEdit('2412'); self.tip_code=QLineEdit('0012')
        self.root_scale=QLineEdit('1.0'); self.tip_scale=QLineEdit('0.6')
        self.span_input=QLineEdit('5.0')
        form.addRow('Profil racine NACA :',self.root_code)
        form.addRow('Scale racine         :',self.root_scale)
        form.addRow('Profil saumon NACA  :',self.tip_code)
        form.addRow('Scale saumon        :',self.tip_scale)
        form.addRow('Envergure (span)    :',self.span_input)
        btn_naca=QPushButton('Générer profils NACA')
        btn_naca.clicked.connect(self.on_generate_naca)
        form.addWidget(btn_naca); vpanel.addLayout(form)

        tabs=QTabWidget()
        self.root_editor=ProfileEditor(); self.tip_editor=ProfileEditor()
        tabs.addTab(self.root_editor,'Racine'); tabs.addTab(self.tip_editor,'Saumon')
        vpanel.addWidget(tabs)

        btn_clear=QPushButton('Effacer profil actif')
        btn_clear.clicked.connect(lambda: tabs.currentWidget().clear())
        vpanel.addWidget(btn_clear)

        self.chk_mesh=QCheckBox('Afficher fil de fer'); self.chk_mesh.setChecked(True)
        self.chk_texture=QCheckBox('Afficher surface'); self.chk_texture.setChecked(True)
        vpanel.addWidget(self.chk_mesh); vpanel.addWidget(self.chk_texture)

        btn_gen=QPushButton('Générer aile 3D'); btn_export=QPushButton('Exporter OBJ')
        self.btn_theme=QPushButton('Mode clair')  # label inversé : début en sombre
        btn_gen.clicked.connect(self.generate_wing)
        btn_export.clicked.connect(self.export_obj)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.is_dark=True
        vpanel.addWidget(btn_gen)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(self.btn_theme)
        vpanel.addLayout(btn_layout)

        vpanel.addStretch(); hbox.addWidget(panel,stretch=0)

        self.view=gl.GLViewWidget(); self.view.opts['distance']=10
        hbox.addWidget(self.view,stretch=1)

        self.mesh_wire=self.mesh_solid=None
        self.current_verts=self.current_faces=None

        self.chk_texture.stateChanged.connect(self.update_visibility)
        self.chk_mesh.stateChanged.connect(self.update_visibility)

    def toggle_theme(self):
        app=QApplication.instance()
        if self.is_dark:
            # Passer en clair
            app.setStyle('Fusion')
            app.setPalette(QPalette())
            self.btn_theme.setText('Mode sombre')
        else:
            # Repasser en sombre
            dark=QPalette()
            dark.setColor(QPalette.Window, QColor(53,53,53))
            dark.setColor(QPalette.WindowText, Qt.white)
            dark.setColor(QPalette.Base, QColor(25,25,25))
            dark.setColor(QPalette.AlternateBase, QColor(53,53,53))
            dark.setColor(QPalette.ToolTipBase, Qt.white)
            dark.setColor(QPalette.ToolTipText, Qt.white)
            dark.setColor(QPalette.Text, Qt.white)
            dark.setColor(QPalette.Button, QColor(53,53,53))
            dark.setColor(QPalette.ButtonText, Qt.white)
            app.setStyle('Fusion')
            app.setPalette(dark)
            self.btn_theme.setText('Mode clair')
        self.is_dark = not self.is_dark

    def on_generate_naca(self):
        try:
            cr=float(self.root_scale.text()); ct=float(self.tip_scale.text())
            root_pts=naca4(self.root_code.text(), chord=cr)
            tip_pts =naca4(self.tip_code.text(),  chord=ct)
        except Exception as e:
            QMessageBox.warning(self, "Erreur NACA", f"{e}")
            return
        self.root_editor.set_profile(root_pts)
        self.tip_editor.set_profile(tip_pts)

    def generate_wing(self):
        root=self.root_editor.get_profile(); tip=self.tip_editor.get_profile()
        if root is None or tip is None: return
        span=float(self.span_input.text())
        verts,faces=generate_mesh(root,tip,span,twist_tip=-4.0)
        self.current_verts=verts; self.current_faces=faces
        # Centre caméra sur l'aile
        center=verts.mean(axis=0)
        self.view.opts['center']=QVector3D(*center)
        # Nettoyage et création des items
        for it in (self.mesh_wire,self.mesh_solid):
            if it: self.view.removeItem(it)
        meshdata=gl.MeshData(vertexes=verts, faces=faces)
        self.mesh_wire=gl.GLMeshItem(meshdata=meshdata, smooth=False,
                                     drawEdges=True, edgeColor=(1,1,1,1), shader='balloon')
        if self.chk_texture.isChecked():
            self.mesh_solid=gl.GLMeshItem(meshdata=meshdata, smooth=True,
                                          drawEdges=False, shader='shaded', glOptions='opaque')
        else:
            self.mesh_solid=None
        if self.mesh_solid: self.view.addItem(self.mesh_solid)
        if self.mesh_wire:  self.view.addItem(self.mesh_wire)

    def update_visibility(self):
        if self.mesh_solid: self.mesh_solid.setVisible(self.chk_texture.isChecked())
        if self.mesh_wire:  self.mesh_wire.setVisible(self.chk_mesh.isChecked())

    def export_obj(self):
        if self.current_verts is None or self.current_faces is None:
            QMessageBox.warning(self, "Aucun maillage", "Générez d'abord la géométrie.")
            return
        path,_=QFileDialog.getSaveFileName(self,"Enregistrer OBJ","wing.obj","OBJ Files (*.obj)")
        if not path: return
        try:
            with open(path,'w') as f:
                for v in self.current_verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.current_faces:
                    idx=face+1
                    f.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")
            QMessageBox.information(self,"Export réussi",f"{path}")
        except Exception as e:
            QMessageBox.critical(self,"Erreur export",f"{e}")

def main():
    app=QApplication(sys.argv)
    # Initialisation en sombre
    dark=QPalette()
    dark.setColor(QPalette.Window, QColor(53,53,53))
    dark.setColor(QPalette.WindowText, Qt.white)
    dark.setColor(QPalette.Base, QColor(25,25,25))
    dark.setColor(QPalette.AlternateBase, QColor(53,53,53))
    dark.setColor(QPalette.ToolTipBase, Qt.white)
    dark.setColor(QPalette.ToolTipText, Qt.white)
    dark.setColor(QPalette.Text, Qt.white)
    dark.setColor(QPalette.Button, QColor(53,53,53))
    dark.setColor(QPalette.ButtonText, Qt.white)
    app.setStyle('Fusion'); app.setPalette(dark)

    win=MainWindow()
    win.resize(1200,600)
    win.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
