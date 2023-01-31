from Orange.data import Table
from Orange.base import Learner
from Orange.base import Model
from Orange.widgets import gui, settings
from Orange.widgets.settings import Setting
from Orange.widgets.gui import QComboBox
from Orange.widgets.widget import OWWidget, Input, Output, Msg,  AttributeList
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import shap
import numpy as np
from IPython.display import HTML,display
from PyQt5.QtCore import QRunnable, Qt, QThreadPool
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
from sklearn.utils import parallel_backend, register_parallel_backend
from joblib._parallel_backends import ThreadingBackend
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend
from dask.distributed import Client
from joblibspark import register_spark
from lime import lime_tabular
import os
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import imgkit
from PyQt5.QtWidgets import *
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.io import export_png

class MyWidget(OWWidget):
    name = "Lime explainable AI"
    description = "Lime for model explainability"
    icon = "icons/mywidget.svg"
    priority = 100 
    keywords = ["widget", "data"]
    want_main_area = True
    resizing_enabled = True
    
    label = Setting("")


    class Inputs:
        data = Input("Data", Table)
        model = Input("Model", Model)
        test_instance = Input("Instance", Table)

    class Outputs:
        data = Output("Data", Table, default=True)
    
    class Warning(OWWidget.Warning):
        warning = Msg("My warning!")

    want_main_area = True
    target_combo_choose = settings.Setting('empty')

    def __init__(self):
        super().__init__()
        self.data = None
        self.test_instance = None
        self.model = None

        self.instanceInfo = gui.widgetBox(self.controlArea, "Selected instance for explanation")
        self.infoInstanceLoaded = gui.widgetLabel(self.instanceInfo, '')
        button = gui.button(self.controlArea, self, "Show graph", callback = self.run_lime)
        button_save = gui.button(self.controlArea, self, "Save HTML graph", callback = self.show_html)

        self.webEngineView = QWebEngineView()
        self.webEngineView.show()
        
 

    @Inputs.data
    def set_data(self, data):
        if data:
            self.data = data
        else:
            self.data = None

    @Inputs.model
    def set_model(self, model):
        if model:
            self.model = model 
        else:
            self.model = None
    
    @Inputs.test_instance
    def set_instance(self, instance):
        if instance:
            self.test_instance= instance
            self.instance_info()
        else:
            self.test_instance = None

    def instance_info(self):
        domain = self.data.domain
        target = domain.class_var
        metas = self.data.metas
        str_feat = []
        for i in range(len(domain.attributes)):
            feat = str(domain.attributes[i])
            str_feat.append(feat)
        str_feat.append(str(target))
        df = pd.DataFrame(self.test_instance)
        df.columns = str_feat
        self.infoInstanceLoaded.setText(str(df.iloc[0]))
    
    def show_html(self):
        self.loadPage()
        self.mainArea.layout().addWidget(self.webEngineView)

    def run_lime(self):
        
        domain = self.data.domain
        target = domain.class_var
        metas = self.data.metas
        str_feat = []
        for i in range(len(domain.attributes)):
            feat = str(domain.attributes[i])
            str_feat.append(feat)
        str_feat.append(str(target))
        df = pd.DataFrame(self.data)
        df.columns = str_feat
        input_cols = df[df.columns.difference([str(target)])]
        output_col = df[str(target)]
        class_unique = df[str(target)].unique()
        
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(input_cols, output_col, random_state=1, test_size = 0.2)
        explainer = lime_tabular.LimeTabularExplainer(df[input_cols.columns].astype(int).values, 
                                                            mode='classification', 
                                                            class_names= class_unique,  
                                                            training_labels=df[str(target)], 
                                                            feature_names=input_cols.columns)
        #data frame za test instance
        df_instance = pd.DataFrame(self.test_instance)
        df_instance.columns = str_feat
        df_instance = df[df.columns.difference([str(target)])]
        df_instance_row = df_instance.iloc[0]

        instance_to_explain = df_instance_row
        exp=explainer.explain_instance(instance_to_explain, self.model.predict_proba, num_features=8)

        self.exp_html = exp.save_to_file('./example/widgets/new.html')
        #self.exp_html = exp.as_html()
        print(self.exp_html)
        self.exp_graph = exp.as_pyplot_figure()

        self.canvas = FigureCanvas(self.exp_graph)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.mainArea.layout().addWidget(self.toolbar)
        self.mainArea.layout().addWidget(self.canvas)

    def loadPage(self):
        file_path = QtCore.QUrl.fromLocalFile('./example/widgets/new.html')
        self.webEngineView.load(file_path)
        self.webEngineView.show()
    
       

    def commit(self):
        self.Outputs.data.send(self.data)
    
    def send_report(self):
        # self.report_plot() includes visualizations in the report
        self.report_caption(self.label)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(MyWidget).run()
