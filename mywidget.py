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
from IPython.display import HTML
from PyQt5.QtCore import QRunnable, Qt, QThreadPool
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
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
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QLineEdit
# class Worker(QRunnable):

#     def __init__(self):
#         super().__init__()

#     def run(self, data):
#         print('Worker thread started')
#         domain = data.domain
#         target = domain.class_var
#         str_feat = []
#         for i in range(len(domain.attributes)):
#             feat = str(domain.attributes[i])
#             str_feat.append(feat)
#         str_feat.append(str(target))
#         df = pd.DataFrame(data)
#         df.columns = str_feat
#         input_cols = df[df.columns.difference(['Class'])]
#         output_col = df['Class']
#         X_train, X_test, y_train, y_test = \
#         sklearn.model_selection.train_test_split(input_cols, output_col, random_state=1, test_size = 0.2)
#         automl = autosklearn.classification.AutoSklearnClassifier(
#             time_left_for_this_task=120,
#             per_run_time_limit=30,
#             memory_limit = 30000,
#             n_jobs=-1
#         )
#         print('Creating model done. Starting training')
#         automl.fit(X_train, y_train)
#         y_hat = automl.predict(X_test)
#         accuracy = sklearn.metrics.accuracy_score(y_test,y_hat)
#         #print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))    
#         print(accuracy)

    
    
        

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

    target_combo_choose = settings.Setting('empty')

    def __init__(self):
        super().__init__()
        self.data = None
        #self.test_instance = None
        self.model = None
        self.show_html = QtWebEngineWidgets.QWebEngineView()
        self.test_index = QLineEdit()
        self.test_index.setAlignment(Qt.AlignRight)
        button = gui.button(self.controlArea, self, "Show graph", callback = self.run_lime)
        #self.mainArea.layout().addWidget(self.show_html)
        #self.controlArea.layout().addWidget(self.test_index)
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
        else:
            self.test_instance = None


    def run_lime(self):
        
        # print('Model ready to begin training...')
        # threadCount = QThreadPool.globalInstance().maxThreadCount()
        # pool = QThreadPool.globalInstance()
        # for i in range(threadCount):
        #     #runnable = Worker(self, self.data)
        #     pool.start(Worker().run(self.data))
        
        # self.thread = QThread()
        # self.worker = Worker()
        # self.worker.moveToThread(self.thread)
        # self.thread.started.connect(self.worker.run(self.data))
        # self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        # self.thread.start()

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
        predict_values = []
        predict_value_zero = []
        predict_value_one  = []
        predict_labels=[]
        # for i in range(len(self.data.metas)):
        #     predict_labels.append(metas[i][0])
        #     predict_value_zero.append(metas[i][1])
        #     predict_value_one.append(metas[i][2])
        #     predict_values.append([predict_value_zero[i], predict_value_one[i]])
        def get_predict_proba_at_index(i):
            predict_values[i] = np.array(predict_values[i]) 
            ret = predict_values[i].reshape(-1,1)
            return ret
        def get_labels():
            return predict_labels
        
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(input_cols, output_col, random_state=1, test_size = 0.2)
        #model = RandomForestClassifier()
        #model.fit(X_train, y_train)
        explainer = lime_tabular.LimeTabularExplainer(df[input_cols.columns].astype(int).values, 
                                                            mode='classification', 
                                                            class_names= class_unique,  
                                                            training_labels=df[str(target)], 
                                                            feature_names=input_cols.columns)
        #predict_proba = lambda x : get_predict_proba_at_index(0)
        index = 2
        print(self.model)
        #predict_fn = lambda x: get_predict_proba_at_index(index)
        #predict_fn = lambda x: model.predict_proba
        #exp=explainer.explain_instance(df.loc[index,input_cols.columns], predict_fn, num_features=5, labels= [0,1])
        instance_to_explain = df.loc[index,input_cols.columns],
        exp=explainer.explain_instance(instance_to_explain, self.model.predict_proba, num_features=8)
        exp_html = exp.save_to_file('./example/widgets/new.html')
        print(self.test_instance)
        #exp_graph = exp.as_pyplot_figure()
        #print(type(exp_graph.show()))
        #self.show_html.setHtml(exp_html)
        #HTML(data=exp_html)
        #self.show_html.load(QtCore.QUrl().fromLocalFile('/home/Downloads/orange3-example-addon-master/orangecontrib/example/widgets/new.html'))
        #self.show_html.load(QtCore.QUrl().fromLocalFile('./example/widgets/new.html'))
        #self.show_html.show()
        self.image_view = QPixmap("./example/widgets/new.jpg")
        self.image_label = QLabel()
        self.image_label.setPixmap(self.image_view)
        self.canvas = FigureCanvas(exp.as_pyplot_figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.mainArea.layout().addWidget(self.toolbar)
        self.mainArea.layout().addWidget(self.canvas)
        

       

    def commit(self):
        self.Outputs.data.send(self.data)
    
    def send_report(self):
        # self.report_plot() includes visualizations in the report
        self.report_caption(self.label)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(MyWidget).run()
