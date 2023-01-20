from Orange.data import Table
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
    name = "Lime"
    description = "Lime for model explainability"
    icon = "icons/mywidget.svg"
    priority = 100 
    keywords = ["widget", "data"]
    want_main_area = True
    resizing_enabled = True
    
    label = Setting("")


    class Inputs:
        data = Input("Data", Table)
        #features = Input("Features", AttributeList)

    class Outputs:
        data = Output("Data", Table, default=True)
    
    class Warning(OWWidget.Warning):
        warning = Msg("My warning!")

    target_combo_choose = settings.Setting('empty')

    def __init__(self):
        super().__init__()
        self.data = None
        #self.target_combo = gui.comboBox(None, self.controlArea, 'target_combo_choose')
        self.show_html = QtWebEngineWidgets.QWebEngineView()
        button = gui.button(self.controlArea, self, "Explain results", callback = self.run_lime)
        self.mainArea.layout().addWidget(self.show_html)

       

    @Inputs.data
    def set_data(self, data):
        if data:
            self.data = data
        else:
            self.data = None

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
        str_feat = []
        for i in range(len(domain.attributes)):
            feat = str(domain.attributes[i])
            str_feat.append(feat)
        str_feat.append(str(target))
        df = pd.DataFrame(self.data)
        df.columns = str_feat
        input_cols = df[df.columns.difference(['Class'])]
        output_col = df['Class']
        class_unique = df['Class'].unique()
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(input_cols, output_col, random_state=1, test_size = 0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predict_fn = lambda x: model.predict_proba(x)
        explainer = lime_tabular.LimeTabularExplainer(df[input_cols.columns].astype(int).values, 
                                                            mode='classification', 
                                                            class_names= class_unique,  
                                                            training_labels=df['Class'], 
                                                            feature_names=input_cols.columns)
        exp=explainer.explain_instance(df.loc[11,input_cols.columns].astype(int).values, predict_fn, num_features=15)
        exp_html = exp.save_to_file('./example/widgets/lime.html')
        #self.show_html.setHtml(exp_html)
        self.show_html.load(QtCore.QUrl(('./example/widgets/lime.html')))
        self.show_html.show()

       

    def commit(self):
        self.Outputs.data.send(self.data)
    
    def send_report(self):
        # self.report_plot() includes visualizations in the report
        self.report_caption(self.label)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(MyWidget).run()
