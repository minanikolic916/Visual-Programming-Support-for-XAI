#orange imports
from Orange.data import Table
from Orange.base import Learner
from Orange.base import Model
from Orange.widgets import gui, settings
from Orange.widgets.settings import Setting
from Orange.widgets.gui import QComboBox
from Orange.widgets.widget import OWWidget, Input, Output, Msg,  AttributeList
#scikit imports
import sklearn.datasets
import sklearn.metrics
#utility imports
import pandas as pd
import numpy as np
#pyqt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
#other imports
from html2image import Html2Image
from lime import lime_tabular
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import dataframe_image as dfi
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
    method_idx=settings.Setting(1)
    target_combo_choose = settings.Setting('empty')

    def __init__(self):
        super().__init__()
        self.data = None
        self.test_instance = None
        self.model = None

        self.layout = QVBoxLayout()
        self.controlArea.setLayout(self.layout)
        self.instanceInfo = gui.widgetBox(self.controlArea, "Selected instance for explanation")
        self.infoInstanceLoaded = gui.widgetLabel(self.instanceInfo, '')

        self.boxOptions = box = gui.radioButtonsInBox(self.controlArea,self,"method_idx", box="Graph type selection")
        self.plot_radio_button = gui.appendRadioButton(box,"Plot", addToLayout=True)
        self.html_radio_button = gui.appendRadioButton(box,"Html", addToLayout=True)

        lime_processing_button = gui.button(self.controlArea, self, "Explain prediction", callback = self.graph_choice)


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
            self.run_lime()
        else:
            self.test_instance = None

    def graph_choice(self):
        exp = self.run_lime()
        if(self.plot_radio_button.isChecked()):
            widg = self.show_plot(exp)
        else:
            widg = self.show_html()
        for i in reversed(range(self.mainArea.layout().count())):
            self.mainArea.layout().itemAt(i).widget().deleteLater()
        self.mainArea.layout().addWidget(widg)
                
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
        df = df.transpose()
        dfi.export(df, 'dataframe.png')
        self.infoInstanceLoaded.setPixmap(QPixmap('dataframe.png'))
        #self.infoInstanceLoaded.setText(str(df.iloc[0]))
        #self.infoInstanceLoaded.addWidget(self.image_label_info)
          

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
        df_instance = df_instance[df.columns.difference([str(target)])]
        print(df_instance)
        df_instance_row = df_instance.iloc[0]
        
        instance_to_explain = df_instance_row
        exp=explainer.explain_instance(instance_to_explain, self.model.predict_proba, num_features=8)
        self.exp_file= exp.save_to_file('./example/widgets/results.html')
        return exp
      
    def show_plot(self,exp):
        self.exp_graph = exp.as_pyplot_figure()
        self.canvas = FigureCanvas(self.exp_graph)
        #self.mainArea.layout().addWidget(self.canvas)
        return self.canvas


    def show_html(self):
        with open('./example/widgets/results.html', 'r', encoding='utf-8') as f:
            html = f.read()
            hti = Html2Image()
            hti.screenshot(html_str=html, css_str = ['body {background:white;}'], save_as= 'new.jpeg')
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setPixmap(QPixmap('new.jpeg').scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            #self.mainArea.layout().addWidget(self.image_label)
        return self.image_label

    def commit(self):
        self.Outputs.data.send(self.data)
    
    def send_report(self):
        # self.report_plot() includes visualizations in the report
        self.report_caption(self.label)
            
    


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(MyWidget).run()