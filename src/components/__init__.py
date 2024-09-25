# 一些复用的组件
from .custom_btn import CustomBtn
from .combo_options import ComboOptions
from .single_plot_widget import SinglePlotWidget
from .multi_plot_widget import MultiPlotWidget
from .pop_up import PopUp

# 侧边栏
from .side_menu import SideMenu

# 采集页面组成
from .sample_nc_link_setting import NCLinkSetting
from .sample_tem_card_setting import TemCardSetting
from .sample_serial_port_setting import SerialPortSetting
from .sample_rule_setting import SampleRuleSetting

# 分析、选点页面组成
from .tsp_import_setting import TspImportSetting
from .tsp_config import TspConfig
from .tsp_res import TspRes

# 模型导入、训练页面组成
from .model_choose import ModelChoose
from .graph_edit import GraphEdit
from .model_train import ModelTrain
from .model_plot import ModelPlot

# 补偿页面组成
from .compen_tem_model import CompenTemModel
from .compen_import_rpm import CompenImportRpm
from .compen_get_para import CompenGetPara

__all__ = [
    "CustomBtn",
    "ComboOptions",
    "SinglePlotWidget",
    "MultiPlotWidget",
    "SideMenu",
    "NCLinkSetting",
    "TemCardSetting",
    "SerialPortSetting",
    "SampleRuleSetting",
    "TspImportSetting",
    "TspConfig",
    "TspRes",
    "ModelChoose",
    "GraphEdit",
    "ModelTrain",
    "ModelPlot",
    "CompenTemModel",
    "CompenImportRpm",
    "CompenGetPara",
    "PopUp",
]
