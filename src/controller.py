from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import Qt

from src.view import View
from src.state import State

fontname = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [fontname]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False


class Controller:
    def __init__(self):
        self.state = State()
        self.view = View()
        self.connect_slots()
        self.view.vis_config(self.state.ui_para.data)
        self.view.show()

    def connect_slots(self):
        self.view.signal_close_window.connect(self.on_close_window)
        self.view.signal_connect_nc.connect(self.on_connect_nc)
        self.view.signal_disconnect_nc.connect(self.on_disconnect_nc)
        self.state.signal_connect_nc_status.connect(self.on_connect_nc_status)
        self.view.signal_connect_tem_card.connect(self.on_connect_tem_card)
        self.view.signal_disconnect_tem_card.connect(self.on_disconnect_tem_card)
        self.state.signal_connect_tem_card_status.connect(self.on_connect_tem_card_status)
        self.view.signal_open_port.connect(self.on_open_port)
        self.state.signal_open_port_status.connect(self.on_open_port_status)
        self.view.signal_sample_save_path.connect(self.on_sample_save_path)
        self.view.signal_start_sample.connect(self.on_start_sample)
        self.view.signal_change_orin_sample.connect(self.on_change_orin_sample)
        self.view.signal_stop_sample.connect(self.on_stop_sample)
        self.state.error_assert_nc_client_not_none.connect(self.on_error_nc_client_none)
        self.state.error_assert_temp_card_not_none.connect(self.on_error_temp_card_none)
        self.state.signal_start_sample_success.connect(self.on_set_stop_sample_btn)
        self.state.signal_show_orin_data.connect(self.on_show_orin_data)
        self.state.signal_update_time.connect(self.on_update_time)
        self.state.signal_show_sample_data.connect(self.on_show_rule_data)

        self.view.signal_import_data.connect(self.on_import_data)
        self.view.signal_plot_files.connect(self.on_plot_files)
        self.view.signal_tra_tsp.connect(self.on_tra_tsp)
        self.view.signal_ga_tsp.connect(self.on_ga_tsp)
        self.view.signal_saved_data_path.connect(self.on_saved_data)
        self.view.signal_mlr_fit.connect(self.on_mlr_fit)
        self.view.signal_send_coef.connect(self.on_send_coef)

        self.view.signal_import_model.connect(self.on_import_model)
        self.view.signal_start_train.connect(self.on_start_train)
        self.view.signal_increase_train.connect(self.on_increase_train)
        self.view.signal_pause_train.connect(self.on_pause_train)
        self.view.signal_resume_train.connect(self.on_resume_train)
        self.view.signal_stop_train.connect(self.on_stop_train)
        self.view.signal_save_model.connect(self.on_save_model)

        self.state.signal_get_datasets.connect(self.ui_get_datasets)
        self.state.signal_start_train.connect(self.ui_start_train)
        self.state.signal_train_val_loss.connect(self.ui_train_val_loss)
        self.state.signal_train_finished.connect(self.ui_train_finished)

        self.view.signal_import_tem_model.connect(self.on_import_tem_model)
        self.view.signal_import_rpm.connect(self.on_import_rpm)
        self.view.signal_linear_fit.connect(self.on_linear_fit)
        self.view.signal_quadratic_fit.connect(self.on_quadratic_fit)

    # 关闭应用前从界面更新配置字典给state进行保存
    def on_close_window(self):
        self.state.on_close_window(self.view.update_config(self.state.ui_para.data))

    def on_connect_nc(self, para):
        self.view.setCursor(Qt.CursorShape.WaitCursor)
        self.state.connect_nc(para)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def on_disconnect_nc(self):
        self.state.disconnect_nc()
        self.view.show_message("机床断连", 2000)

    def on_connect_nc_status(self, flag):
        if flag[0]:
            self.view.sample_page.nc_link_widget.set_btn_disconnect()
            self.view.show_message("连接机床成功", 3000)
        else:
            self.view.sample_page.nc_link_widget.btn_connect_nc.setText("重新连接")
            QMessageBox.critical(self.view, "错误", f"{flag[1]}")

    def on_connect_tem_card(self, para):
        self.view.setCursor(Qt.CursorShape.WaitCursor)
        self.state.connect_tem_card(para)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def on_disconnect_tem_card(self):
        self.state.disconnect_tem_card()
        self.view.show_message("采集卡断联", 2000)

    def on_connect_tem_card_status(self, flag):
        if flag[0]:
            self.view.sample_page.tem_card_widget.set_btn_disconnect()
            self.view.show_message("连接采集卡成功", 3000)
        else:
            self.view.sample_page.tem_card_widget.btn_connect.setText("重新连接")
            QMessageBox.critical(self.view, "错误", f"{flag[1]}")

    def on_open_port(self, para):
        self.view.setCursor(Qt.CursorShape.WaitCursor)
        self.state.open_port(para)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def on_close_port(self):
        self.state.close_port()
        self.view.show_message("量表断联", 2000)

    def on_open_port_status(self, flag):
        if flag:
            self.view.sample_page.serial_port_widget.set_btn_close()
            self.view.show_message("打开串口成功", 3000)
        else:
            self.view.sample_page.serial_port_widget.btn_connect.setText("重新打开")
            QMessageBox.critical(self.view, "错误", "打开串口失败！")

    def on_sample_save_path(self, para):
        self.state.set_sample_save_path(para)

    def on_error_nc_client_none(self):
        QMessageBox.critical(self.view, "错误", "选择从机床采集温度但未连接NC卡")

    def on_error_temp_card_none(self):
        QMessageBox.critical(self.view, "错误", "选择从采集卡采集温度但未连接采集卡")

    def on_start_sample(self, para):
        self.view.show_message("开始采集数据", 3000)
        self.state.start_sample(para)

    def on_change_orin_sample(self):
        self.state.show_orin = not self.state.show_orin
        # 因为切换到数据类型需要等到下次更新数据才能切换, 这里直接切换显示数据
        if self.state.show_orin:
            self.view.show_message("切换为显示原始采集数据", 3000)
            self.on_show_orin_data()
        else:
            self.view.show_message("切换为显示规则采集数据", 3000)
            self.on_show_rule_data()

    def on_set_stop_sample_btn(self):
        self.view.sample_page.sample_rule_widget.set_btn_stop_sample()

    def on_stop_sample(self):
        self.view.show_message("采集停止", 3000)
        for btn in self.view.side_menu.btns:
            btn.setEnabled(True)
        self.state.stop_sample()

    def on_show_orin_data(self):
        self.view.sample_page.plot_widget.plot_sample_data(
            self.state.orin_data, self.state.tem_from_nc
        )

    def on_show_rule_data(self):
        self.view.sample_page.plot_widget.plot_sample_data(
            self.state.rule_data, self.state.tem_from_nc
        )

    def on_update_time(self):
        h, m, s = self.convert_seconds(self.state.orin_count)
        self.view.show_message(
            f"原始数据: {self.state.orin_count}组; 规则数据: {self.state.rule_count}组; 采集时长: {h}时{m}分{s}秒",
            1000,
        )

    @staticmethod
    def convert_seconds(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds_remaining = seconds % 60
        return hours, minutes, seconds_remaining

    def on_import_data(self, para):
        self.view.setCursor(Qt.CursorShape.BusyCursor)
        self.state.get_data(para)
        # 只要导入了数据, 后面的按钮才能启用
        self.view.tsp_page.tsp_config.btn_tra_tsp.setEnabled(True)
        self.view.tsp_page.tsp_config.btn_ga_tsp.setEnabled(True)
        self.view.tsp_page.tsp_res.btn_mlr_fit.setEnabled(True)
        self.view.tsp_page.tsp_res.btn_save_data.setEnabled(True)
        self.view.tsp_page.tsp_res.btn_send_para.setEnabled(True)
        self.view.model_page.model_train.btn_start_train.setEnabled(True)
        # 用表格显示数据(效果不好)
        # self.view.tsp_page.import_data.show_data(self.state.data.data_cat)
        self.view.show_message(f"从{para[0]}导入数据", 3000)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def on_plot_files(self):
        self.view.tsp_page.plot_files(self.state.data.data_arrays)
        self.view.show_message("已图示所有导入的数据", 3000)

    def on_tra_tsp(self, para):
        # para[0] - 测点数量
        # para[1] - 聚类方法
        # para[2] - 关联度方法
        if self.state.data is None:
            QMessageBox.critical(self.view, "错误", "请先导入数据")
            return

        self.state.get_cluster_res(method_name=para[1], tsp_num=para[0])
        self.state.get_tra_tsp_res(method_name=para[2])

        self.view.show_message(f"传统测点筛选完成:{self.state.tsp_res}", 2000)
        self.view.tsp_page.tsp_res.edit_tsp.setText(",".join(map(str, self.state.tsp_res)))

    def on_ga_tsp(self, para):
        # para[0] - 测点数量
        # para[1] - 聚类方法
        # para[2] - 种群数量
        # para[3] - 迭代次数
        import time

        if self.state.data is None:
            QMessageBox.critical(self.view, "错误", "请先导入数据")
            return

        self.view.setCursor(Qt.CursorShape.BusyCursor)  # 耗时较长, 鼠标样式设为等待
        start_time = time.time()
        self.state.get_cluster_res(method_name=para[1], tsp_num=para[0])
        self.state.get_ga_tsp_res(pop_size=para[2], iters=para[3])
        end_time = time.time()
        cos_time = end_time - start_time
        self.view.show_message(
            f"耗时:{cos_time:.2f}s, 迭代测点筛选完成 {self.state.tsp_res}", 5000
        )
        self.view.tsp_page.tsp_res.edit_tsp.setText(",".join(map(str, self.state.tsp_res)))
        self.view.setCursor(Qt.CursorShape.ArrowCursor)  # 恢复鼠标样式

    def on_saved_data(self, para):
        file_path = para[0]
        inter_num = para[1]
        tsp_res_text = self.view.tsp_page.tsp_res.edit_tsp.text()
        success, message = self.state.save_chosen_data(file_path, tsp_res_text, inter_num)
        if not success:
            QMessageBox.critical(self.view, "保存错误", message)
        else:
            self.view.show_message("已保存筛选后的数据", 2000)

    def on_mlr_fit(self):
        tsp_res_text = self.view.tsp_page.tsp_res.edit_tsp.text()
        success, message = self.state.mlr_fit(tsp_res_text)
        if not success:
            QMessageBox.critical(self.view, "错误", message[0])
        else:
            self.view.tsp_page.plot_widget.plot_pred(
                data_list=self.state.data.data_arrays,
                pred_list=message[0],
                tsp_list=message[1],
            )
            self.view.tsp_page.tsp_res.show_mlr_fit_res(
                intercept=message[2], coef=message[3], rmse=message[4]
            )
            self.view.show_message("已显示筛选温度点曲线和多元线性回归拟合结果", 2000)

    def on_send_coef(self, para):
        print(f"导入拟合参数:{para}")
        self.state.send_coef(para)

    def on_import_model(self, para):

        self.view.setCursor(Qt.CursorShape.BusyCursor)
        self.view.show_message("正在导入Pytorch模型...")
        QApplication.processEvents()  # 强制刷新界面, 不然鼠标样式不改变
        self.state.load_err_model(para)
        self.view.model_page.model_train.btn_increase_train.setEnabled(True)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)
        self.view.model_page.model_train.btn_start_train.setText("重新训练")
        self.view.show_message("模型导入完成", 2000)

    def on_start_train(self, para):
        tsp_res_text = self.view.tsp_page.tsp_res.edit_tsp.text()
        self.state.start_train(para, tsp_res_text)

    def on_increase_train(self, para):
        tsp_res_text = self.view.tsp_page.tsp_res.edit_tsp.text()
        self.state.increase_train(para, tsp_res_text)

    def on_pause_train(self):
        self.state.thread_train.pause()
        self.view.show_message("训练暂停...")

    def on_resume_train(self):
        self.state.thread_train.resume()
        self.view.show_message("继续训练...")

    def on_stop_train(self):
        self.state.thread_train.stop()
        self.view.show_message("训练结束", 2000)

    def on_save_model(self, file_path):
        self.state.save_model(file_path)

    def ui_get_datasets(self):
        self.view.show_message("正在处理数据集")
        QApplication.processEvents()  # 强制刷新界面

    def ui_start_train(self):
        self.view.show_message("开始训练模型...")

    def ui_train_val_loss(self, para):
        self.view.model_page.update_loss_canvas(para, self.state.data.Xdata)
        self.view.show_message(
            f"训练损失: {para[3]:.3f}, 验证损失: {para[4]:.3f}, 测试损失: {para[5]:.3f}"
        )

    def ui_train_finished(self, para):
        self.view.model_page.update_finished_canvas(para)
        self.view.model_page.model_train.btn_start_train.setEnabled(True)
        self.view.model_page.model_train.btn_increase_train.setEnabled(True)
        self.view.show_message(
            f"训练完成, 最优测试损失: {para[2]:.3f}, 出现在第{para[0]+1}折训练的第{para[1]+1}步"
        )
        self.state.best_model = para[3]  # 保存最佳模型

    def on_import_tem_model(self, para):
        self.state.import_tem_model(para)

    def on_import_rpm(self, file_path):
        self.state.import_rpm_file(file_path)
        self.view.compen_page.import_rpm.show_avg_rpm(self.state.sampled_rpm)

    def on_linear_fit(self):
        self.state.cal_para(1)
        self.view.compen_page.get_para.show_fit_para(1, self.state.coef_linear)

    def on_quadratic_fit(self):
        self.state.cal_para(2)
        self.view.compen_page.get_para.show_fit_para(2, self.state.coef_quadratic)
