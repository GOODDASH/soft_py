from numpy import vstack


class MyData:
    def __init__(self, para):
        self.tsp_T = None
        self.data_arrays = []
        self.temp_begin_index = para[3]
        self.error_index = para[5]
        flag_transpose = False if para[1] == 0 else True

        match para[2]:
            case 0:
                self.sep = "\t"
            case 1:
                self.sep = ","
            case 2:
                self.sep = " "
            case _:
                self.sep = None

        for file_path in para[0]:
            from src.core.file2numpy import read_datafile_to_numpy

            array = read_datafile_to_numpy(
                file_path,
                trans=flag_transpose,
                sep=self.sep,
                t_begin=para[3],
                t_end=para[4],
                e_idx=para[5],
            )
            array = array - array[0, :]  # 减去第一行
            self.data_arrays.append(array)
        self.data_cat = vstack(self.data_arrays)
        # 提取温度、热误差数据
        self.Tdata = self.data_cat[:, :-1]
        self.Xdata = self.data_cat[:, -1]

    def write_file(self, file_path: str, chosen_idx: list[int]):
        from src.core.file2numpy import write_datafile_from_numpy

        base_name, extension = file_path.rsplit(".", 1)
        chosen_idx = [idx + self.temp_begin_index for idx in chosen_idx]
        chosen_idx.append(self.error_index)
        for i, data_array in enumerate(self.data_arrays):
            new_file_path = f"{base_name}_{i+1}.{extension}"
            write_datafile_from_numpy(data_array[:, chosen_idx], new_file_path)
