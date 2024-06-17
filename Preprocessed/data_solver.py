import hashlib
import os
import datetime


class DataFilter:
    def __init__(self):
        self._input_dir = ''
        self._time = 0
        self.train_time = [['2021-07-29 00:40:00.000000', '2021-07-29 01:20:00.000000']]
        for i in range(len(self.train_time)):
            self.train_time[i][0] = datetime.datetime.strptime(self.train_time[i][0].split('.')[0],
                                                               "%Y-%m-%d %H:%M:%S").timestamp()
            self.train_time[i][1] = datetime.datetime.strptime(self.train_time[i][1].split('.')[0],
                                                               "%Y-%m-%d %H:%M:%S").timestamp()
    def set_input_dir(self, input_dir):
        if not os.path.exists(input_dir):
            print('{} doesn\'t exist'.format(input_dir))
        all_file = []
        for file_name in os.listdir(input_dir):
            all_file.append(file_name)
        for i in ['train.txt', 'entity2id.txt', 'relation2id.txt']:
            if i not in all_file:
                print('file not found:{}/{}'.format(input_dir, i))
                print('fail to set input_dir!')
                return
        self._input_dir = input_dir

    def set_time(self, t):
        self._time = int(t)

    def _tuple_label(self, tuple_data):
        fields = tuple_data.split()
        for t in self.train_time:
            if t[0] <= int(fields[3]) <= t[1]:
                return 'test'
        return None

    def _output_set(self, data_name, data_set, output_dir):
        data_map = {}
        with open(os.path.join(self._input_dir, data_name)) as f:
            for i in f.readlines():
                data_map[i.split()[1]] = (i.split()[0], i.split()[1])
        with open(os.path.join(output_dir, data_name), 'w+') as out:
            for i in data_set:
                out.write('{}\t{}\n'.format(data_map[i][0], data_map[i][1]))
            out.flush()

    def output(self, output_dir):
        if self._input_dir == '':
            print('please set input_dir!')
            return
        if not os.path.exists(output_dir):
            print('output_dir:{}:doesn\'t exist and will create it!'.format(output_dir))
            os.mkdir(output_dir)
        entity_set = set()
        relation_set = set()
        with open(os.path.join(output_dir, 'train.txt'), 'w+') as out_train:
            with open(os.path.join(output_dir, 'valid.txt'), 'w+') as out_valid:
                with open(os.path.join(output_dir, 'test.txt'), 'w+') as out_test:
                    with open(os.path.join(self._input_dir, 'train.txt')) as f:
                        for line in f.readlines():
                            label = self._tuple_label(line)
                            if label is None:
                                continue
                            elif label == 'train':
                                out_train.write(line)
                            elif label == 'valid':
                                out_valid.write(line)
                            elif label == 'test':
                                out_test.write(line)
                            entity_set.add(line.split()[0])
                            relation_set.add(line.split()[1])
                            entity_set.add(line.split()[2])
                        out_train.flush()
                        out_valid.flush()
                        out_test.flush()
        os.system('cp {} {}'.format(os.path.join(self._input_dir, 'entity2id.txt'), output_dir))
        os.system('cp {} {}'.format(os.path.join(self._input_dir, 'relation2id.txt'), output_dir))
        # self._output_set('entity2id.txt',entity_set,output_dir)
        # self._output_set('relation2id.txt',relation_set,output_dir)


class DataMerger:
    def __init__(self):
        self._data_list = []

    def set_data_dir_to_null(self):
        self._data_list = []

    def add_data_dir(self, d):
        if not self._data_is_valid(d):
            print('Add dir failed!!!')
            return
        self._data_list.append(str(d))

    def _data_is_valid(self, data_dir):
        if not os.path.exists(data_dir):
            print('{} doesn\'t exist'.format(data_dir))
            return
        file_name_list = []
        need_file_list = ['entity2id.txt', 'relation2id.txt', 'train.txt', 'test.txt', 'valid.txt']
        for file_name in os.listdir(data_dir):
            file_name_list.append(file_name)
        for need_file in need_file_list:
            if need_file not in file_name_list:
                print('dir:{}:not contains file:{}'.format(data_dir, need_file))
                return False
        return True

    def _get_merge_result(self, file_name):
        file_entityID_map = {}
        entity_id_map = {}
        new_id = 0
        for data_dir in self._data_list:
            file_entityID_map[data_dir] = {}
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f.readlines():
                    e = str(line.split()[0])
                    index = int(line.split()[1])
                    if entity_id_map.get(e) is None:
                        entity_id_map[e] = new_id
                        new_id += 1
                    file_entityID_map[data_dir][index] = e
        return file_entityID_map, entity_id_map

    def _output_map(self, result_map, output_dir, file_name):
        lines = list(map(lambda t: '{}\t{}\n'.format(t[0], str(t[1])), list(result_map.items())))
        lines.sort(key=lambda line: int(line.split()[1]))
        with open(os.path.join(output_dir, file_name), 'w+') as out:
            out.writelines(lines)
            out.flush()

    def _merge_file(self, file_name, output_dir, entity2id_result, relation2id_result):
        with open(os.path.join(output_dir, file_name), 'w+') as out:
            for data_dir in self._data_list:
                file_entityID_map, entity_id_map = entity2id_result
                file_relationID_map, relation_id_map = relation2id_result
                gE = lambda x: entity_id_map[file_entityID_map[data_dir][int(x)]]
                gR = lambda x: relation_id_map[file_relationID_map[data_dir][int(x)]]
                with open(os.path.join(data_dir, file_name)) as f:
                    for line in f.readlines():
                        fields = line.split()
                        out.write(
                            '{}\t{}\t{}\t{}\t{}\n'.format(gE(fields[0]), gR(fields[1]), gE(fields[2]), fields[3],
                                                          fields[4]))
            out.flush()
        self._sort_file_lines(os.path.join(output_dir, file_name), lambda x: int(x.split()[3]))

    def _sort_file_lines(self, file_path, key_function):
        with open(file_path) as f:
            lines = f.readlines()
        lines.sort(key=key_function)
        with open(file_path, 'w+') as f:
            f.writelines(lines)

    def output_data(self, output_dir):

        if not os.path.exists(output_dir):
            print('output_dir:{}:doesn\'t exist and will create it!'.format(output_dir))
            os.mkdir(output_dir)
        entity_result = self._get_merge_result('entity2id.txt')
        relation_result = self._get_merge_result('relation2id.txt')
        for file_name in ['train.txt', 'valid.txt', 'test.txt']:
            self._merge_file(file_name, output_dir, entity_result, relation_result)
        self._output_map(entity_result[1], output_dir, 'entity2id.txt')
        self._output_map(relation_result[1], output_dir, 'relation2id.txt')

        with open(os.path.join(output_dir, 'stat.txt'), 'w+') as out:
            out.write('{}\t{}\t{}'.format(str(len(entity_result[1])), str(len(relation_result[1])), '0'))
            out.flush()


class DataSolver:

    def __init__(self):
        self.data_dir = []
        self.dataFilter = DataFilter()
        self.dataMerger = DataMerger()
        self.temp = ''

    def get_time_line_list(self):
        result = []
        with open('run_table.csv') as f:
            f.readline()
            for line in f.readlines():
                if line == '\"\n': continue
                try:
                    t_str = line.split(',')[2].strip(' ').strip('"').strip('\'')
                    t = int(datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S').timestamp())
                except:
                    continue
        return result

    def set_temp_dir(self, temp_dir):
        self.temp = temp_dir

    def output(self, output_dir):
        if self.temp == '':
            return
        for i in os.listdir(self.temp):
            print('temp dir should be empty!!!')
            return

        flag = False
        for t, run_line in self.get_time_line_list():
            if os.path.exists(os.path.join(output_dir, str(t))):
                continue
            else:
                flag = True
            self.dataMerger.set_data_dir_to_null()
            for i in os.listdir(self.temp):
                os.system('rm -r {}'.format(os.path.join(self.temp, i)))

            for d in self.data_dir:
                self.dataFilter.set_input_dir(d)
                self.dataFilter.set_time(int(t))
                self.dataFilter.output(os.path.join(self.temp, str(hashlib.md5(d.encode('utf-8')).hexdigest())))
                self.dataMerger.add_data_dir(os.path.join(self.temp, str(hashlib.md5(d.encode('utf-8')).hexdigest())))
            self.dataMerger.output_data(os.path.join(output_dir, str(t)))

            for i in os.listdir(self.temp):
                os.system('rm -r {}'.format(os.path.join(self.temp, i)))
            print('{}:complete->{}'.format(run_line, str(t)))

            if flag:
                break

    def add_data(self, data_dir):
        self.data_dir.append(data_dir)


dataSolver = DataSolver()
dataSolver.add_data('log')
dataSolver.add_data('metric')
dataSolver.add_data('trace')

dataSolver.set_temp_dir('temp')

dataSolver.output('output')
