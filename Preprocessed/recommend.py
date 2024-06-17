import pandas as pd
import os
from collections import Counter

failure_type = ['access_permission_denied', 'file_moving', 'memory_anomalies' ,'normal_memory_freed_label']
entity_df = pd.read_csv('GAIA/access_permission_denied/dbservice1[access_permission_denied]1626998400/entity2id.txt',sep = '\t',header=None)
entity_dict = dict(zip(entity_df[1],entity_df[0]))

for ff in failure_type:
    g = os.walk('../data/GAIA_TKG/{}'.format(ff))
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name == 'test.txt':
                result_list = []
                case_time = int(path[-10:])
                result_list.append(case_time)
                kg_df = pd.read_csv(os.path.join(path, file_name), sep='\t', header=None)
                kg_df = kg_df[(kg_df[3] >= case_time) & (kg_df[3] <= (case_time + 600))]

                # exception log
                log_df = kg_df[(kg_df[1] == 0) | (kg_df[1] == 1) | (kg_df[1] == 2)]
                log_list = set(log_df[2].values.tolist())
                log_result = [entity_dict[x] if x in entity_dict else x for x in log_list]
                if ('979a5d07' in log_result) or ('9dece557' in log_result):
                    result_list.append('Access denied')
                    print(result_list)
                    continue
                if '117254af' in log_result:
                    result_list.append('File not found')
                    print(result_list)
                    continue

                # Abnormal indicators
                metric_df = kg_df[(kg_df[1] == 3) | (kg_df[1] == 4) | (kg_df[1] == 6) | (kg_df[1] == 7)]
                metric_list = set(metric_df[2].values.tolist())
                rep = [(entity_dict[x].split('_')[0] + '_' + entity_dict[x].split('_')[1]) if x in entity_dict else x
                       for x in metric_list]
                result_list.extend(list(Counter(rep)))
                print(result_list)

