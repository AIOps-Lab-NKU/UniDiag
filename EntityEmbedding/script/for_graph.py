import numpy 
import os

# GAIA
def for_GAIA_graph(config):
    # index: cases' index in training set
    index = [0,
    1,
    3,
    4,
    5,
    6,
    9,
    13,
    14,
    16,
    17,
    18,
    20,
    23,
    24,
    25,
    26,
    27,
    28,
    30,
    31,
    32,
    33,
    35,
    36,
    39,
    40,
    43,
    44,
    46,
    47,
    48,
    50,
    51,
    53,
    54,
    56,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    67,
    68,
    70,
    71,
    72,
    74,
    76,
    77,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    90,
    91,
    92,
    93,
    95,
    98,
    99,
    100,
    101,
    102,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    124,
    125]
    # train_index: all timestamp in training set cases.
    train_index = []
    for i in range(len(index)):
        if index[i]<49:
            temp = [j for j in range(index[i]*10,index[i]*10+10)]
            train_index += temp
        else:
            temp = [j for j in range(index[i]*10-1,index[i]*10+9)]
            train_index += temp

    a = numpy.load(os.path.join(os.path.join(config['output_data_path_v'],'file_moving'),'file_moving.npz'))
    b = numpy.load(os.path.join(os.path.join(config['output_data_path_v'],'normal_memory_freed_label'),'normal_memory_freed_label.npz'))
    c = numpy.load(os.path.join(os.path.join(config['output_data_path_v'],'memory_anomalies'),'memory_anomalies.npz'))
    d = numpy.load(os.path.join(os.path.join(config['output_data_path_v'],'access_permission_denied'),'access_permission_denied.npz'))
    all_data = list(a['x'])
    b_x = b['x']
    c_x = c['x']
    d_x = d['x']
    label = list(a['y'])
    b_y = b['y']
    c_y = c['y']
    d_y = d['y']
    for i in range(len(b_x)):
        if b_y[i] == 1:
            all_data.append(b_x[i])
            label.append(2)
    for i in range(len(c_x)):
        if c_y[i] == 1:
            all_data.append(c_x[i])
            label.append(3)
    for i in range(len(d_x)):
        if d_y[i] == 1:
            all_data.append(d_x[i])
            label.append(4)
            
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i in range(len(all_data)):
        if i in train_index:
            train_data.append(all_data[i])
            train_label.append(label[i])
        else:
            test_data.append(all_data[i])
            test_label.append(label[i])

    a_data = numpy.array(all_data)
    a_label = numpy.array(label)

    t_data = numpy.array(train_data)
    t_label = numpy.array(train_label)
    if not os.path.exists(config['output_data_path_g']):
        os.makedirs(config['output_data_path_g'])
    numpy.savez(os.path.join(config['output_data_path_g'],'GAIA_all.npz'),x = a_data,y=a_label)
    numpy.savez(os.path.join(config['output_data_path_g'],'GAIA_train.npz'),x = t_data,y=t_label)

