def parse_model_cfg(path):  # 解析YOLOv3——1088...这些配置文件
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces（去掉每一行的空白）
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block，[代表着新块的开始
            module_defs.append({})  # 如果是一个新块的话，就引入一个新的字典
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 读取网络块的类型：卷积层、或者等等其它层
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")   # 如果不是一个新块的话，用键值对进行保存。还是放在 moudle_defs这个字典中。
                                           # 遍历循环之后，所有的yolov3_1088...的信息都保存在了moudle_defs这个字典中
            value = value.strip()
            if value[0] == '$':
                value = module_defs[0].get(value.strip('$'), None)
            module_defs[-1][key.rstrip()] = value
    # 函数返回的是一个列表，保存所有的层块
    return module_defs

def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
