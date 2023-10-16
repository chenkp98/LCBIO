import pandas as pd
import pickle


def read_config_file(config_file):
    params = {}
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=")
                params[key.strip()] = value.strip()
    return params

def print_params(params):
    print("参数：")
    for key, value in params.items():
        print(f"{key}: {value}")

def read_data(file_path):
    file_extension = file_path.split(".")[-1].lower()
    if file_extension == "csv":
        data = pd.read_csv(file_path)
    elif file_extension == "txt":
        data = pd.read_csv(file_path)
    elif file_extension == "xlsx":
        data = pd.read_excel(file_path)
    elif file_extension == "pickle":
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
        # 如果pickle文件中保存的是DataFrame对象，则直接返回
            return data.iloc[:, 1:-1], data.iloc[:, -1]
        else:
        # 如果pickle文件中保存的是其他对象，则根据具体情况进行处理
        # 这里只是一个示例，你可以根据你的数据类型进行适当的处理
            return pd.DataFrame(data), None
    else:
        raise ValueError("不支持的文件类型：{}".format(file_extension))

    x = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    return x, y
               


from sklearn.model_selection import train_test_split
def Train_test(X, y, test_size, random_state):                       
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)               
    return Xtrain, Xtest, Ytrain, Ytest             


               