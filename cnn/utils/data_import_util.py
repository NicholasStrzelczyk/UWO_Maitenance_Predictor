from sklearn.model_selection import train_test_split


def data_to_xy(data, seperator=" "):
    x_set, y_set = [], []
    for line in data:
        x, y = line.split(seperator)
        x_set.append(x.strip())
        y_set.append(y.strip())
    return x_set, y_set


def get_data_from_list(list_path, split=None):
    all_data = []
    for line in open(list_path, "r"):
        all_data.append(line)

    x1, y1, x2, y2 = None, None, None, None

    if split is not None:
        assert (0.01 <= split <= 0.99)
        data_p1, data_p2 = train_test_split(all_data, test_size=split, random_state=42, shuffle=True)
        x1, y1 = data_to_xy(data_p1, seperator=" ")
        x2, y2 = data_to_xy(data_p2, seperator=" ")
    else:
        x1, y1 = data_to_xy(all_data, seperator=" ")

    return x1, y1, x2, y2