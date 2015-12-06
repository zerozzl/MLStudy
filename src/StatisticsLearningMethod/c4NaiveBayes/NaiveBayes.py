
def MLE(X, y, target):
    yp = float(y.count(1));
    yn = float(y.count(-1));
    
    py_p = yp / len(y);
    py_n = yn / len(y);
    
    pro_map = {};
    m = len(X);
    for i in range(m):
        key0 = "x0" + str(X[i][0]) + "_y" + str(y[i]);
        key1 = "x1" + str(X[i][1]) + "_y" + str(y[i]);
        pro_map[key0] = pro_map.get(key0, 0) + 1;
        pro_map[key1] = pro_map.get(key1, 0) + 1;
    
    pt_x0_yp = pro_map.get("x0" + str(target[0]) + "_y1", 0) / yp;
    pt_x0_yn = pro_map.get("x0" + str(target[0]) + "_y-1", 0) / yn;
    pt_x1_yp = pro_map.get("x1" + str(target[1]) + "_y1", 0) / yp;
    pt_x1_yn = pro_map.get("x1" + str(target[1]) + "_y-1", 0) / yn;
    
    pt_p = py_p * pt_x0_yp * pt_x1_yp;
    pt_n = py_n * pt_x0_yn * pt_x1_yn;
    
    if pt_p > pt_n:
        return 1;
    else:
        return -1;

def BayesianEstimation(X, y, target):
    yp = float(y.count(1));
    yn = float(y.count(-1));
    
    py_p = (yp + 1) / (len(y) + 2);
    py_n = (yn + 1) / (len(y) + 2);
    
    x0_set = set();
    x1_set = set();
    
    for x in X:
        x0_set.add(x[0]);
        x1_set.add(x[1]);
    
    pro_map = {};
    m = len(X);
    for i in range(m):
        key0 = "x0" + str(X[i][0]) + "_y" + str(y[i]);
        key1 = "x1" + str(X[i][1]) + "_y" + str(y[i]);
        pro_map[key0] = pro_map.get(key0, 0) + 1;
        pro_map[key1] = pro_map.get(key1, 0) + 1;
    
    pt_x0_yp = (pro_map.get("x0" + str(target[0]) + "_y1", 0) + 1) / (yp + len(x0_set));
    pt_x0_yn = (pro_map.get("x0" + str(target[0]) + "_y-1", 0) + 1) / (yn + len(x0_set));
    pt_x1_yp = (pro_map.get("x1" + str(target[1]) + "_y1", 0) + 1) / (yp + len(x1_set));
    pt_x1_yn = (pro_map.get("x1" + str(target[1]) + "_y-1", 0) + 1) / (yn + len(x1_set));
    
    pt_p = py_p * pt_x0_yp * pt_x1_yp;
    pt_n = py_n * pt_x0_yn * pt_x1_yn;
    
    if pt_p > pt_n:
        return 1;
    else:
        return -1;


def main():
    X = [[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2], [2, 3], [2, 3], 
         [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]];
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1];
    target = [2, 1];
    
#     predict = MLE(X, y, target);
    predict = BayesianEstimation(X, y, target);
    
    print predict;

main();