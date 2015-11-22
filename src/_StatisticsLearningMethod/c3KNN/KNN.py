import __StaticContent.Path.PATH_SLM as path

class KDTree:

    class node:
        def __init__(self, point):
            self.left = None;
            self.right = None;
            self.point = point;
            pass;
    
    def __init__(self, data, d):
        self.root = self.build_kdtree(data, d);
    
    def build_kdtree(self, data, d):
        data = sorted(data, key=lambda x: x[d]);
        p, m = self.median(data);
        tree = self.node(p);
        
        del data[m];
#         print data, p
        
        if m > 0:
            tree.left = self.build_kdtree(data[:m], not d);
        if len(data) > 1:
            tree.right = self.build_kdtree(data[m:], not d);
        
        return tree;
    
    def median(self, lst):
        m = len(lst) / 2;
        return lst[m], m;


def main():
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]];
    kd_tree = KDTree(data, 0);

main();