import os
import csv
import numpy as np
n_pts = 4096
output_dir = "/home/theresa/p/datav1_balanced"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
input_dir = "/home/theresa/p/datav1"
for ff in os.listdir(input_dir):
    if ff != '.csv':
        filename = os.path.join(input_dir, ff)
        outfilename = os.path.join(output_dir, ff)
        class_a, class_b = [], []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            dists = []
            for row in reader:
                data = [float(element) for element in row[:]]
                c = float(row[-1])
                # print(distance)
                if c == 0.:
                    class_a.append(data)
                elif c == 1.:
                    class_b.append(data)
                else:
                    print("unknown class")
        # make class a the smaller class
        class_c = class_a
        if len(class_b) < len(class_a):
            class_a = class_b
            class_b = class_c

        # sample half of each
        class_a_n = min(len(class_a), n_pts / 2)
        remainder = 0
        if class_a_n < n_pts / 2:
            remainder = (n_pts / 2) - class_a_n
        class_b_n = remainder + (n_pts / 2)
        bidx = np.random.choice(len(class_b), int(class_b_n), replace=False)
        aidx = np.random.choice(len(class_a), int(class_a_n), replace=False)
        newa = [a  for idx, a in filter( lambda x:x[0] in aidx, list(enumerate(class_a)))]
        newb = [b  for idx, b in filter( lambda x:x[0] in bidx, list(enumerate(class_b)))]
        # print(len(newa + newb))
        with open(outfilename, "w") as o:
            reader = csv.writer(o)
            for row in newa + newb:
                reader.writerow(row)






