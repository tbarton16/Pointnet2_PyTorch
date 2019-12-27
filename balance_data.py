import os
import csv
import numpy as np
version = 4
n_pts = 4096
output_dir = f"/home/theresa/p/datav{version}_balanced"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
input_dir = f"/home/theresa/p/datav{version}"
for ff in os.listdir(input_dir):
    if ff != '.csv':
        sub = os.path.join(input_dir, ff)
        outpath = os.path.join(output_dir, ff)
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        for fi in os.listdir(sub):
            filename = os.path.join(sub, fi)
            outfile = os.path.join(outpath, fi)
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
            if (len(class_a + class_b)< n_pts):
                print("not enough points")
                continue
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
            with open(outfile, "w") as o:
                reader = csv.writer(o)
                for row in newa + newb:
                    reader.writerow(row)






