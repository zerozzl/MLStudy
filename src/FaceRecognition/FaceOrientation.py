import numpy as np
import matplotlib.pyplot as plt

def read_pgm(pgm):
    fr = open(pgm);
    header = fr.readline();
    magic = header.split()[0];
    maxsample = 1 if (magic == 'P4') else 0;
    while (len(header.split()) < 3 + (1, 0)[maxsample]):
        s = fr.readline();
        header += s if (len(s) and s[0] != '#') else '';
    width, height = [int(item) for item in header.split()[1:3]];
    samples = 3 if (magic == 'P6') else 1;
    if(maxsample == 0):
        maxsample = int(header.split()[3]);
    pixels = np.fromfile(fr, count=width * height * samples, dtype='u1' if maxsample < 256 else '>u2');
    pixels = pixels.reshape(height, width) if samples == 1 else pixels.reshape(height, width, samples);
    return pixels, height, width;

def plot_pgm(pgm):
    img = np.mat(pgm);
    fig = plt.figure();
    fig.add_subplot(111);
    plt.imshow(img , cmap="gray");
    plt.show();

pixels, height, width = read_pgm("/home/hadoop/ProgramDatas/FaceRecognition/faces_4/an2i/an2i_left_angry_open_4.pgm");
plot_pgm(pixels);
