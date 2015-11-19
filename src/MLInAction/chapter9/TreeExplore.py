import numpy
import Tkinter
import RegTrees
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def helloWorld():
    root = Tkinter.Tk()
    myLabel = Tkinter.Label(root, text="Hello world")
    myLabel.grid()
    root.mainloop()
    
def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: 
            tolN = 2
        myTree = RegTrees.createTree(reDraw.rawDat, RegTrees.modelLeaf, RegTrees.modelErr, (tolS, tolN))
        yHat = RegTrees.createForeCast(myTree, reDraw.testDat, RegTrees.modelTreeEval)
    else:
        myTree = RegTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = RegTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNentry.delete(0, Tkinter.END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0, Tkinter.END)
        tolSentry.insert(0, '1.0')
    return tolS, tolN

def drawNewTree():
    tolS, tolN = getInputs()
    reDraw(tolS, tolN)

root = Tkinter.Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Tkinter.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = Tkinter.IntVar()
chkBtn = Tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = numpy.mat(RegTrees.loadData("E:/TestDatas/MachineLearningInAction/Ch09/sine.txt"))
reDraw.testDat = numpy.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()

