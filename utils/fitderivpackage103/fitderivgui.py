try:
    # Python3
    from tkinter import *
    from tkinter.messagebox import *
    from tkinter.filedialog import *
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    # Python2
    from Tkinter import *
    from tkMessageBox import *
    from tkFileDialog import *
    from ScrolledText import ScrolledText
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from fitderiv import fitderiv
import numpy as np
import pandas as pd
from os import path

# Version 1.02

class GuiOutput():
    font= ('courier', 16, 'normal')
    def __init__(self, parent= None):
        self.text= ScrolledText(parent)
        self.text.config(font= self.font)
        self.text.pack(side= BOTTOM, fill= BOTH, expand= YES)
    def write(self, text):
        self.text.insert(END, str(text))
        self.text.see(END)
        self.text.update()                       # update gui after each line
    def writelines(self, lines):
        for line in lines: self.write(line)



class fitgui(Frame):
    '''
    Launches gui and establishes main window
    '''

    # bounds on hyperparameters
    bds= {0 : (-5,5), 1: (-6,2), 2: (-5,2)}
    minh= -10
    maxh= 10


    def __init__(self, parent= None):
        '''
        Set up main window

        Arguments
        --
        parent: potential enclosing frame
        '''

        Frame.__init__(self, parent)
        self.pack(expand= YES, fill= BOTH)

        # add toolbar
        items= [ ('Load data', self.selectOpenFile, dict(side= LEFT)),
                 ('Run fit', self.runfit, dict(side= LEFT)),
                 ('Export fit', self.exportfit, dict(side= LEFT)),
                 ('Quit', self.quit, dict(side= RIGHT)),
                 ('Help', self.help, dict(side=RIGHT)) ]
        self.firstoff= ['Run fit', 'Export fit']
        toolbar= Frame(self, cursor= 'hand2', relief= SUNKEN, bd= 2)
        toolbar.pack(side= TOP, fill= X)
        self.tbuttons= {}
        for (name, action, where) in items:
            titem= Button(toolbar, text= name, command= action)
            titem.pack(where)
            self.tbuttons[name]= titem
            titem.config(fg= 'blue', font= ('helvetica', 25))
        for button in self.firstoff:
            self.tbuttons[button]['state']= 'disabled'

        # frame for data entry
        self.tf= Frame(self)
        self.tf.pack(side= TOP, fill= BOTH, expand= YES)
        self.tfl= Frame(self.tf)
        self.tfl.pack(side= LEFT, fill= X, expand= YES)
        self.tfr= Frame(self.tf)
        self.tfr.pack(side= RIGHT, fill= X, expand= YES)
        self.tflt= Frame(self.tfl)
        self.tflt.pack(side= TOP, fill= BOTH, expand= YES)
        self.tflb= Frame(self.tfl)
        self.tflb.pack(side= BOTTOM, fill= BOTH, expand= YES)

        # frame for output of fitderiv.py
        self.bf= Frame(self)
        self.bf.pack(side= BOTTOM, fill= BOTH, expand= YES)

        # redirect stdout and stderr to gui
        saveStreams= sys.stdout
        saveErr= sys.stderr
        gout= GuiOutput(self.bf)
        sys.stderr= gout
        sys.stdout= gout


    def quit(self):
        '''
        Quit GUI
        '''
        if askyesno('Verify quit', 'Are you sure you want to quit?'):
            root.quit()

    def help(self):
        '''
        Display a detailed workflow
        '''
        helpstr= ''
        helpw= Toplevel()
        helpw.title('Help')
        htext= Text(helpw)
        hscrollbar= Scrollbar(helpw)
        hscrollbar.pack(side= RIGHT, fill= Y)
        htext.pack(side= LEFT, fill= BOTH, expand= True)
        hscrollbar.config(command= htext.yview)
        htext.config(yscrollcommand= hscrollbar.set)
        htext.config(font= ('helvetica', 20), width= 40, height= 24, wrap= WORD)
        htext.insert(END, '1. Load the data set. Data can be from Excel or as a .txt or .dat (data separated by spaces) or .csv (data separated by commas) file with the measurement times first (in any units) and then the corresponding OD measurements. Replicate OD measurements can also be included. The data can be in either rows or columns (the software assumes more time points than replicates).\n\n')
        htext.tag_add('no', '1.0', '1.2')
        htext.insert(END, '2. Select which of the replicates to fit.\n\n')
        htext.tag_add('no', '3.0', '3.2')
        htext.insert(END, '3. Choose the number of samples for estimating statistics, such as the maximum growth rate and lag time. Latent functions (and their time-derivative) consistent with the data are sampled, the statistics calculated for each sample latent function, and the average statistic over all samples displayed.\n\nMore samples give more robust measurements but take more computing time.\n\n')
        htext.tag_add('no', '5.0', '5.2')
        htext.insert(END, '4. The maximum and minimum possible values of the hyperparameters can be changed using the sliders. The bounds are given in log10 space and so, for example, -1 specifies a bound of 0.1 in real space. \n\nHyperparameter 0 determines the amplitude (the magnitude of the variation) in the fitted curve; hyperparameter 1 determines how flexible the curve will be (smaller values imply a straighter curve); and hyperparameter 2 determines the magnitude of the measurement error.\n\nIf the fit is poorly following the data, we recommend decreasing the upper bound on the measurement error.\n\n')
        htext.tag_add('no', '9.0', '9.2')
        htext.insert(END, '5. Run the fit.\n\nAn initial random guess for the best-fit hyperparameters is made and then optimized.\n\nChanging the bounds on the hyperparameters may improve the fitting.\n\n')
        htext.tag_add('no', '15.0', '15.2')
        htext.insert(END, '6. The results of the fit are displayed both numerically and graphically.\n\nThe logarithm of the maximum likelihood is shown for the fit of each replicate with a higher value implying a better fit. A warning is given if a best-fit hyperparameter lies on a boundary, but having a hyperparameter on a boundary does not necessarily imply a bad fit. It is often best to judge the fit visually.\n\nTwo graphs in one figure are produced: one of the natural logarithm of the OD data and the best fit of this OD data and the other of the inferred growth rate, both as a function of time. The best-fit is shown in dark blue (the mean of the Gaussian process with the best-fit hyperparameters) and the estimated error (the standard deviation) is shown in light blue.\n\nThe maximum growth rate, the time at which this maximum growth rate occurs, the corresponding doubling time, the maximum value reached by the OD, and the lag time are also calculated.\n\n')
        htext.tag_add('no', '21.0', '21.2')
        htext.insert(END, '7. The results of the fit can be exported. The measurement times, the best-fit OD curve, the error-bar for this best-fit OD curve (the standard deviation), the inferred growth rate, the error-bar for this growth rate, and the original OD data used in the fit are all exported in one file (either .xlsx, .csv, .txt, or .dat). The statistics describing the growth curve are automatically exported in another.\n\n')
        htext.tag_add('no', '29.0', '29.2')
        htext.insert(END, '8. Three further fitting parameters can be specified.\n\nThe number of runs of the optimization routine can be increased. With multiple runs, the best result is taken (the one with the highest likelihood).\n\nThe natural logarithm of the data is taken by default (as is typical for fitting optical densities), but the original data can be fit instead by toggling the "Log data" box. \n\nA parameter describing the magnitude of the measurement errors is fit by default. An underlying assumption of this fitting is that the typical size of the measurement errors does not change with time. If the data do not appear to have a uniform measurement error, then the relative magnitude of the measurement error can be empirically estimated by toggling the "Non-uniform errors" box, although there must be multiple replicates to do so.\n\n')
        htext.tag_add('no', '31.0', '31.2')
        htext.insert(END, '9. Trouble-shooting:\n\nIf there is a poor fit to the data, try decreasing the upper bound on the hyperparameter for the measurement error.\n\nIf the best-fit appears to change when you re-run the fitting, then the optimization is probably finding a local optimum. Try increasing the number of runs, say to either 5 or 10.\n\nIf the inferred growth rate is too "noisy", try either increasing the lower bound on the hyperparameter for the measurement error to prevent over-fitting or decreasing the upper bound on the hyperparameter for the flexibility.\n\n')
        htext.tag_add('no', '39.0', '39.2')
        htext.insert(END, '10. Acknowledgment:\n\nPS Swain, K Stevenson, A Leary, LF Montano-Gutierrez, IBN Clark, J Vogel, and T Pilizota. Inferring time derivatives including growth rates using Gaussian processes. Nat Commun 7 (2016) 13766')
        htext.tag_add('no', '47.0', '47.3')
        htext.tag_config('no', foreground= 'OrangeRed4', font= ('helvetica', 25))
        htext.config(state= DISABLED)


    def selectOpenFile(self, ifile= "", idir= "."):
        '''
        Load in data

        Arguments
        --
        file: name of file to be opened
        dir: directory containing file
        '''

        # subframes
        tflt, tfr, tflb= self.tflt, self.tfr, self.tflb

        # load data
        fname= askopenfilename(filetypes= (('Excel files', ('*.xls', '*.xlsx')),
                                           ('CSV files', '*.csv'),
                                           ('Text files', '*.txt'),
                                           ('Matlab files', '*.dat')),
                                           initialdir= idir, initialfile= ifile)
        if fname:
            self.direc= path.split(ifile)[0]
            ftype= fname.split('.')[-1]
            if ftype == 'txt' or ftype  == 'dat' or ftype == 'csv':
                if ftype == 'csv':
                    ld= pd.read_csv(fname).values
                else:
                    ld= pd.read_table(fname).values
            elif ftype == 'xls' or ftype == 'xlsx':
                ld= pd.ExcelFile(fname).parse(0).values
            else:
                showerror('Open File', 'Data format is not recognized.')
            if ld.shape[0] < ld.shape[1]:
                # data organized in rows
                self.rows= True
                self.t= ld[0,:]
                self.d= np.transpose(ld[1:,:])
            else:
                # data organized in columns
                self.rows= False
                self.t= ld[:,0]
                self.d= ld[:,1:]

            # activate 'Run fit' button
            if hasattr(self, 'd'): self.tbuttons[self.firstoff[0]]['state']= 'normal'

            # clear frames (for a new data set)
            for widget in tflt.winfo_children():
                widget.destroy()
            for widget in tflb.winfo_children():
                widget.destroy()
            for widget in tfr.winfo_children():
                widget.destroy()
            if hasattr(self, 'sepf'):
                self.sepf.destroy()

            # check buttons to choose replicates to analyze
            Label(tflt, text= 'Choose which replicates to fit',
                bg= "lightsteelblue1", relief= SUNKEN).pack(side= TOP, fill= X, expand= YES)
            self.cbvars= []
            for key in range(self.d.shape[1]):
                var= IntVar()
                var.set(1)
                Checkbutton(tflt, text= str(key), variable= var).pack(side= LEFT, anchor= W, expand= YES)
                self.cbvars.append(var)

            Label(tflb, text= 'Set fitting parameters',
                bg= "lightsteelblue1", relief= SUNKEN).pack(side= TOP, fill= X, expand= YES)

            # choose number of runs
            Label(tflb, text= 'No. of runs').pack(side= LEFT, anchor= W)
            self.ent0= Entry(tflb, width= 2)
            self.ent0.insert(0, '3')
            self.ent0.pack(side= LEFT, anchor= W, expand= YES)

            # choose number of samples
            Label(tflb, text= 'No. of samples').pack(side= LEFT, anchor= W)
            self.ent1= Entry(tflb, width= 4)
            self.ent1.insert(0, '100')
            self.ent1.pack(side= LEFT, anchor= W, expand= YES)

            # toggle logs
            self.logvar= IntVar()
            self.logvar.set(1)
            Checkbutton(tflb, text= 'Log data', variable= self.logvar).pack(side= LEFT, anchor= W, expand= YES)

            # toggle estimate errors
            self.errvar= IntVar()
            self.errvar.set(0)
            Checkbutton(tflb, text= 'Non-uniform errors',
                        variable= self.errvar).pack(side= LEFT, anchor= W, expand= YES)

            # choose hyperparameters
            Label(tfr, text= 'Min and max of hyperparameters in log10',
                bg= "lightsteelblue1", relief= SUNKEN).pack(side= TOP, fill= X, expand= YES)
            self.sbvars= []
            for i in range(len(self.bds)):
                tempf= Frame(tfr)
                tempf.pack(side= LEFT, padx= 10, expand= YES)
                labelstr= 'param ' + str(i)
                if i == 0:
                    labelstr += '\n(amplitude)\n'
                elif i == 1:
                    labelstr += '\n(flexibility)\n'
                elif i == 2:
                    labelstr += '\n(measurement\nerror)'
                Label(tempf, text= labelstr).pack(side= TOP, fill= X)
                varl= DoubleVar()
                varu= DoubleVar()
                varl.set(self.bds[i][0])
                varu.set(self.bds[i][1])
                for var in [varl, varu]:
                    Scale(tempf, variable= var, from_= self.minh, to= self.maxh,
                        showvalue= YES).pack(side= LEFT, expand= YES)
                    self.sbvars.append(var)

            # add separator
            self.sepf= Frame(self, height=2, bd=1, relief=SUNKEN)
            self.sepf.pack(side= TOP, fill=X, padx=5, pady=5)



    def runfit(self):
        '''
        Call derivfit to fit the data and infer the time-derivative
        '''

        t= self.t

        # get number of runs
        self.noruns= int(self.ent0.get())

        # get number of samples
        self.nosamples= int(self.ent1.get())

        # choose replicates to analyze
        keep= []
        for i, var in enumerate(self.cbvars):
            if var.get() == 1:
                keep.append(i)
        od= self.d[:, keep]

        # log data?
        if self.logvar.get():
            logs= True
        else:
            logs= False

        # fit or empirically estimate errors
        if self.errvar.get():
            esterrs= True
        else:
            esterrs= False

        # get bounds for hyperparameters
        bns= [var.get() for var in self.sbvars]
        ubds= {}
        for i, j in zip(range(int(len(bns)/2)), np.arange(0, len(bns), 2)):
            ubds[i]= (bns[j], bns[j+1])

        # run fit
        print('\nStarting fit.')
        self.f= fitderiv(t, od, noruns= self.noruns, nosamples= self.nosamples, bd= ubds,
                        logs= logs, esterrs= esterrs, gui= keep,
                        statnames= ['max growth rate','time of max growth rate',
                                    'doubling time', 'max OD', 'lag time'])
        plt.figure()
        plt.subplot(2,1,1)
        self.f.plotfit('f', ylabel= 'log(OD)')
        plt.subplot(2,1,2)
        self.f.plotfit('df', ylabel= 'growth rate')
        plt.show(block= False)

        # tidy up
        print('\nFinished fit.\n')
        root.lift()

        # activate 'Export fit' button
        if hasattr(self, 'f'): self.tbuttons[self.firstoff[1]]['state']= 'normal'


    def exportfit(self):
        '''
        Save the results of the fit to two files: one for the data and the fit and another for the statistics
        '''
        if hasattr(self, 'f'):
            fname= asksaveasfilename(filetypes= (('Excel files', ('*.xls', '*.xlsx')),
                                           ('CSV files', '*.csv'),
                                           ('Text files', '*.txt'),
                                           ('Matlab files', '*.dat')),
                                           initialfile= "", initialdir= self.direc)
            self.f.export(fname, rows= self.rows)
        else:
            showerror('Export Data', 'No fitting has been performed yet.')


########


root= Tk()
root.title("The deODorizer")
fg= fitgui(root)
root.mainloop()
