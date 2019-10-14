import numpy as np
import utils.fitderivpackage103.gaussianprocess as gp
import matplotlib.pyplot as plt

#####
def findsmoothvariance(y, filtsig= 0.1, nopts= False):
    '''
    Estimates and then smooths the variance over replicates of data

    Arguments
    --
    y: data - one column for each replicate
    filtsig: sets the size of the Gaussian filter used to smooth the variance
    nopts: if set, uses estimateerrorbar to estimate the variance
    '''
    from scipy.ndimage import filters
    if y.ndim == 1:
        # one dimensional data
        v= estimateerrorbar(y, nopts)**2
    else:
        # multi-dimensional data
        v= np.var(y, 1)
    # apply Gaussian filter
    vs= filters.gaussian_filter1d(v, int(len(y)*filtsig))
    return vs

######
def mergedicts(original, update):
    '''
    Given two dicts, merge them into a new dict

    Arguments
    --
    x: first dict
    y: second dict
    '''
    z= original.copy()
    z.update(update)
    return z

######
def plotxyerr(x, y, xerr, yerr, xlabel= 'x', ylabel= 'y', title= '', color= 'b', figref= False):
    '''
    Plots a noisy x versus a noisy y with errorbars shown as ellipses.

    Arguments
    --
    x: x variable (a 1D array)
    y: y variable (a 1D array)
    xerr: (symmetric) error in x (a 1D array)
    yerr: (symmetric) error in y (a 1D array)
    xlabel: label for x-axis
    ylabel: label for y-axis
    title: title of figure
    color: default 'b'
    figref: if specified, allows data to be added to an existing figure
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    if figref:
        fig= figref
    else:
        fig= plt.figure()
    ax= fig.add_subplot(111)
    ax.plot(x, y, '.-', color= color)
    for i in range(len(x)):
        e= Ellipse(xy= (x[i], y[i]), width= 2*xerr[i], height= 2*yerr[i], alpha= 0.2)
        ax.add_artist(e)
        e.set_facecolor(color)
        e.set_linewidth(0)
    if not figref:
        plt.xlim([np.min(x-2*xerr), np.max(x+2*xerr)])
        plt.ylim([np.min(y-2*yerr), np.max(y+2*yerr)])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show(block= False)



######


class fitderiv:
    '''
    to fit data and estimate the time derivative of the data using Gaussian processes

    A typical work flow is:

    from fitderiv import fitderiv
    q= fitderiv(t, od, figs= True)
    q.plotfit('df')

    or, for example,

    plot(q.t, q.d, 'r.', q.t, q.y, 'b')

    Any replicate is fit separately, but the results are combined for predictions. The best-fit hyperparameters and their bounds are shown for each replicate.

    The minimum and maximum limits of the hyperparameters can also be changed from their default values. For example,

    q= fitderiv(t, d, bd= {0: [-1, 4], 2: [2, 6]})

    sets the boundaries for the first hyperparameter to be 10^-1 and 10^4 and the boundaries for the third hyperparameter to be 10^2 and 10^6.

    Log data and results are stored as
        q.t : time (an input)
        q.origd : the original data (an input)
        q.d : log of the data (unless logs= False)
        q.f : best-fit
        q.fvar : variance (error) in the best-fit
        q.df : fitted first time-derivative
        q.dfvar : variance (error) in the fitted first time-derivative
        q.ddf : fitted second time-derivative
        q.ddfvar : variance (error) in the fitted second time-derivative

    Statistics are stored in a dictionary, q.ds, with keys:
        'max df' : max time derivative
        'time of max df' : time at which the max time derivative occurs
        'inverse max df' : the timescale found from inverting the max time derivative
        'max f': the maximum value of the fitted curve
        'lag time' : lag time (when the tangent from the point of max time derivative crosses a line parallel to the x-axis and passing through the first data point)
    All statistics can be postfixed by ' var' to find the variance of the estimate.

    Please cite

    PS Swain, K Stevenson, A Leary, LF Montano-Gutierrez, IBN Clark, J Vogel, and T Pilizota.
    Inferring time derivatives including growth rates using Gaussian processes
    Nat Commun 7 (2016) 13766

    to acknowledge the software.
    '''

    def __init__(self, t, d, cvfn= 'sqexp', noruns= 5, exitearly= False, figs= False, bd= False,
                 esterrs= False, optmethod= 'l_bfgs_b', nosamples= 100, logs= True,
                 gui= False, figtitle= False, ylabel= 'y', stats= True, statnames= False,
                 showstaterrors= True, warn= False, linalgmax= 3):
        '''
        Runs a Gaussian process to fit data and estimate the time-derivative

        Arguments
        --
        t: array of time points
        d: array of data with replicates in columns
        cvfn: kernel function for the Gaussian process used in the fit - 'sqexp' (squared exponential: default), 'matern' (Matern with nu= 5/2), or 'nn' (neural network)
        noruns: number of fitting attempts made (default is 5)
        exitearly: if True, stop at the first successful fit; if False, take the best fit from all successful fits
        figs: plot the results of the fit
        bd: can be used to change the limits on the hyperparameters for the Gaussian process used in the fit
        esterrs:  if True, measurement errors are empirically estimated from the variance across replicates at each time point; if False, the size of the measurement error is fit from the data assuming that this size is the same at all time points
        optmethod: the optimization method to maximize the likelihood - 'l_bfgs_b' or 'tnc'
        nosamples: number of samples taken to estimate errors in statistics
        logs: if True, the natural logarithm is taking of the data points before fitting
        gui: if True, extra output is printed for the GUI
        figtitle: title of the figure showing the fit
        ylabel: label of the y-axis of the figure showing the fit
        stats: if True, summary statistics of fit and inferred derivative are calculated
        statnames: a list for specializing the names of the statistics
        showstaterrors: if True, display estimated errors for statistics
        warn: if False, warnings created by covariance matrices that are not positive semi-definite are stopped
        linalgmax: number of attempts (default is 3) if a linear algebra (numerical) error is generated
        '''
        self.version= '1.03'
        self.ylabel= ylabel
        self.logs= logs
        if not warn:
            # warning generated occasionally when sampling from the Gaussian process likely because of numerical errors
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
        try:
            noreps= d.shape[1]
        except:
            noreps= 1
        self.noreps= noreps
        self.t= t
        self.d= d
        # bounds for hyperparameters
        bnn= {0 : (-1,5), 1: (-7,-2), 2: (-6,2)}
        bsqexp= {0: (-5,5), 1: (-6,2), 2: (-5,2)}
        bmatern= {0: (-5,5), 1: (-4,4), 2: (-5,2)}
        # take log of data
        self.origd= d
        if logs:
            print('Taking natural logarithm of the data.')
            if np.any(np.nonzero(d < 0)):
                print('Negative data found, but all data must be positive if taking logs.')
                print('Ignoring request to take logs.')
            else:
                d= np.log(np.asarray(d))
        # run checks and define measurement errors
        merrors= False
        if np.any(esterrs):
            if type(esterrs) == type(True):
                # errors must be empirically estimated.
                if noreps > 1:
                    lod= [len(np.nonzero(~np.isnan(d[:,i]))[0]) for i in range(noreps)]
                    if np.sum(np.diff(lod)) != 0:
                        print('The replicates have different number of data points.')
                        print('Equal numbers of data points are needed for empirically estimating errors.')
                    else:
                        # estimate errors empirically
                        merrors= findsmoothvariance(d)
                        if figs:
                            plt.figure()
                            plt.errorbar(t, np.mean(d,1), np.sqrt(merrors))
                            plt.plot(t, d, '.')
                            plt.show(block= False)
                else:
                    print('Not enough replicates to estimate errors.')
            else:
                # esterrs given as an array of errors
                if len(esterrs) != len(t):
                    print('Each time point requires an estimated error.')
                else:
                    merrors= esterrs
        if not np.any(merrors):
            print('Fitting measurement errors.')
        # display details of covariance functions
        try:
            if bd:
                bds= mergedicts(original= eval('b' + cvfn), update= bd)
            else:
                bds= eval('b' + cvfn)
            if not gui:
                gt= getattr(gp, cvfn + 'GP')(bds, t, d)
                print('Using a ' + gt.description + '.')
                gt.info()
        except NameError:
            print('Gaussian process not recognized.')
            from sys import exit
            exit()
        self.bds= bds
        # combine data into one array
        tb= np.tile(t, noreps)
        db= np.reshape(d, np.size(d), order= 'F')
        if np.any(merrors):
            mb= np.tile(merrors, noreps)
        # remove any nans
        da= db[~np.isnan(db)]
        ta= tb[~np.isnan(db)]
        if np.any(merrors):
            ma= mb[~np.isnan(db)]
        else:
            ma= False
        # run Gaussian process
        g= getattr(gp, cvfn + 'GP')(bds, ta, da, merrors= ma)
        g.findhyperparameters(noruns, exitearly= exitearly, optmethod= optmethod, linalgmax= linalgmax)
        # display results of fit
        if gui:
            print('log(max likelihood)= %e' % (-g.nlml_opt))
            for el in g.hparamerr:
                if el[1] == 'l':
                    print('Warning: hyperparameter ' + str(el[0]) + ' is at a lower bound.')
                else:
                    print('Warning: hyperparameter ' + str(el[0]) + ' is at an upper bound.')
                print('\tlog10(hyperparameter %d)= %4.2f' % (el[0], np.log10(np.exp(g.lth_opt[el[0]]))))
        else:
            g.results()
        g.predict(t, derivs= 2, merrorsnew= merrors)
        fmnp= g.mnp
        fcovp= g.covp
        # save results
        self.g= g
        self.logmaxlike= -g.nlml_opt
        self.hparamerr= g.hparamerr
        self.lth= g.lth_opt
        self.fmnp= fmnp
        self.fcovp= fcovp
        self.t= t
        self.d= d
        self.f= fmnp[:len(t)]
        self.df= fmnp[len(t):2*len(t)]
        self.ddf= fmnp[2*len(t):]
        self.fvar= np.diag(fcovp)[:len(t)]
        self.dfvar= np.diag(fcovp)[len(t):2*len(t)]
        self.ddfvar= np.diag(fcovp)[2*len(t):]
        self.merrors= merrors
        if stats: self.calculatestats(nosamples, statnames, showstaterrors)
        if figs:
            plt.figure()
            self.plotfit()
            plt.xlabel('time')
            if logs:
                plt.ylabel('log ' + ylabel)
            else:
                plt.ylabel(ylabel)
            if figtitle:
                plt.title(figtitle)
            else:
                plt.title('mean fit +/- standard deviation')
            plt.show(block= False)



    def sample(self, nosamples, newt= False):
        '''
        Generate sample values for the latent function and its first two derivatives (returned as a tuple).

        Arguments
        ---
        nosamples: number of samples
        newt: if False, the orginal time points are used; if an array, samples are made for those time points
        '''
        if np.any(newt):
            newt= np.asarray(newt)
            import copy
            # make prediction for new time points
            gps= copy.deepcopy(g)
            gps.predict(newt, derivs= 2)
        else:
            newt= self.t
            gps= self.g
        noreps= self.noreps
        fghs= gps.sample(nosamples)
        f= fghs[:len(newt),:]
        g= fghs[len(newt):2*len(newt),:]
        h= fghs[2*len(newt):,:]
        return f, g, h


    def plotfit(self, char= 'f', errorfac= 1, xlabel= 'time', ylabel= False, figtitle= False):
        '''
        Plots the results of the fit.

        Arguments
        --
        char: the type of fit to plot - 'f' or 'df' or 'ddf'
        errorfac: sets the size of the errorbars to be errorfac times the standard deviation
        ylabel: the y-axis label
        figtitle: the title of the figure
        '''
        x= getattr(self, char)
        xv= getattr(self, char + 'var')
        if char == 'f':
            plt.plot(self.t, self.d, 'r.')
        plt.plot(self.t, x, 'b')
        plt.fill_between(self.t, x-errorfac*np.sqrt(xv), x+errorfac*np.sqrt(xv), facecolor= 'blue', alpha= 0.2)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(char)
        plt.xlabel(xlabel)
        if figtitle: plt.title(figtitle)




    def calculatestats(self, nosamples= 100, statnames= False, showerrors= True):
        '''
        Calculates statistics from best-fit curve and its inferred time derivative - 'max df', 'time of max df', 'inverse max grad', 'max f', 'lag time'.

        Arguments
        --
        nosamples: number of samples used to estimate errors in the statistics
        statnames: a list of alternative names for the statistics
        showerrors: display estimated errors for statistics
        '''
        print('\nCalculating statistics with ' + str(nosamples) + ' samples')
        if showerrors: print('\t(displaying mean +/- standard deviation [standard error])\n')
        if statnames:
            self.stats= statnames
        else:
            self.stats= ['max df', 'time of max df', 'inverse max df', 'max ' + self.ylabel, 'lag time']
        t, noreps= self.t, self.noreps
        fs, gs, hs= self.sample(nosamples)
        # calculate stats
        im= np.argmax(gs, 0)
        mgr= gs[im, np.arange(nosamples)]
        tmgr= np.array([t[i] for i in im])
        dt= np.log(2)/mgr
        if self.logs:
            md= np.exp(np.max(fs, axis= 0))
        else:
            md= np.max(fs, axis= 0)
        lagtime= tmgr + (fs[0, np.arange(nosamples)] - fs[im, np.arange(nosamples)])/mgr
        # store stats
        ds= {}
        for stname, st in zip(self.stats, [mgr, tmgr, dt, md, lagtime]):
            ds[stname]= np.mean(st)
            ds[stname + ' var']= np.var(st)
        self.ds= ds
        self.nosamples= nosamples
        self.printstats(showerrors= showerrors)



    def printstats(self, errorfac= 1, showerrors= True, performprint= True):
        '''
        Creates and prints a dictionary of the statistics of the data and its inferred time-derivative

        Arguments
        --
        errorfac: sets the size of the errorbars to be errorfac times the standard deviation
        showerrors: if True (default), display errors
        performprint: if True, displays results
        '''
        ds= self.ds
        statd= {}
        lenstr= np.max([len(s) for s in self.stats])
        for s in self.stats:
            statd[s]= ds[s]
            statd[s + ' std']= errorfac*np.sqrt(ds[s +' var'])
            statd[s + ' stderr']= np.sqrt(ds[s+' var'])/np.sqrt(self.nosamples)
            if performprint:
                stname= s.rjust(lenstr + 1)
                if showerrors:
                    print('{:s}= {:6e} +/- {:6e} [{:6e}]'.format(stname, statd[s], statd[s +' std'],
                                                                 statd[s + ' stderr']))
                else:
                    print('{:s}= {:6e}'.format(stname, statd[s]))
        return statd



    def plotstats(self):
        '''
        Produces a bar chart of the statistics.
        '''
        try:
            ds, stats= self.ds, self.stats
            data= []
            errs= []
            for s in stats:
                data.append(ds[s])
                errs.append(2*np.sqrt(ds[s + ' var']))
            fig= plt.figure()
            barwidth= 0.5
            ax= fig.add_subplot(111)
            plt.bar(np.arange(len(stats)), data, barwidth, yerr= errs)
            ax.set_xticks(np.arange(len(stats)) + barwidth/2.0)
            ax.set_xticklabels(stats)
            plt.show(block= False)
        except AttributeError:
            print(" Statistics have not been calculated.")


    def plotfvsdf(self, ylabel= 'f', title= ''):
        '''
        Plots fitted f versus inferred time-derivative using ellipses with axes lengths proportional to the error bars.

        Arguments
        --
        ylabel: label for the y-axis
        '''
        if self.logs:
            xlabel= 'fitted log ' + ylabel
        else:
            xlabel= 'fitted ' + ylabel
        ylabel= 'deriv ' + ylabel
        plotxyerr(self.f, self.df, np.sqrt(self.fvar), np.sqrt(self.dfvar), xlabel, ylabel, title)




    def export(self, fname, rows= False):
        '''
        Exports the fit and inferred time-derivative to a text or Excel file.

        Arguments
        --
        fname: name of the file (.csv files are recognized)
        rows: if True (default is False), data are exported in rows; if False, in columns
        '''
        import pandas as pd
        ods= self.origd
        data= [self.t, self.f, np.sqrt(self.fvar), self.df, np.sqrt(self.dfvar), ods]
        if ods.ndim == 1:
            labels= ['t', 'log(OD)', 'log(OD) error', 'gr', 'gr error', 'od']
        else:
            labels= ['t', 'log(OD)', 'log(OD) error', 'gr', 'gr error'] + ['od']*ods.shape[1]
        orgdata= np.column_stack(data)
        # make dataframes
        if rows:
            df= pd.DataFrame(np.transpose(orgdata), index= labels)
        else:
            df= pd.DataFrame(orgdata, columns= labels)
        statd= self.printstats(performprint= False)
        dfs= pd.DataFrame(statd, index= [0], columns= statd.keys())
        # export in appropriate format
        ftype= fname.split('.')[-1]
        if ftype == 'csv' or ftype == 'txt' or ftype == 'dat':
            if ftype == 'txt' or ftype == 'dat':
                sep= ' '
            else:
                sep= ','
            if rows:
                df.to_csv(fname, sep= sep, header= False)
            else:
                df.to_csv(fname, sep= sep, index= False)
            dfs.to_csv('.'.join(fname.split('.')[:-1]) + '_stats.' + ftype, sep= sep, index= False)
        elif ftype == 'xls' or ftype == 'xlsx':
            if rows:
                df.to_excel(fname, sheet_name= 'Sheet1', header= False)
            else:
                df.to_excel(fname, sheet_name= 'Sheet1', index= False)
            dfs.to_excel('.'.join(fname.split('.')[:-1]) + '_stats.xlsx', sheet_name= 'Sheet1', index= False)
        else:
            print('!! File type is either not recognized or not specified. Cannot save as', fname)


#####

if __name__ == '__main__': print(fitderiv.__doc__)
