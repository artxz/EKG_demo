from re import T
from pydub import AudioSegment
import matplotlib.pyplot as pl
import numpy as np
from scipy.signal import find_peaks
from glob import glob
import pandas as pd


class cardio():
    def __init__(self, file_path):
        self.fpath = file_path
        self.format = self.fpath.split('.')[-1]

    def load_data(self,):
        self.audio = AudioSegment.from_file(self.fpath, self.format)
        self.data = np.array(self.audio.get_array_of_samples())
        print('data succesfully loaded!')
        
        self.T = self.audio.frame_count()
        self.fps = self.audio.frame_rate
        self.xsec = np.arange(self.T)/self.fps
        self.xmin = self.xsec/60
        self.xmsec = self.xsec*1000

        self.df = pd.DataFrame(data = self.data, columns = ['data'])
        self.df['xsec'] = self.xsec
        self.df['xmin'] = self.xmin

    def load_events(self, txtpath = None):
        if txtpath is None:
            print(self.format)
            tmp = glob(self.fpath.split(self.format)[0] + '*txt')
            if len(tmp)>0:
                txtpath = tmp[0]
            else:
                print('no event file found')
        else:
            pass
        if txtpath is not None:
            print(txtpath)
            dft = pd.read_csv(txtpath, sep = ',\t', header=1, engine = 'python');
            dft.rename(columns={'# Marker ID':'ID', 'Time (in s)': 'time'}, inplace = True)
            stimIDs = [i for i in np.unique(dft['ID']) if len(str(i)) ==1]
            dft = dft[dft['ID'].isin(stimIDs)]
            dft['ID'] = dft['ID'].astype('int')
            self.dft = dft


    def run_analysis(self, threshold = None):
        self.peaks, self.threhold, self.ibi_sec, self.hr_sec, self.hr_min = self.find_mypeaks(self.data, threshold) 
        self.df['ibi'] = None
        self.df['hr'] = None
        self.df['hr_min'] = None
        self.df.loc[self.peaks[1:],'ibi'] = self.ibi_sec
        self.df.loc[self.peaks[1:],'hr'] = self.hr_sec
        self.df.loc[self.peaks[1:],'hr_min'] = self.hr_min

    def create_IDblocks(self,):
        blocks = {}
        for ind, i in enumerate(self.dft['ID']):
            start = self.dft['time'][ind]
            start_ind = self.df['xsec'][self.df['xsec']>start].index[0]    
            self.df.loc[start_ind, 'ID'] = int(i)
            if ind<len(self.dft['ID'])-1:
                end = self.dft['time'][ind+1]
                end_ind = self.df['xsec'][self.df['xsec']>end].index[0]
                self.df.loc[start_ind:end_ind, 'IDblocks'] = int(i)
            blocks[ind] = {}
            blocks[ind]['time'] = np.array([start, end]) 
            blocks[ind]['inds'] = np.array([start_ind, end_ind])
            blocks[ind]['ID'] = int(i) 
            blocks[ind]['dur'] = end-start
            blocks[ind]['ibi'] = self.df.loc[start_ind:end_ind, 'ibi'].dropna()
        self.blockinds = blocks

        self.IDs = np.unique(self.df['ID'][~self.df['ID'].isna()])
        for ind, i in enumerate(self.IDs):
            self.df[int(i)] = None
            self.df.loc[self.df['ID'] == i, int(i)] = 1
  

    def find_mypeaks(self, data, threshold = None, maxrate = 3):
        if threshold is None:
            height = np.percentile(data, 98)
        else:
            height = threshold
        peaks, _ = find_peaks(data, height = height, distance = int(self.fps/maxrate))
        ibi_sec = np.diff(peaks)/self.fps
        hr_sec = 1/(np.diff(peaks)/self.fps)
        hr_min = hr_sec*60
        return peaks, threshold, ibi_sec, hr_sec, hr_min 
    
    def get_stats(self, ibi):
        stats = {}
        stats['min'] = np.min(ibi)
        stats['max'] = np.min(ibi)
        stats['std'] = np.std(ibi)
        stats['mean'] = np.mean(ibi)
        return stats


def load_data(fpath, fps):
    ftype = fpath.split('.')[-1]
    print(ftype)
    audio = AudioSegment.from_file(fpath, ftype)
    audio_ds = audio.set_frame_rate(fps)
    data = np.array(audio_ds.get_array_of_samples())
    T = len(data)
    xax = np.arange(T)/fps
    return data, xax

def find_mypeaks(data, fps):
    height = np.percentile(data, 98)
    print(height)
    peaks, _ = find_peaks(data, height = height, distance = int(fps/3))
    ibi = np.diff(peaks)/fps
    return peaks, ibi


def plot_ibi_hist(ibi, spath = None, figsize = (6, 2), nbin = 50):
    pl.figure(figsize = figsize)
    pl.hist(ibi, bins = nbin)
    pl.title('inter beat interval')
    name = 'ibi-histogram'
    pl.savefig(spath + name + '.pdf',transparent = True)
    pl.savefig(spath + name + '.png',transparent = True)

    pl.figure(figsize = figsize)
    pl.hist(1/ibi*60, bins = nbin)
    pl.title('heart rate')
    name = 'hr-histogram'
    if spath is not None:
        pl.savefig(spath + name + '.pdf',transparent = True)
        pl.savefig(spath + name + '.png',transparent = True)

def plot_ibi(ibi, xax, peaks, spath = None, figsize = (6, 2)):
    hr = 1/ibi*60
    hrmin = np.min(hr)
    hrmax = np.max(hr)
    hrstd = np.std(hr)

    pl.figure(figsize = figsize)
    pl.plot(xax[peaks[1:]]/60, 1/ibi*60,'-')
    pl.title('heart rate over time\n min: %.2f, max:%.2f, std:%.2f' % (hrmin, hrmax, hrstd))
    pl.xlabel('time (min)')
    name = 'hr'
    if spath is not None:
        pl.savefig(spath + name + '.pdf',transparent = True)
        pl.savefig(spath + name + '.png',transparent = True)

def plot_peaks(data, xax, peaks, spath = None, sl = None, figsize = (21,4), markersize = 5):
    pl.figure(figsize = figsize)
    pl.plot(xax, data)
    pl.plot(xax[peaks], data[peaks],'.')
    name = 'peak detection - all'
    if spath is not None:
        pl.savefig(spath + name + '.pdf',transparent = True)
        pl.savefig(spath + name + '.png',transparent = True)

    pl.figure(figsize = figsize)
    pl.plot(xax, data)
    pl.plot(xax[peaks], data[peaks],'.', markersize = markersize)
    if sl is None:  
        pl.xlim([10,12])
    else:
        pl.xlim(sl)
    name = 'peak detection - {}'.format(sl)
    if spath is not None:
        pl.savefig(spath + name + '.pdf',transparent = True)
        pl.savefig(spath + name + '.png',transparent = True)

def plot_average_waveform(data, xax, peaks, fps, pre, post, spath = None, figsize = (12, 4)):
    data = np.array(data)
    pre, post = int(pre*fps), int(post*fps)
    xax = (np.arange(pre+post)-pre)/fps*1000
    T = len(data)
    inds = np.stack([np.arange(i-pre, i+post) for i in peaks if i+post<T])
    data_mat = data[inds].reshape([-1, pre+post])

    pl.figure(figsize = figsize)
    pl.title('average EKG waveform')
    pl.plot(xax, data_mat.T, alpha = .5, lw = 2)
    pl.plot(xax, data_mat.mean(0), c = 'k', lw = 2)
    pl.xlabel('time (msec)')
    name = 'average waveform'
    if spath is not None:
        pl.savefig(spath + name + '.pdf',transparent = True)
        pl.savefig(spath + name + '.png',transparent = True)