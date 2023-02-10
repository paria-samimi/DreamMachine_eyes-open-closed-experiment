% This is for pre-processing EEG signal generating from DreamMachine
% based on fildtrip codes which are adjusted to our data and device
% written by Paria Samimi 
%% load data
clear variables;
close all;
clc;

restoredefaultpath
addpath('/Users/pariasamimi/fieldtrip')
addpath('/Users/pariasamimi/Downloads/MATLAB/xdf')
addpath('/Users/pariasamimi/fieldtrip/external/brewermap')
addpath('/Users/pariasamimi/fieldtrip/external/matplotlib')
addpath('/Users/pariasamimi/fieldtrip/external/cmocean')
addpath('/Users/pariasamimi/fieldtrip/utilities/private');
addpath('/Users/pariasamimi/Documents/PhD/Elisa Project/codes')
addpath('/Applications/brainstorm3')
addpath('/Users/pariasamimi/Documents/PhD/Elisa Project/codes/fieldtrip')
dir_tmp='/Users/pariasamimi/Documents/PhD/Elisa Project/codes';
A=cd(dir_tmp);
ft_defaults
%% define the filenames, parameters and other information that is subject specific
subjectdata.subjectdir        = '/Users/pariasamimi/Documents/PhD/Elisa Project/codes';
subjectdata.datadir           = '/Users/pariasamimi/Documents/PhD/Elisa Project/codes';
subjectdata.subjectnr         = 'EyesOpenClosedAmplifier';
subjectdata.badtrials         = []; % subject made a mistake on the first and third trial

%% .xdf 
% for p = 1:length(participant)    
subjectdata.subjectdir=cd('/Users/pariasamimi/Documents/PhD/Elisa Project/codes'); % directory of file
EEG.EEG='eoec-2.xdf'; % change the name of file
namE=erase(EEG.EEG,'.xdf'); %% for removing the xdf name
[Data, Event] = xdf2fieldtrip(EEG.EEG);
%% 1. defining trials
cfg                 = [];
cfg.dataset         = EEG.EEG;
cfg.trialfun        = 'ft_trialfun_trigtraum_amp'; % in this file the number of the triggers and filename should change
[cfg]                = ft_definetriall(cfg);
addpath('/Users/pariasamimi/Documents/PhD/Elisa Project/codes')
%% making your data .xdf
for i=1:8 % number of triggers
    duration_start(i,:)=cfg.trl(i,1);
    duration_end(i,:)=cfg.trl(i,2);
    dtime=Data.time; 
    dtril=Data.trial;
    datapreproc.sampleinfo(i,:)=cfg.trl(i,1:2);
    datapreproc.trialinfo(i,:)=cfg.trl(i,3:4);
end
%time
datapreproc=[];
datapreproc.time{:,1}=dtime{1}(1,duration_start(1,1):duration_end(1,1));   
datapreproc.time{:,2}=dtime{1}(1,duration_start(2,1):duration_end(2,1));  
datapreproc.time{:,3}=dtime{1}(1,duration_start(3,1):duration_end(3,1));   
datapreproc.time{:,4}=dtime{1}(1,duration_start(4,1):duration_end(4,1));  
datapreproc.time{:,5}=dtime{1}(1,duration_start(5,1):duration_end(5,1));   
datapreproc.time{:,6}=dtime{1}(1,duration_start(6,1):duration_end(6,1));  
datapreproc.time{:,7}=dtime{1}(1,duration_start(7,1):duration_end(7,1));   
datapreproc.time{:,8}=dtime{1}(1,duration_start(8,1):duration_end(8,1));  
% datapreproc.time{:,9}=dtime{1}(1,duration_start(9,1):duration_end(9,1));   
% datapreproc.time{:,10}=dtime(1,duration_start(10,1):duration_end(10,1));  
%trial
datapreproc.trial{:,1}=dtril{1}(:,duration_start(1,:):duration_end(1,:));   
datapreproc.trial{:,2}=dtril{1}(:,duration_start(2,:):duration_end(2,:));  
datapreproc.trial{:,3}=dtril{1}(:,duration_start(3,:):duration_end(3,:));   
datapreproc.trial{:,4}=dtril{1}(:,duration_start(4,:):duration_end(4,:));  
datapreproc.trial{:,5}=dtril{1}(:,duration_start(5,:):duration_end(5,:));   
datapreproc.trial{:,6}=dtril{1}(:,duration_start(6,:):duration_end(6,:));  
datapreproc.trial{:,7}=dtril{1}(:,duration_start(7,:):duration_end(7,:));   
datapreproc.trial{:,8}=dtril{1}(:,duration_start(8,:):duration_end(8,:));  
% datapreproc.trial{:,9}=dtril(:,duration_start(9,:):duration_end(9,:));   
% datapreproc.trial{:,10}=dtril(:,duration_start(10,:):duration_end(10,:));  

datapreproc.hdr=Data.hdr;
%
datapreproc.label=Data.label;
%
datapreproc.fsample=250;
%
datapreproc.cfg=cfg;
%% 2. preprocessing and referencing 

cfg.padding         = 0; % length (in seconds) to which the trials are padded for filtering (default = 0)
%   cfg.padtype      = string, type of padding (default: 'data' padding or 'mirror', depending on feasibility)
cfg.continuous      = 'yes'; % whether the file contains continuous data

cfg.lpfilter        = 'yes'; % or 'yes'  lowpass filter (default = 'no')
cfg.hpfilter        = 'yes'; % or 'yes'  highpass filter (default = 'no')
cfg.bpfilter        = 'no'; % or 'yes'   bandpass filter (default = 'no')
cfg.bsfilter        = 'no'; % or 'yes'  bandstop filter (default = 'no')
cfg.dftfilter       = 'no'; % or 'yes'  line noise removal using discrete fourier transform (default = 'no')
cfg.medianfilter    = 'no'; % or 'yes'  jump preserving median filter (default = 'no')
cfg.lpfreq        = 50;                %lowpass  frequency in Hz
cfg.hpfreq        = 2;              %highpass frequency in Hz
%   cfg.bpfreq        = bandpass frequency range, specified as [lowFreq highFreq] in Hz
%   cfg.bsfreq        = bandstop frequency range, specified as [low high] in Hz (or as Nx2 matrix for notch filter)
% cfg.bpfreq          = [1 40]; % bandpass frequency range, specified as [lowFreq highFreq] in Hz
% cfg.bpfilttype      = 'but'; % digital filter type, 'but' or 'firws' or 'fir' or 'firls' (default = 'but')

% cfg.demean          = 'no'; % or 'yes', whether to apply baseline correction (default = 'no')
cfg.baselinewindow  = [-0.1 0.02]; % in seconds, the default is the complete trial (default = 'all')

cfg.detrend         = 'no'; % or 'yes', remove linear trend from the data (done per trial) (default = 'no')
cfg.polyremoval     = 'no'; % or 'yes', remove higher order trend from the data (done per trial) (default = 'no')
cfg.derivative      = 'no'; % or 'yes', computes the first order derivative of the data (default = 'no')
cfg.hilbert         = 'no'; % 'abs', 'complex', 'real', 'imag', 'absreal', 'absimag' or 'angle' (default = 'no')
cfg.rectify         = 'no'; % or 'yes' (default = 'no')
%   cfg.precision       = 'single' or 'double' (default = 'double')

%cfg.refchannel      = subjectdata.refchannel; % {'TP10'};
cfg.reref           = 'no'; % re-referencing
% cfg.channel         = {'all','-Traumschreiber-EEG_M1','-Traumschreiber-EEG_P3','-Traumschreiber-EEG_FPZ','-Traumschreiber-EEG_FP2','-Traumschreiber-EEG_M2','-Traumschreiber-EEG_POZ','-Traumschreiber-EEG_T7','-Traumschreiber-EEG_CZ','-Traumschreiber-EEG_F7','-Traumschreiber-EEG_F8'}; % remove bad channels
cfg.channel         = {'all','-Traumschreiber-EEG_POZ','-Traumschreiber-EEG_O2','-Traumschreiber-EEG_M1'};
cfg.refmethod       = 'avg'; %'avg', 'median', or 'bipolar' for bipolar derivation of sequential channels (default = 'avg')
cfg.groupchans    = 'no';
% cfg.method          = 'trial'; % or 'channel', read data per trial or per channel (default = 'trial')
cfg.trials        = 'all';

datapreproc         = ft_preprocessing(cfg,datapreproc);
save(strcat((namE),'_Preprocessing','.mat'),'datapreproc') % save the trial definition
load(strcat((namE),'_Preprocessing','.mat'),'datapreproc')

%%
cfg          = [];
cfg.method   = 'summary';
cfg.ylim     = [-1e-12 1e-12];
dummy        = ft_rejectvisual(cfg, datapreproc);

%% clean data
cfg          = [];
cfg.method   = 'trial';
cfg.ylim     = 'all';
%The following settings are useful for identifying muscle artifacts:
  cfg.bpfilter    = 'no';
  cfg.bpfreq      = [110 140];
  cfg.bpfiltord   =  8;
  cfg.bpfilttype  = 'but';
  cfg.rectify     = 'yes';
  cfg.boxcar      = 0.2;
cfg.megscale = 1;
cfg.eogscale = 5e-8;
dummy        = ft_rejectvisual(cfg, dummy);
datapreproc=dummy;
%% visually inspect the data
cfg            = [];
cfg.viewmode   = 'vertical';
ft_databrowser(cfg, datapreproc);
%% eyes closed
closed=[];

closed.trial{1,1}=datapreproc.trial{1, 1}; % odd matrix_close eye
closed.trial{1,2}=datapreproc.trial{1, 3}; % odd matrix_close eye
closed.trial{1,3}=datapreproc.trial{1, 5}; % odd matrix_close eye
closed.trial{1,4}=datapreproc.trial{1, 7}; % odd matrix_close eye

closed.time{:,1}=datapreproc.time{1, 1};
closed.time{:,2}=datapreproc.time{1, 3};
closed.time{:,3}=datapreproc.time{1, 5};
closed.time{:,4}=datapreproc.time{1, 7};

closed.sampleinfo(1,:)=datapreproc.sampleinfo(1, :);
closed.sampleinfo(2,:)=datapreproc.sampleinfo(3, :);
closed.sampleinfo(3,:)=datapreproc.sampleinfo(5, :);
closed.sampleinfo(4,:)=datapreproc.sampleinfo(7, :);

closed.hdr=datapreproc.hdr;
closed.label=datapreproc.label;
closed.fsample=250;
closed.cfg=cfg;
%% visually inspect the closed data
cfg            = [];
cfg.viewmode   = 'vertical';
ft_databrowser(cfg, closed);
%% eyes open
open=[];
open.trial{1,1}=datapreproc.trial{1, 2}; % odd matrix_close eye
open.trial{1,2}=datapreproc.trial{1, 4}; % odd matrix_close eye
open.trial{1,3}=datapreproc.trial{1, 6}; % odd matrix_close eye
open.trial{1,4}=datapreproc.trial{1, 8}; % odd matrix_close eye

open.time{:,1}=datapreproc.time{1, 2};
open.time{:,2}=datapreproc.time{1, 4};
open.time{:,3}=datapreproc.time{1, 6};
open.time{:,4}=datapreproc.time{1, 8};

open.sampleinfo(1,:)=datapreproc.sampleinfo(2, :);
open.sampleinfo(2,:)=datapreproc.sampleinfo(4, :);
open.sampleinfo(3,:)=datapreproc.sampleinfo(6, :);
open.sampleinfo(4,:)=datapreproc.sampleinfo(8, :);
open.hdr=datapreproc.hdr;
open.label=datapreproc.label;
open.fsample=250;
open.cfg=cfg;
%% visually inspect the open data
cfg            = [];
cfg.viewmode   = 'vertical';
ft_databrowser(cfg, open);
%% layout of channels locations
load('Antnew_with_ref.mat');
CH_loc=readtable('channels_loc.xlsx');
CH_loc=table2cell(CH_loc);
layout=[];
layout.pos=[-0.139058,0,0.139058,-0.54058,-0.328783,0,0.328783,0.54058,-0.67,-0.45,-0.225,0,0.225,0.54058,0.67,-0.54058,-0.328783,0,0.328783,0.54058,0,-0.139058,0,0.139058;0.430423,0.45,0.430423,0.285114,0.252734,0.25,0.252734,0.285114,0.04,0.05,0.05,0.05,0.05,0.05,0.04,-0.185114,-0.152734,-0.15,-0.152734,-0.185114,-0.25,-0.330422,-0.35,-0.330422]';
layout.width=lay.width(1:24,:);
layout.height=lay.height(1:24, :);
layout.label=Data.label;
%remove bad channels
layout.pos([9 21 24],:)=[]; %remove bad channels
layout.width([9 21 24],:)=[]; %remove bad channels
layout.height([9 21 24],:)=[]; %remove bad channels
layout.label([9 21 24],:)=[];%remove bad channels
%% plot layout
cfg        = [];
cfg.layout = layout;
ft_layoutplot(cfg)

%%  performs frequency and time-frequency analysis on time series data over multiple trials
cfg = [];
cfg.channel     = 'all';
cfg.method      ='mtmfft'; %'mtmfft', analyses an entire spectrum % 'mtmconvol', implements multitaper time-frequenc
cfg.output      = 'pow';      % return the power-spectra
cfg.taper     = 'hanning';
cfg.edgartnan = 'no';
cfg.foi          = 1:2:40;                         % analysis 1 to 40 Hz in steps of 2 Hz
cfg.t_ftimwin    = ones(length(cfg.foi),1).*1;   % length of time window = 0.5 sec
cfg.toi          =   0.5:0.25:45;                 % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)
freq_closed = ft_freqanalysis(cfg, closed);
freq_open = ft_freqanalysis(cfg, open);
%% plot
figure;
hold on;
ch=1;  % select number of channel
Namech=freq_closed.label{ch,1};
plot(freq_closed.freq, freq_closed.powspctrm(ch,:))
plot(freq_open.freq, freq_open.powspctrm(ch,:))% select number of channel
lgd=legend('eyes-closed', 'eyes-open');
lgd.FontSize = 12;
lgd.FontWeight = 'bold';
xlabel('Frequency (Hz)');
ylabel('absolute power (uV^2)');
title('\fontsize{16}powspctrm of Channel',Namech);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mean_Power_Spectra_open   = pow2db(squeeze(mean(freq_open.powspctrm,1)));
Mean_Power_Spectra_closed = pow2db(squeeze(mean(freq_closed.powspctrm,1)));

% %%  calculate descriptive statistics (e.g. mean, standard deviation) of the PSD data for each condition
% cfg = [];
% cfg.channel='all';
% closedfrq = ft_freqdescriptives(cfg, freq_closed);
% openfrq= ft_freqdescriptives(cfg, freq_closed);
% %% perform statistical analysis (e.g. t-test) to compare the PSD data between the eyes open and eyes closed conditions.
% cfg = [];
% cfg.channel='all';
% cfg.design           = [1 1 1 1 2 2 2 2]; % design matrix
% cfg.method           = 'montecarlo';
% [stat]= ft_freqstatistics(cfg, closedfrq, openfrq);