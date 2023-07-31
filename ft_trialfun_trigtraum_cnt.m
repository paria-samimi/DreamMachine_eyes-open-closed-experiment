function [trl, event] = ft_trialfun_trigtraum_cnt(cfg)

% This is based on the output needed for ft_definetrial but written for the
% traumschreiber data. Importantly, this is optimised for the first dataset
% and individual fields being accessed have to be adjusted for future
% recordings. 

% written by Paria Samimi & Debora Nolte

%% let's load the data again so we can manually adjust the event struct
% (from triggers_without_eeglab.m)
dir_tmp='/Users/pariasamimi/Documents/PhD/Elisa Project/Data/Fynn';
cd(dir_tmp);
A=dir;
streams = ft_read_data( 'Fynn.cnt');     %%%%%%%%%%%%%%%%%%%%% change%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

selected_ch=[1,2,3,4,5,6,7,8,13,14,15,16,17,18,19,24,25,26,27,28,29,30,31,32];
streams=streams(selected_ch, :);
%% now let's add to the event struct
event = [];
C = cell(1,11)'; % here adjust length so it will get it manually      %%%%%%%%%%%%%%%%%%%%% change%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C(:) = {'Markers'};
T = cell2table(C,'VariableNames',{'type'});
event = table2struct(T);
event=event';
values = num2cell([0 1 0 1 0 1 0 1 0 1 0]');
[event.value] = values{:};
% find out the minimum indicies
v=([0 2 4 6 8 10 12 14 16 18 20]*60)*1024;
ts_event = v;
ts_data =(1:length(streams))';
B = repmat(ts_data,[1,length(ts_event)]);
[~,closestIndex] = min(abs(B-ts_event));
C = num2cell(closestIndex');
[event.sample] = C{:};

% add event.duration
C = num2cell(1 * ones(length(values),1))';
[event.duration] = C{:};
% add event.offset
C = num2cell(zeros(length(values),1))';
[event.offset] = C{:};
%%
trl = [];
for i = 1:length(closestIndex)
    if i < length(closestIndex)
        trialbegin = closestIndex(i);
        trialend = closestIndex(i+1);
        off = 0;
        type = values(i);
        newtrl = [trialbegin,trialend,off,type];
        trl = [trl; newtrl];
    else
        trialbegin = closestIndex(i);
        trialend = length(streams);
        off = 0;
        type = values(i);
        newtrl = [trialbegin,trialend,off,type];
        trl = [trl; newtrl];
        trl=cell2table(trl);
        trl=double(table2array(trl));

    end

end

