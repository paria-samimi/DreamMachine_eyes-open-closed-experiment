
function [trl, event] = ft_trialfun_trigtraum_amp(cfg)

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
streams = load_xdf('Fynn.xdf');     %%%%%%%%%%%%%%%%%%%%% change%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% now let's add to the event struct
event = [];
C = cell(1,11)'; % here adjust length so it will get it manually      %%%%%%%%%%%%%%%%%%%%% change%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C(:) = {'Markers'};
T = cell2table(C,'VariableNames',{'type'});
event = table2struct(T);
event=event';
values = num2cell(streams{1,1}.time_series(2,:)');
[event.value] = values{:};
% find out the minimum indicies
ts_event = streams{1,1}.time_stamps;
ts_data = streams{1,2}.time_stamps';
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
        trialend = length(streams{1,2}.time_stamps);
        off = 0;
        type = values(i);
        newtrl = [trialbegin,trialend,off,type];
        trl = [trl; newtrl];
        trl=cell2table(trl);
        trl=double(table2array(trl));

    end

end

