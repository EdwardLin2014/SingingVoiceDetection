clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Parmaters Setting
ToolDirStr = '../../00_Tools/';
WavDirStr = '../../03_Database/iKala/Wavfile/';
PitchDirStr = '../../03_Database/iKala/PitchLabel/';
H5FileDirStr = '../../03_Database/iKala/SM_HDF5/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Addpath for SineModel/UtilFunc/BSS_Eval
addpath(genpath(ToolDirStr));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Audio File Name
WavFileNames = iKalaWavFileNames(WavDirStr);
numMusics = numel(WavFileNames);

%% Step 0 - Parmaters Setting
% STFT
Parm.M = 1024;                  % Window Size, 46.44ms
Parm.window = hann(Parm.M);     % Window in Vector Form
Parm.N = 1024;                  % Analysis FFT Size, 46.44ms
Parm.H = 315;                   % Hop Size, 14.29ms
Parm.fs = 22050;                % Sampling Rate, 22.05K Hz
Parm.numBins = 372;
Parm.numFrames = 2101;           
Parm.numFTBins = 42780;         % 372(0~8.00kHz) * 115 (~1.5790 sec)
% Input
numIFrames = 115;               % Each input for CNN is ~1.6429 sec
hNumIFrames = floor(numIFrames/2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Pitch/Voice Label and resample
tic
PitchFileNames = iKalaPitchLabelFileNames(PitchDirStr);
oldPitchMask = iKalaPitchMask(PitchFileNames,numMusics);
PitchMask = zeros(numMusics,Parm.numFrames);
for t = 1:numMusics
    PitchMask(t,:) = resample(oldPitchMask(t,:), Parm.numFrames, 937);
    PitchMask(t,PitchMask(t,:)>0.5) = 0.98;
    PitchMask(t,PitchMask(t,:)<=0.5) = 0.2;
end
PitchMask = [zeros(numMusics,hNumIFrames),PitchMask,zeros(numMusics,hNumIFrames)];
fprintf('Obtain Pitch/Voice Label and resample needs %.2f sec\n', toc);

trainMusic = [3,4,5,6,7,8,9,10,11,13,14,16,17,20,21,23,27,29,30,33,34,35,36,37,38,39,41,44,46,50,52,53,54,55,56,57,59,60,61,63,64,65,66,67,70,71,73,77,78,82,84,85,86,88,91,92,95,96,97,98,102,103,108,109,114,115,116,117,119,123,124,126,127,128,130,133,141,142,144,145,146,147,149,150,151,153,154,155,156,158,159,160,162,163,164,165,169,170,172,173,174,176,180,184,185,186,187,188,189,190,191,193,194,196,197,199,200,201,203,207,208,209,210,211,215,216,217,219,221,224,225,226,227,230,231,232,233,234,235,236,237,238,239,240,241,242,246,248,249,250,251,252];
valMusic = [1,22,28,32,42,48,49,58,62,72,74,80,83,89,90,93,101,106,120,121,122,125,129,131,136,140,143,148,157,161,168,178,181,182,183,192,195,198,205,206,212,213,214,220,222,229,243,244,245,247];
testMusic = [2,12,15,18,19,24,25,26,31,40,43,45,47,51,68,69,75,76,79,81,87,94,99,100,104,105,107,110,111,112,113,118,132,134,135,137,138,139,152,166,167,171,175,177,179,202,204,218,223,228];

trainIdx = 1;
valIdx = 1;
testIdx = 1;
rows = Parm.numFTBins;
H5FileName = [H5FileDirStr,'Spec_1024_1024_315.h5'];
h5create(H5FileName,'/train',[rows,Inf],'Datatype','double','ChunkSize',[rows,1]);
h5create(H5FileName,'/trainLabel',[1,Inf],'Datatype','double','ChunkSize',[1,1]);
h5create(H5FileName,'/valid',[rows,Inf],'Datatype','double','ChunkSize',[rows,1]);
h5create(H5FileName,'/validLabel',[1,Inf],'Datatype','double','ChunkSize',[1,1]);
h5create(H5FileName,'/test',[rows,Inf],'Datatype','double','ChunkSize',[rows,1]);
h5create(H5FileName,'/testLabel',[1,Inf],'Datatype','double','ChunkSize',[1,1]);

hopNumFrames = 11;                                  % 0.4989 ms
numInputPerSong = Parm.numFrames/hopNumFrames;      % 191 CNN input instances per one song
data = zeros(Parm.numFTBins,numInputPerSong);
label = zeros(1,numInputPerSong);
for t = 1:numMusics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - Import Audio and Create Power Spectrogram
    tic
    [x, fs] = audioread(WavFileNames{t});
    Mix.x = resample( (x(:,1)+x(:,2)), 1, 2);
    % Spectrogram Dimension - Parm.numBins:372 X Parm.numFrames:2101 = 781,572
    [~, mX, ~, ~, ~, ~] = stft(Mix.x, Parm);
    mX = mX(1:Parm.numBins,:);
    mX = [zeros(Parm.numBins,hNumIFrames),mX,zeros(Parm.numBins,hNumIFrames)];
    mX = ScaleTo01(log(1 + mX));
    if t <= 137
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2 - Write data to hdf5 file for tensorflow
    tic
    % Label is an indicator variable, 1: Voice, 0: Song
    % Python hdf5 will autotranspose
    for i = 1:numInputPerSong
        startIdx = (i-1)*hopNumFrames+1;
        endIdx = startIdx + numIFrames-1;
        labelIdx = startIdx + hNumIFrames;
        data(:,i) = reshape(mX(:,startIdx:endIdx),Parm.numFTBins,1);
        label(i) = PitchMask(t,labelIdx);
    end
    count = size(data);
    labelsize = size(label);
    if ismember(t,trainMusic)
        start = [1, (trainIdx-1)*numInputPerSong+1];
        h5write(H5FileName,'/train',data,start,count);
        h5write(H5FileName,'/trainLabel',label,start,labelsize);
        trainIdx = trainIdx + 1;
    end
    if ismember(t,valMusic)
        start = [1, (valIdx-1)*numInputPerSong+1];
        h5write(H5FileName,'/valid',data,start,count);
        h5write(H5FileName,'/validLabel',label,start,labelsize);
        valIdx = valIdx + 1;
    end
    if ismember(t,testMusic)
        start = [1, (testIdx-1)*numInputPerSong+1];
        h5write(H5FileName,'/test',data,start,count);
        h5write(H5FileName,'/testLabel',label,start,labelsize);
        testIdx = testIdx + 1;
    end
    if t <= 137
        fprintf('Write hdf5 - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Write hdf5 - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    fprintf('=================================================\n');
end
