clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Parmaters Setting
ToolDirStr = '../../00_Tools/';
WavDirStr = '../../03_Database/iKala/Wavfile/';
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
rows = Parm.numFTBins;
H5FileName = [H5FileDirStr,'Test_1Song_1024_1024_315.h5'];
h5create(H5FileName,'/mX',[rows,Inf],'Datatype','double','ChunkSize',[rows,1]);
h5create(H5FileName,'/mXImg',[Parm.numBins,Parm.numFrames],'Datatype','double','ChunkSize',[Parm.numBins,1]);

data = zeros(Parm.numFTBins,Parm.numFrames);
for t = 100         % Choose one song from 1 to 252
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - Import Audio and Create Power Spectrogram
    tic
    [x, fs] = audioread(WavFileNames{t});
    Mix.x = resample( (x(:,1)+x(:,2)), 1, 2);
    % Spectrogram Dimension - Parm.numBins:372 X Parm.numFrames:2101 = 781,572
    [~, mX, ~, ~, ~, ~] = stft(Mix.x, Parm);
    mXImg = mX(1:372,:);
    mX = [zeros(372,57),mXImg,zeros(372,57)];
    mX = ScaleTo01(log(1 + mX));
    mXImg = ScaleTo01(log(1 + mXImg));
    if t <= 137
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2 - Write data to hdf5 file for tensorflow
    tic
    % Python hdf5 will autotranspose
    for i = 1:Parm.numFrames
        startIdx = i;
        endIdx = startIdx + 114;
        data(:,i) = reshape(mX(:,startIdx:endIdx),Parm.numFTBins,1);
    end
    h5write(H5FileName,'/mX',data,[1,1],size(data));
    h5write(H5FileName,'/mXImg',mXImg,[1,1],size(mXImg));
    if t <= 137
        fprintf('Write hdf5 - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Write hdf5 - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    fprintf('=================================================\n');
end
