clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Parmaters Setting
ToolDirStr = '../../00_Tools/';
WavDirStr = '../../03_Database/iKala/Wavfile/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Addpath for SineModel/UtilFunc/BSS_Eval
addpath(genpath(ToolDirStr));
%% Step 0 - Parmaters Setting
% STFT
Parm.M = 1024;                  % Window Size, 23.22ms
Parm.window = hann(Parm.M);     % Window in Vector Form
Parm.N = 1024;                  % Analysis FFT Size, 23.22ms
Parm.H = 512;                   % Hop Size, 11.61ms
Parm.fs = 44100;                % Sampling Rate, 44.10K Hz
Parm.numBins = 513;
Parm.numFrames = 2584;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Audio File Name
WavFileNames = iKalaWavFileNames(WavDirStr);
numMusics = numel(WavFileNames);
%% Step 0 - Parmaters Setting
% STFT
Parm.M = 1024;                  % Window Size, 46.44ms
Parm.window = hann(Parm.M);     % Window in Vector Form
Parm.N = 1024;                  % Analysis FFT Size, 46.44ms
Parm.H = 512;                   % Hop Size, 23.22ms
Parm.fs = 44100;                % Sampling Rate, 44.10K Hz
Parm.numBins = 513;
Parm.numFrames = 2584;
Parm.FreqCrit = 1:25;
Parm.numTFBin = 5;
trainMusic = [3,4,5,6,7,8,9,10,11,13,14,16,17,20,21,23,27,29,30,33,34,35,36,37,38,39,41,44,46,50,52,53,54,55,56,57,59,60,61,63,64,65,66,67,70,71,73,77,78,82,84,85,86,88,91,92,95,96,97,98,102,103,108,109,114,115,116,117,119,123,124,126,127,128,130,133,141,142,144,145,146,147,149,150,151,153,154,155,156,158,159,160,162,163,164,165,169,170,172,173,174,176,180,184,185,186,187,188,189,190,191,193,194,196,197,199,200,201,203,207,208,209,210,211,215,216,217,219,221,224,225,226,227,230,231,232,233,234,235,236,237,238,239,240,241,242,246,248,249,250,251,252];
valMusic = [1,22,28,32,42,48,49,58,62,72,74,80,83,89,90,93,101,106,120,121,122,125,129,131,136,140,143,148,157,161,168,178,181,182,183,192,195,198,205,206,212,213,214,220,222,229,243,244,245,247];
testMusic = [2,12,15,18,19,24,25,26,31,40,43,45,47,51,68,69,75,76,79,81,87,94,99,100,104,105,107,110,111,112,113,118,132,134,135,137,138,139,152,166,167,171,175,177,179,202,204,218,223,228];
BSS = zeros(numMusics,3);
for t = 1:numMusics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - Import Audio and Create Power Spectrogram
    tic
    [x, fs] = audioread(WavFileNames{t});
    Mix.x = x(:,1)+x(:,2);
    Voice.x = x(:,2);
    Song.x = x(:,1);
    % Spectrogram Dimension - Parm.numBins:513 X Parm.numFrames:2584 = 1,325,592
    [~, Voice.mX, ~, ~, ~] = stft(Voice.x, Parm);
    [~, Song.mX, ~, ~, ~] = stft(Song.x, Parm);
    [~, Mix.mX, Mix.pX, Parm.remain, Parm.numFrames, Parm.numBins] = stft(Mix.x, Parm);
    if t <= 137
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2 - Create Ideal Binary Mask
    tic    
    Voice.IBM = Voice.mX > Song.mX;
    %Song.IBM = Voice.mX <= Song.mX;
    
    Voice.frameMask = Voice.IBM;
    frameStat = sum(Voice.IBM(Parm.FreqCrit,:));
    Voice.frameMask(:,frameStat>Parm.numTFBin) = 1;
    Voice.frameMask(:,frameStat<=Parm.numTFBin) = 0;
    fprintf('Create IBM needs %.2f sec\n', toc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3 - iSTFT
    tic
    mV = Voice.frameMask.*Mix.mX;
    %mS = Song.IBM.*Mix.mX;
    Voice.y = istft(mV, Mix.pX, Parm );
    %Song.y = istft(mS, Mix.pX, Parm );
    fprintf('%d: Computing iSTFT needs %.2f sec\n', t, toc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3 - BSS Evaluation
    tic
    trueVoice = gpuArray(x(:,2));
    estimatedVoice = gpuArray(Voice.y);
    [SDR, SIR, SAR] = bss_eval_sdr(estimatedVoice', trueVoice');
    BSS(t,1) = gather(SDR);
    BSS(t,2) = gather(SIR);
    BSS(t,3) = gather(SAR);
    fprintf('SDR:%.4f\n', SDR);
    fprintf('SIR:%.4f\n', SIR);
    fprintf('SAR:%.4f\n', SAR);
    fprintf('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, toc);
end

fprintf('All: %.4f\n', mean(BSS(:,1)) );
fprintf('Train: %.4f\n', mean(BSS(trainMusic,1)) );
fprintf('Val: %.4f\n', mean(BSS(valMusic,1)) );
fprintf('Test: %.4f\n', mean(BSS(testMusic,1)) );
fprintf('Verse: %.4f\n', mean(BSS(1:137,1)) );
fprintf('Chorus: %.4f\n', mean(BSS(138:end,1)) );
