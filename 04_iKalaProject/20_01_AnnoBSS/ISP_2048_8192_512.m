clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Parmaters Setting
ToolDirStr = '../../00_Tools/';
WavDirStr = '../../03_Database/iKala/Wavfile/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Addpath for SineModel/UtilFunc/BSS_Eval
addpath(genpath(ToolDirStr));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Audio File Name
WavFileNames = iKalaWavFileNames(WavDirStr);
numMusics = numel(WavFileNames);
%% Step 0 - Parmaters Setting
% STFT
Parm.M = 2048;                  % Window Size, 46.44ms
Parm.window = hann(Parm.M);     % Window in Vector Form
Parm.N = 8192;                  % Analysis FFT Size, 185.76ms
Parm.H = 512;                   % Hop Size, 11.61ms
Parm.fs = 44100;                % Sampling Rate, 44.10K Hz
Parm.t = 42;                    % Dicard Peaks below Mag level 42
Parm.FreqCrit = 1:4097;
trainMusic = [3,4,5,6,7,8,9,10,11,13,14,16,17,20,21,23,27,29,30,33,34,35,36,37,38,39,41,44,46,50,52,53,54,55,56,57,59,60,61,63,64,65,66,67,70,71,73,77,78,82,84,85,86,88,91,92,95,96,97,98,102,103,108,109,114,115,116,117,119,123,124,126,127,128,130,133,141,142,144,145,146,147,149,150,151,153,154,155,156,158,159,160,162,163,164,165,169,170,172,173,174,176,180,184,185,186,187,188,189,190,191,193,194,196,197,199,200,201,203,207,208,209,210,211,215,216,217,219,221,224,225,226,227,230,231,232,233,234,235,236,237,238,239,240,241,242,246,248,249,250,251,252];
valMusic = [1,22,28,32,42,48,49,58,62,72,74,80,83,89,90,93,101,106,120,121,122,125,129,131,136,140,143,148,157,161,168,178,181,182,183,192,195,198,205,206,212,213,214,220,222,229,243,244,245,247];
testMusic = [2,12,15,18,19,24,25,26,31,40,43,45,47,51,68,69,75,76,79,81,87,94,99,100,104,105,107,110,111,112,113,118,132,134,135,137,138,139,152,166,167,171,175,177,179,202,204,218,223,228];
BSS = zeros(numMusics,30);
for t = 1:numMusics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - Import Audio and Create Power Spectrogram
    tic
    [x, fs] = audioread(WavFileNames{t});
    Mix.x = x(:,1)+x(:,2);
    Voice.x = x(:,2);
    Song.x = x(:,1);
    % Spectrogram Dimension - Parm.numBins:4097 X Parm.numFrames:2584 = 10,586,648
    [~, Voice.mX, ~, ~, ~] = stft(Voice.x, Parm);
    [~, Song.mX, ~, ~, ~] = stft(Song.x, Parm);
    [~, Mix.mX, Mix.pX, Parm.remain, Parm.numFrames, Parm.numBins] = stft(Mix.x, Parm);
    Mix.mXdB = MagTodB(Mix.mX);
    Parm.mindB = min(min(Mix.mXdB));
    Parm.maxdB = max(max(Mix.mXdB));
    if t <= 137
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-14:end), toc);
    else
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileNames{t}(end-15:end), toc);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2 - Create Ideal Spectral Peaks
    tic
    Voice.IBM = Voice.mX > Song.mX;
    Mix.ploc = peakDetection( Mix.mXdB, Parm );
    Voice.IBMPeak = Voice.IBM .* Mix.ploc;
    fprintf('Create Ideal Spectral Peaks needs %.2f sec\n', toc);
    
    for numPeaks = 0:40
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 3 - Create frameMask
        tic
        Voice.frameMask = Voice.IBMPeak;
        frameStat = sum(Voice.IBMPeak(Parm.FreqCrit,:));
        NonVoiceIdx = frameStat<=numPeaks;
        Voice.frameMask(:,NonVoiceIdx) = 0;
        fprintf('%d,%d: - Create frameMask needs %.2f sec\n', t, numPeaks, toc);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 4 - Additive Synthesis
        tic
        mV = Voice.frameMask.*Mix.mX;
        mVdB = MagTodB(mV);
        mVdB = prepareSineSynth( mVdB, Voice.frameMask, Parm );
        pV = prepareSineSynth( Mix.pX, Voice.frameMask, Parm );
        Voice.y = sineSynth( mVdB, pV, Voice.frameMask, Parm );
        fprintf('%d,%d: Computing Additive Synthesis needs %.2f sec\n', t, numPeaks, toc);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 5 - BSS Evaluation
        tic
        trueVoice = gpuArray(x(:,2));
        estimatedVoice = gpuArray(Voice.y);
        [SDR, SIR, SAR] = bss_eval_1source(estimatedVoice', trueVoice');
        BSS(t,(3*(numPeaks+1))-2) = gather(SDR);
        BSS(t,(3*(numPeaks+1))-1) = gather(SIR);
        BSS(t,3*(numPeaks+1)) = gather(SAR);
        fprintf('SDR:%.4f\n', SDR);
        fprintf('SIR:%.4f\n', SIR);
        fprintf('SAR:%.4f\n', SAR);
        fprintf('Computing %d,%d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, numPeaks, toc);
    end
end
