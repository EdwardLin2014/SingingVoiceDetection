clear all; close all; clc

ToolDirStr = '../../../00_Tools/';
DatabaseDirStr = '../../../03_Database/iKala/Wavfile/';
AudioOutDirStr = '../../../02_Audio/01_SinExtraction/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Addpath for SineModel/UtilFunc/BSS_Eval
addpath(genpath(ToolDirStr));
%% Step 0 - Parmaters Setting
% STFT
Parm.M = 1024;                  % Window Size, 46.44ms
Parm.window = hann(Parm.M);     % Window in Vector Form
Parm.N = 4096;                  % Analysis FFT Size, 185.76ms
Parm.H = 256;                   % Hop Size, 11.61ms
Parm.fs = 22050;                % Sampling Rate, 22.05K Hz
Parm.t = 1;                     % Need All Peaks, in term of Mag Level

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Audio File Name
WavFileDirs = iKalaWavFileNames(DatabaseDirStr);
numMusics = numel(WavFileDirs);
% NSDR, SIR, SAR
IBM_BSS = zeros(numMusics,6);
IBMPeak_BSS = zeros(numMusics,6);
IBMPeakSine_BSS = zeros(numMusics,6);

for t = 1:numMusics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - Import Audio and Create Power Spectrogram
    tic
    % import audio
    [x, fs] = audioread(WavFileDirs{t});
    Voice.x = resample(x(:,2),1,2);
    Song.x = resample(x(:,1),1,2);
    Mix.x = resample( (x(:,1)+x(:,2)), 1, 2);
    %% For Synthesize, constraint the amplitude to either the original max min, or [-1, 1]
    MinAmp = min(Mix.x); if MinAmp<-1; MinAmp = -1; end
    MaxAmp = max(Mix.x); if MaxAmp>1; MaxAmp = 1; end
    % Spectrogram Dimension - Parm.numBins:2049 X Parm.numFrames:2584 = 5,294,616
    [~, Voice.mX, ~, ~, ~] = stft(Voice.x, Parm);
    [~, Song.mX, ~, ~, ~] = stft(Song.x, Parm);
    [~, Mix.mX, Mix.pX, Parm.remain, Parm.numFrames, Parm.numBins] = stft(Mix.x, Parm);
    Mix.mXdB = MagTodB(Mix.mX);
    Parm.mindB = min(min(Mix.mXdB)); 
    Parm.maxdB = max(max(Mix.mXdB));
    if t <= 137
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileDirs{t}(end-14:end), toc);
    else
        fprintf('Import audio - %d:%s - needs %.2f sec\n', t, WavFileDirs{t}(end-15:end), toc);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2 - Create Ideal Binary Mask
    tic    
    Voice.IBM = Voice.mX > Song.mX;
    Song.IBM = Voice.mX <= Song.mX;
    Mix.ploc = peakDetection( Mix.mXdB, Parm );
    
    Voice.IBMPeak = Voice.IBM .* Mix.ploc;
    Song.IBMPeak = Song.IBM .* Mix.ploc;
    fprintf('Create IBM needs %.2f sec\n', toc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3 - iSTFT
    tic
    mV = Voice.IBM .* Mix.mX;
    mS = Song.IBM .* Mix.mX;
    Voice.IBMy = istft(mV, Mix.pX, Parm ); 
    Voice.IBMy = resample(Voice.IBMy,2,1);
    Voice.IBMy = scaleAudio( Voice.IBMy, MinAmp, MaxAmp );
    Song.IBMy = istft(mS, Mix.pX, Parm ); 
    Song.IBMy = resample(Song.IBMy,2,1);
    Song.IBMy = scaleAudio( Song.IBMy, MinAmp, MaxAmp );
    
    mV = Voice.IBMPeak .* Mix.mX;
    mS = Song.IBMPeak .* Mix.mX;
    Voice.IBMPeaky = istft(mV, Mix.pX, Parm ); 
    Voice.IBMPeaky = resample(Voice.IBMPeaky,2,1);
    Voice.IBMPeaky = scaleAudio( Voice.IBMPeaky, MinAmp, MaxAmp );
    Song.IBMPeaky = istft(mS, Mix.pX, Parm ); 
    Song.IBMPeaky = resample(Song.IBMPeaky,2,1);
    Song.IBMPeaky = scaleAudio( Song.IBMPeaky, MinAmp, MaxAmp );
    
    mVdB = Voice.IBMPeak .* Mix.mXdB;
    mVdB = prepareSineSynth( mVdB, Voice.IBMPeak, Parm );
    pV = prepareSineSynth( Mix.pX, Voice.IBMPeak, Parm );
    mSdB = Song.IBMPeak .* Mix.mXdB;
    mSdB = prepareSineSynth( mSdB, Song.IBMPeak, Parm );
    pS = prepareSineSynth( Mix.pX, Song.IBMPeak, Parm );
    Voice.IBMPeakSiney = sineSynth( mVdB, pV, Voice.IBMPeak, Parm );
    Voice.IBMPeakSiney = resample( Voice.IBMPeakSiney,2,1 );
    Voice.IBMPeakSiney = scaleAudio( Voice.IBMPeakSiney, MinAmp, MaxAmp );
    Song.IBMPeakSiney = sineSynth( mSdB, pS, Song.IBMPeak, Parm );
    Song.IBMPeakSiney = resample( Song.IBMPeakSiney,2,1 );
    Song.IBMPeakSiney = scaleAudio( Song.IBMPeakSiney, MinAmp, MaxAmp );
    fprintf('Computing iSTFT need %.2f sec\n', toc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 4 - genSound
    tic
    if t <= 137
        audiowrite([AudioOutDirStr, '/IBMS_iSTFT_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-14:end)], Voice.IBMy, fs );
        audiowrite([AudioOutDirStr, '/IBMS_iSTFT_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-14:end)], Song.IBMy, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_iSTFT_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-14:end)], Voice.IBMPeaky, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_iSTFT_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-14:end)], Song.IBMPeaky, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_AS_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-14:end)], Voice.IBMPeakSiney, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_AS_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-14:end)], Song.IBMPeakSiney, fs );
    else
        audiowrite([AudioOutDirStr, '/IBMS_iSTFT_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-15:end)], Voice.IBMy, fs );
        audiowrite([AudioOutDirStr, '/IBMS_iSTFT_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-15:end)], Song.IBMy, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_iSTFT_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-15:end)], Voice.IBMPeaky, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_iSTFT_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-15:end)], Song.IBMPeaky, fs );    
        audiowrite([AudioOutDirStr, '/IBMPeak_AS_1024_4096_256/',num2str(t),'_Voice_',WavFileDirs{t}(end-15:end)], Voice.IBMPeakSiney, fs );
        audiowrite([AudioOutDirStr, '/IBMPeak_AS_1024_4096_256/',num2str(t),'_Song_',WavFileDirs{t}(end-15:end)], Song.IBMPeakSiney, fs );
    end
    fprintf('genSound needs %.2f sec\n', toc);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 5 - BSS Evaluation
    tic
    trueVoice = gpuArray(x(:,2));
    trueKaraoke = gpuArray(x(:,1));
    trueMixed = gpuArray(x(:,1)+x(:,2));
    
    estimatedVoice = gpuArray(Voice.IBMy);
    estimatedKaraoke = gpuArray(Song.IBMy);
    [SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / norm(estimatedVoice + estimatedKaraoke), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    [NSDR, ~, ~] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    NSDR = SDR - NSDR;
    
    IBM_BSS(t,1) = gather(NSDR(1));
    IBM_BSS(t,2) = gather(NSDR(2));
    IBM_BSS(t,3) = gather(SIR(1));
    IBM_BSS(t,4) = gather(SIR(2));
    IBM_BSS(t,5) = gather(SAR(1));
    IBM_BSS(t,6) = gather(SAR(2));
    
    fprintf('IBM NSDR:%.4f, %.4f\n', NSDR(1), NSDR(2));
    fprintf('IBM SIR:%.4f, %.4f\n', SIR(1), SIR(2));
    fprintf('IBM SAR:%.4f, %.4f\n', SAR(1), SAR(2));
    fprintf('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, toc);
    
    estimatedVoice = gpuArray(Voice.IBMPeaky);
    estimatedKaraoke = gpuArray(Song.IBMPeaky);
    [SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / norm(estimatedVoice + estimatedKaraoke), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    [NSDR, ~, ~] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    NSDR = SDR - NSDR;
    
    IBMPeak_BSS(t,1) = gather(NSDR(1));
    IBMPeak_BSS(t,2) = gather(NSDR(2));
    IBMPeak_BSS(t,3) = gather(SIR(1));
    IBMPeak_BSS(t,4) = gather(SIR(2));
    IBMPeak_BSS(t,5) = gather(SAR(1));
    IBMPeak_BSS(t,6) = gather(SAR(2));
        
    fprintf('IdealPeak NSDR:%.4f, %.4f\n', NSDR(1), NSDR(2));
    fprintf('IdealPeak SIR:%.4f, %.4f\n', SIR(1), SIR(2));
    fprintf('IdealPeak SAR:%.4f, %.4f\n', SAR(1), SAR(2));
    fprintf('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, toc);
    
    estimatedVoice = gpuArray(Voice.IBMPeakSiney);
    estimatedKaraoke = gpuArray(Song.IBMPeakSiney);
    [SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / norm(estimatedVoice + estimatedKaraoke), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    [NSDR, ~, ~] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    NSDR = SDR - NSDR;
    
    IBMPeakSine_BSS(t,1) = gather(NSDR(1));
    IBMPeakSine_BSS(t,2) = gather(NSDR(2));
    IBMPeakSine_BSS(t,3) = gather(SIR(1));
    IBMPeakSine_BSS(t,4) = gather(SIR(2));
    IBMPeakSine_BSS(t,5) = gather(SAR(1));
    IBMPeakSine_BSS(t,6) = gather(SAR(2));
        
    fprintf('IdealPeakSine NSDR:%.4f, %.4f\n', NSDR(1), NSDR(2));
    fprintf('IdealPeakSine SIR:%.4f, %.4f\n', SIR(1), SIR(2));
    fprintf('IdealPeakSine SAR:%.4f, %.4f\n', SAR(1), SAR(2));
    fprintf('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, toc);
    fprintf('=================================================\n');
end