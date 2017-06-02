function [SDR, SIR, SAR] = bss_eval_sdr(se,s)

%%% Performance criteria %%%
[s_true,e_spat,e_interf,e_artif] = bss_decomp_mtifilt(se(1,:),s,512);
[SDR, SIR, SAR] = bss_source_crit(s_true,e_spat,e_interf,e_artif);

return;

function [s_true,e_spat,e_interf,e_artif] = bss_decomp_mtifilt(se,s,flen)

%%% Decomposition %%%
% True source image
s_true = [s,zeros(1,flen-1)];
% Spatial (or filtering) distortion
e_spat = project(se,s,flen) - s_true;
% Interference is always zeros
e_interf = zeros(size(e_spat));
% Artifacts
e_artif = [se,zeros(1,flen-1)] - s_true - e_spat;

return;

function sproj = project(se,s,flen)

% SPROJ Least-squares projection of each channel of se on the subspace
% spanned by delayed versions of the channels of s, with delays between 0
% and flen-1
[~,nsampl,~] = size(s);

%%% Computing coefficients of least squares problem via FFT %%%
% Zero padding and FFT of input data
s = [s,zeros(1,flen-1)];
se = [se,zeros(1,flen-1)];
fftlen = 2^nextpow2(nsampl+flen-1);
sf = fft(s,fftlen,2);
sef = fft(se,fftlen,2);

% Inner products between delayed versions of s
ssf = sf(1,:).*conj(sf(1,:));
ssf = real(ifft(ssf));
ss = toeplitz(ssf([1 fftlen:-1:fftlen-flen+2]),ssf(1:flen));
G = ss.';

% Inner products between se and delayed versions of s
ssef = sf(1,:).*conj(sef(1,:));
ssef = real(ifft(ssef,[],2));
D = ssef(:,[1 fftlen:-1:fftlen-flen+2]).';

%%% Computing projection %%%
% Distortion filters
C = G\D;
% Filtering
sproj = fftfilt(C(:,1,1).',s);

return;

function [SDR,SIR,SAR] = bss_source_crit(s_true,e_spat,e_interf,e_artif)

%%% Energy ratios %%%
s_filt = s_true + e_spat;
numerator = sum(sum(s_filt.^2));
% SDR
SDR = 10*log10(numerator/sum(sum((e_interf+e_artif).^2)));
% SIR
SIR = 10*log10(numerator/sum(sum(e_interf.^2)));
% SAR
SAR = 10*log10(sum(sum((s_filt+e_interf).^2))/sum(sum(e_artif.^2)));

return;