function [ Hz ] = FBinToHz( FBin, Parm )

%       FBin: [0 ... N-1]
%    Parm.fs: sampling rate
%     Parm.N: DFT Size
MaxFBin = floor(Parm.N/2)+1;

if FBin > MaxFBin
    FBin = MaxFBin;
elseif FBin < 1
    FBin = 1;
end

Hz = Parm.fs/Parm.N * (FBin-1);

end

