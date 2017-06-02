function [ ScaledX ] = ScaleTo01( X )

[M,~] = size(X);

OldMin = min(X);
OldMax = max(X);

ScaledFactor = 1./(OldMax-OldMin);
ScaledX = ScaledFactor.*(X - repmat(OldMin,M,1));

end

