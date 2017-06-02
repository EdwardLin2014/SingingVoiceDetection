function [ PitchMask ] = iKalaPitchMask( PitchFileNames,numMusics )
%IKALAPITCHMASK Summary of this function goes here
%   Detailed explanation goes here

PitchMask = zeros(numMusics,937);
delimiter = ' ';
formatSpec = '%f%[^\n\r]';
for n = 1:numMusics
    fileID = fopen(PitchFileNames{n},'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
    fclose(fileID);
    PitchMask(n,:) = dataArray{1,1}';
end
PitchMask(PitchMask>1) = 1;

end

