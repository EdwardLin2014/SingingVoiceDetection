function [ FileDirs ] = iKalaPitchLabelFileNames(DatabaseDirStr)
%% Output 
% FileDirs = cell(252,1);
% 137 Verse = cell(1:137,1);
% 115 Chorus = cell(138:252,1);

%% Function Body
iKala = dir(DatabaseDirStr);
numFiles = 0;
startIdx = 0;
for i = 1:numel(iKala)
    filename = iKala(i).name;
    if numel(filename) > 3
        if strcmp(filename(end-2:end), '.pv')
            numFiles = numFiles + 1;
        else
            startIdx = i;
        end
    else
        startIdx = i;
    end
end

FileDirs = cell(numFiles,1);
l = 0;
for i = startIdx+1:numFiles+startIdx
    wavname = iKala(i).name;
    if strcmp(wavname(end-8:end), '_verse.pv')
        l = l + 1;
        FileDirs{l} = [DatabaseDirStr, wavname];
    end
end
for i = startIdx+1:numFiles+startIdx
    wavname = iKala(i).name;
    if strcmp(wavname(end-9:end), '_chorus.pv')
        l = l + 1;
        FileDirs{l} = [DatabaseDirStr, wavname];
    end
end

end

