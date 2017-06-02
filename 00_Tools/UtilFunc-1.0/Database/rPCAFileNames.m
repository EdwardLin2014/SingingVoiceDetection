function [ SongFileDirs,VoiceFileDirs ] = rPCAFileNames(DatabaseDirStr)
%% Output 
% SongFileDirs = cell(252,1);
% VoiceFileDirs = cell(252,1);
% 137 Verse = cell(1:137,1);
% 115 Chorus = cell(138:252,1);

%% Function Body
rPCA = dir(DatabaseDirStr);
numFiles = 0;
startIdx = 0;
for i = 1:numel(rPCA)
    filename = rPCA(i).name;
    if numel(filename) > 3
        if strcmp(filename(end-3:end), '.wav')
            numFiles = numFiles + 1;
        else
            startIdx = i;
        end
    else
        startIdx = i;
    end
end

SongFileDirs = cell(numFiles/2,1);
VoiceFileDirs = cell(numFiles/2,1);
m = 0;
for i = startIdx+1:numFiles+startIdx
    wavname = rPCA(i).name;
    if strcmp(wavname(end-9:end), '_verse.wav')
        if strcmp(wavname(end-19:end-16), 'Song')
            switch numel(wavname)
                case 22
                    m = 1;
                case 23
                    m = 2;
                case 24
                    m = 3;
            end
            idx = str2double(wavname(1:m));
            SongFileDirs{idx} = [DatabaseDirStr, wavname];
        end
        if strcmp(wavname(end-20:end-16), 'Voice')
            switch numel(wavname)
                case 23
                    m = 1;
                case 24
                    m = 2;
                case 25
                    m = 3;
            end
            idx = str2double(wavname(1:m));
            VoiceFileDirs{idx} = [DatabaseDirStr, wavname];
        end
    end
    
    if strcmp(wavname(end-10:end), '_chorus.wav')
        if strcmp(wavname(end-20:end-17), 'Song')
            switch numel(wavname)
                case 23
                    m = 1;
                case 24
                    m = 2;
                case 25
                    m = 3;
            end
            idx = str2double(wavname(1:m));
            SongFileDirs{idx} = [DatabaseDirStr, wavname];
        end
        if strcmp(wavname(end-21:end-17), 'Voice')
            switch numel(wavname)
                case 24
                    m = 1;
                case 25
                    m = 2;
                case 26
                    m = 3;
            end
            idx = str2double(wavname(1:m));
            VoiceFileDirs{idx} = [DatabaseDirStr, wavname];
        end
    end
end

end

