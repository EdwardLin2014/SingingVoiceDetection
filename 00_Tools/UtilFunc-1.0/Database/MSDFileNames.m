function [ MixFiles, SourceFiles ] = MSDFileNames( DatabaseDirStr )


%% Function Body
Mix.TestDirStr = fullfile(DatabaseDirStr,'Mixtures','Test');
Mix.TestDir = dir(Mix.TestDirStr);

Mix.DevDirStr = fullfile(fullfile(DatabaseDirStr,'Mixtures','Dev'));
Mix.DevDir = dir(Mix.DevDirStr);

Sources.TestDirStr = fullfile(fullfile(DatabaseDirStr,'Sources','Test'));
Sources.TestDir = dir(Sources.TestDirStr);

Sources.DevDirStr = fullfile(fullfile(DatabaseDirStr,'Sources','Dev'));
Sources.DevDir = dir(Sources.DevDirStr);

MixFiles = cell(100,1);
SourceFiles = cell(100,4);
SourceType = [{'bass.wav'}; {'drums.wav'};{'other.wav'};{'vocals.wav'}];

for i = 1:50
    MixFiles{i} = fullfile(Mix.TestDirStr, Mix.TestDir(i+3).name,'mixture.wav');
    MixFiles{i+50} = fullfile(Mix.DevDirStr, Mix.DevDir(i+3).name,'mixture.wav');
    
    for j = 1:4
        SourceFiles{i,j} = fullfile(Sources.TestDirStr,Sources.TestDir(i+3).name,SourceType{j});
        SourceFiles{i+50,j} = fullfile(Sources.DevDirStr,Sources.DevDir(i+3).name,SourceType{j});
    end
end

end

