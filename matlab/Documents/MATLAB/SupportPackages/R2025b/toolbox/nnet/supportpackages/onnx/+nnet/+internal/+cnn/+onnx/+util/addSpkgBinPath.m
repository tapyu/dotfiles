%Function to add "C:\ProgramData\MATLAB\SupportPackages\R2021a\bin\win64" to
%System Path

%   Copyright 2021-2024 The MathWorks, Inc.    

function addSpkgBinPath
    % Only add the spkg bin path when the function is called from a MATLAB
    % session
    mlock
    persistent pathSet
    if isempty(pathSet) && ~isdeployed
        pathSet = true;
        spPath = matlabshared.supportpkg.getSupportPackageRoot;
        if ~isempty(spPath)
            archStr = computer('arch');
            dllPath = fullfile(spPath, 'bin', archStr);
            if strcmp(archStr, 'win64')
                sysPath = getenv('PATH');
                setenv('PATH',[sysPath pathsep dllPath]);
            end
        end
    end
end