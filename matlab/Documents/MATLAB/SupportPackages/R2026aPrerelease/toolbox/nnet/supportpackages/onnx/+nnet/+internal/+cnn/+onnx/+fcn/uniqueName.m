function str = uniqueName(desiredString)
% Generate a new name within the maximum supported MATLAB 
% identifier limit defined by 'namelengthmax'. 
%   Copyright 2020-2024 The MathWorks, Inc.    

persistent nameNumber
if isempty(nameNumber)
    nameNumber = 1000;
else
    nameNumber = nameNumber + 1;
end
suff = num2str(nameNumber);
desiredString = char(desiredString);
str = [desiredString(1:min(numel(desiredString), namelengthmax - numel(suff))) suff];
end
