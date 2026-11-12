function str = uniqueCoderName(desiredString)
% Generate a new name within the maximum supported MATLAB 
% identifier limit defined by 'namelengthmax'. 
%   Copyright 2024 The MathWorks, Inc.    

persistent coderNameNumber
if isempty(coderNameNumber)
    coderNameNumber = 1000;
else
    coderNameNumber = coderNameNumber + 1;
end
suff = num2str(coderNameNumber);
desiredString = char(desiredString);
str = [desiredString(1:min(numel(desiredString), namelengthmax - numel(suff))) suff];
end
