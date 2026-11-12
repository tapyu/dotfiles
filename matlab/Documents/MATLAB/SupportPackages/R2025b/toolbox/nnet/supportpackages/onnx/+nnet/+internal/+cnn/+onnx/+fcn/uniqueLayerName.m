function str = uniqueLayerName(desiredString)
% Generate a new name within the maximum supported MATLAB 
% identifier limit defined by 'namelengthmax'. 
%   Copyright 2024 The MathWorks, Inc.    

persistent layerNameNumber
if isempty(layerNameNumber)
    layerNameNumber = 1000;
else
    layerNameNumber = layerNameNumber + 1;
end
suff = num2str(layerNameNumber);
desiredString = char(desiredString);
str = [desiredString(1:min(numel(desiredString), namelengthmax - numel(suff))) suff];
end
