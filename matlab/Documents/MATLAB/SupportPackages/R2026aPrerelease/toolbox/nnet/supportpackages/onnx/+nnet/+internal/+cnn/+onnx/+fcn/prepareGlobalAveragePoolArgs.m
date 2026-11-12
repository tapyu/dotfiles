function [poolsize, dataFormat, numDimsX] = prepareGlobalAveragePoolArgs(X, numDimsX)
% Prepares arguments for implementing the ONNX GlobalAveragePool operator

%   Copyright 2020 The MathWorks, Inc.    
poolsize    = size(X, 1:numDimsX-2);                   % The spatial dimensions
dataFormat  = [repmat('S', [1 numDimsX-2]) 'CB'];
end
