function [poolsize, dataFormat, numDimsX] = prepareGlobalMaxPoolArgs(X, numDimsX)
% Prepares arguments for implementing the ONNX GlobalMaxPool operator

%   Copyright 2021 The MathWorks, Inc.    
poolsize    = size(X, 1:numDimsX-2);                   % The spatial dimensions
dataFormat  = [repmat('S', [1 numDimsX-2]) 'CB'];
end
