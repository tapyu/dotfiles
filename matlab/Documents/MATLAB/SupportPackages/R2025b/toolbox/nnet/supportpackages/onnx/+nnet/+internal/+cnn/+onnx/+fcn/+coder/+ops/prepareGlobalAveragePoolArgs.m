function [poolsize, dataFormat, numDimsX] = prepareGlobalAveragePoolArgs(X_, numDimsX_)
% Prepares arguments for implementing the ONNX GlobalAveragePool operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);   
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_); 

poolsize    = size(X, 1:numDimsX-2);                   % The spatial dimensions
dataFormat  = [repmat('S', [1 numDimsX-2]) 'CB'];
end
