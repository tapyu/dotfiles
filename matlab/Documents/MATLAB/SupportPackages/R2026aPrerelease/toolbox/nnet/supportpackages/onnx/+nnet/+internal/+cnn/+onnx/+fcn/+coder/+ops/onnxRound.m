function [Y, numDimsX] = onnxRound(X_, numDimsX_)
% Implements the ONNX Round operator, which rounds halves to the nearest
% even integer. 
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

Y = round(X);
halves = mod(X,.5) == 0 & mod(X,1) ~= 0;
Y(halves) = 2 * round(X(halves)/2);
end
