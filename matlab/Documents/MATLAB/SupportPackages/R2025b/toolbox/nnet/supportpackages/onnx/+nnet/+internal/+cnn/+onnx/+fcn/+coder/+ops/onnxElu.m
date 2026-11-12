function [X, numDimsX] = onnxElu(X_, alpha_, numDimsX_)
% Implements the ONNX Elu operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
alpha      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(alpha_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

X(X<=0) = alpha * (exp(X(X<=0))-1);
end
