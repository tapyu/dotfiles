function [Y, numDimsX] = onnxHardSigmoid(X_, alpha_, beta_, numDimsX_)
% Implements the ONNX HardSigmoid operator
%#codegen

% Copyright 2024-2026 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
alpha      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(alpha_);
beta       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(beta_);
numDimsX   = coder.const(nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_));

Y = max(0, min(1, alpha * X + beta));
end
