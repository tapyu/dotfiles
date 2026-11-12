function [Y, numDimsY, Mask, numDimsMask] = onnxDropout7(X_, ratio_, Training_, numDimsX_)
% Implements the ONNX Dropout-7 operator. 
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
ratio      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ratio_);
Training   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Training_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

if Training
    Scale = 1/(1 - ratio);
    Mask = rand(size(X)) >= ratio;
    Y = Scale .* X .* Mask;
else
    Y = X;
    Mask = ones(size(X));
end

numDimsY = numDimsX;
numDimsMask = numDimsX;
end
