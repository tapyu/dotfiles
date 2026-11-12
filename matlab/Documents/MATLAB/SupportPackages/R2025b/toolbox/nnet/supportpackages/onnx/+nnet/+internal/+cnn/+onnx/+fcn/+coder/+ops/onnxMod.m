function [C, numDimsA] = onnxMod(A_, B_, fmod, numDimsA_)
% Implements the ONNX Mod operator.
%#codegen

% Copyright 2024 The MathWorks, Inc.

A          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(A_);
B          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(B_);
numDimsA   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsA_);

if fmod
    % The sign of the remainder is the same as the Dividend 
    C = A - B.*fix(A./B);
else
    % The sign of the remainder is the same as that of the Divisor.
    C = A - B.*floor(A./B);
end

% The behavior when B is zero, based on ONNX documentation, is platform
% dependent. Convert the "nan" value to 0 for further computation. 
C(isnan(C)) = 0;
end