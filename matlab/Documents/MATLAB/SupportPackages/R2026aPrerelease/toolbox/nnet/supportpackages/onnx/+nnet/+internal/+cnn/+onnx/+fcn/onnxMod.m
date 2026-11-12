function [C, numDimsC] = onnxMod(A, B, fmod, numDimsA)

% Copyright 2024 The MathWorks, Inc.

% Implements the ONNX Mod operator.

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
numDimsC = numDimsA;
end