function [B, numDimsB] = onnxOneHot(ONNXIndices, Depth, Values, ONNXAxis, numDimsA)
% Implements the ONNX OneHot operator.

% Copyright 2020 The MathWorks, Inc.

Depth       = extractdata(Depth);
Values      = extractdata(Values);
DLTIndices  = extractdata(ONNXIndices) + 1;   % ONNXIndices is origin-0.
% First handle negative axes:
% ONNXaxis is origin-0. axis=-1 means add the new dimension at the end.
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsA+1;
end
DLTAxis = numDimsA+1 - ONNXAxis;     % DLT location of the new axis to be added.
% Prepare call to onehotencode:
% Add a singleton dimension in the desired location
sz          = size(DLTIndices, 1:max(2,numDimsA));
shape       = [sz(1:DLTAxis-1), 1, sz(DLTAxis:end)];
DLTIndices  = reshape(DLTIndices, shape);
% encode
B = onehotencode(DLTIndices, DLTAxis, 'double', 'ClassNames', 1:Depth);
% Set nondefault on and off values
if ~isequal(Values, [0;1])
    zeroLocs = B==0;
    B(zeroLocs) = Values(1);
    B(~zeroLocs) = Values(2);
end
B = dlarray(B);
numDimsB = numDimsA+1;
end
