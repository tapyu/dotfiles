function [DLTScales, dataFormat, Method, numDimsY] = prepareResize10Args(ONNXScales, mode, numDimsX)
% Prepares arguments for implementing the ONNX Resize-10 operator

%   Copyright 2020 The MathWorks, Inc.    

% ONNXScales are in ONNX dimension ordering
if isdlarray(ONNXScales)
    ONNXScales = extractdata(ONNXScales);
end
DLTScales = flip(ONNXScales(:)');
switch mode
    case "nearest"
        Method = "nearest";
    case "linear"
        Method = "linear";
    otherwise
        assert(false);
end
dataFormat = repmat('S', [1 numDimsX]);
numDimsY = numDimsX;
end
