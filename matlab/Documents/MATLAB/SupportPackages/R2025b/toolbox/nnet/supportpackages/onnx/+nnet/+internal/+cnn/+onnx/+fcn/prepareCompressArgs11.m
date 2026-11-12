function [Indices, numDimsY] = prepareCompressArgs11(X, condition, ONNXAxis, numDimsX)
% Prepares arguments for implementing the ONNX Compress operator

%   Copyright 2020-2024 The MathWorks, Inc.    

if isempty(ONNXAxis)
    % No axis was specified, so X will be flattened first
    Indices = [extractdata(condition); false(numel(X) - numel(condition), 1)];
    numDimsY = 1;
else
    % Handle negative axis
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsX;
    end
    DLTAxis = numDimsX - ONNXAxis;
    condition = [extractdata(condition); false(size(X, DLTAxis) - numel(condition), 1)];
    Indices          = repmat({':'}, 1, numDimsX);
    Indices{DLTAxis} = condition;
    numDimsY = numDimsX;
end
end
