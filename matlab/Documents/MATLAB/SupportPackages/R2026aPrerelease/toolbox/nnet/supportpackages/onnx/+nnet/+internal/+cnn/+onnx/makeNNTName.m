function NNTName = makeNNTName(ONNXLayerName)
% Make the ONNX layer name compatible with NNT. 
% * Change forward slashes to vertical bars
% ONNXLayerName is a char vector.

%   Copyright 2021 The MathWorks, Inc.    

NNTName = strrep(ONNXLayerName, '/', '|');
end