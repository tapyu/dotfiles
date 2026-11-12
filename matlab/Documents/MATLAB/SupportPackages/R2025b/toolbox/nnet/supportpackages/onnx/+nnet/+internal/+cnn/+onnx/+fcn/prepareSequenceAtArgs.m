function [tensorIdx, numDimsIdx] = prepareSequenceAtArgs(seqCell, ONNXPos)
% Prepares arguments to implement the ONNX SequenceAt operator

%   Copyright 2020 The MathWorks, Inc.    

len = numel(seqCell)/2;
if ONNXPos<0
    ONNXPos = ONNXPos + len;
end
tensorIdx  = ONNXPos + 1;
numDimsIdx = tensorIdx + len;
end
