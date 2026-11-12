function [tensorIdx, numDimsIdx] = prepareSequenceAtArgs(seqCell, ONNXPos_)
% Prepares arguments to implement the ONNX SequenceAt operator

%#codegen

% Copyright 2024 The MathWorks, Inc.

ONNXPos = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXPos_);

len = numel(seqCell)/2;
if ONNXPos<0
    ONNXPos = ONNXPos + len;
end
tensorIdx  = ONNXPos + 1;
numDimsIdx = tensorIdx + len;
end
