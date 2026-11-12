function [poolsize, stride, padding, dataFormat, numDimsY, numDimsIndices] = prepareMaxPool8Args(poolsize_, stride_, padding_, numDimsX_)
% Prepares arguments for implementing the ONNX MaxPool-8 operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

poolsize__   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(poolsize_(:)');
stride__     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(stride_(:)');
padding__    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(padding_);
numDimsX     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

poolsize    = fliplr(poolsize__);
stride      = fliplr(stride__);
if isnumeric(padding__)
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the lrflip and semicolon)
    padding = fliplr(transpose(reshape(padding__, [], 2)));
else
    padding = padding__;
end

dataFormat  = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY    = numDimsX;
numDimsIndices = numDimsX;                      % New in opset 8
end
