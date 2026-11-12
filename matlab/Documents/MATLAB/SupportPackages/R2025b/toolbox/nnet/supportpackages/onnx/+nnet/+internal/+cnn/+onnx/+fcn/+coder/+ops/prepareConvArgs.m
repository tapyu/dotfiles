function [weights, bias, stride, dilationFactor, padding, dataFormat, numDimsY] = prepareConvArgs(...
    ONNXWeights_, bias_, stride_, dilationFactor_, padding_, numWtGroups_, numDimsX_, numDimsW_)
% Prepares arguments for implementing the ONNX Conv operator
%#codegen

    % Copyright 2024 The MathWorks, Inc.    
    ONNXWeights        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXWeights_);   
    bias__             = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(bias_);  
    stride__           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(stride_); 
    dilation__         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dilationFactor_); 
    pads__             = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(padding_); 
    numWtGroups        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numWtGroups_);    
    numDimsX           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_); 
    numDimsW           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsW_);

% Weights: The ONNX weight dim is Fcxyz..., where c=C/G, G is numGroups,
% and xyz... are spatial dimensions. DLT "weights" here is the flip of
% that, or ...zyxcF. dlconv requires ...zyxcfG, where f=F/G. So reshape to
% split the last dimension.
sizeW    = size(ONNXWeights, 1:numDimsW);
F        = sizeW(end);
newWSize = [sizeW(1:numDimsW-1), F/numWtGroups, numWtGroups];
weights  = reshape(ONNXWeights, newWSize);
% bias
if isempty(bias__)
    bias___ = 0;
else
    bias___ = bias__;
end

bias = dlarray(bias___(:),'CU');
% Derive missing default attributes from weight tensor
numSpatialDims = numDimsW - 2;
if isempty(pads__)
    pads___ = zeros(1, 2*numSpatialDims);
else
    pads___ = pads__;
end
if isempty(stride__)
    stride___ = ones(1,numSpatialDims);
else
    stride___ = stride__;
end
if isempty(dilation__)
    dilation___ = ones(1,numSpatialDims);
else
    dilation___ = dilation__;
end

% Make the attributes double row vectors, and flip their dimension ordering
% to reverse-onnx:
stride = coder.const(fliplr(double(stride___(:)')));
dilationFactor = coder.const(fliplr(double(dilation___(:)')));
if isnumeric(pads___)       % padding can be "same"
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the lrflip and semicolon)
    padding = coder.const(fliplr(transpose(reshape(pads___, [], 2))));
else
    padding = coder.const(pads___);
end

% Set dataformat and numdims
dataFormat = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY = numDimsX;
end
