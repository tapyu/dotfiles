function [GlobalMaxPoolNode, inits] = createGlobalMaxPoolNode(~, name, input, output)
% A helper function to create a GlobalMaxPool operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
GlobalMaxPoolNode = NodeProto;
GlobalMaxPoolNode.op_type   = 'GlobalMaxPool';
GlobalMaxPoolNode.name      = name;
GlobalMaxPoolNode.input     = input;
GlobalMaxPoolNode.output    = output;
inits                = [];
end