function [ReluNode, inits] = createReluNode(opset, name, input, output)
% A helper function to create a Relu operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
ReluNode = NodeProto;
ReluNode.op_type   = 'Relu';
ReluNode.name      = name;
ReluNode.input     = input;
ReluNode.output    = output;
inits               = [];
end