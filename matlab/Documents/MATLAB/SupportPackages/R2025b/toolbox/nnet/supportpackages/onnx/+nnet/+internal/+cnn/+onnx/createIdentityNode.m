function [ReluNode, inits] = createIdentityNode(opset, name, input, output)
% A helper function to create a Identity operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
ReluNode = NodeProto;
ReluNode.op_type   = 'Identity';
ReluNode.name      = name;
ReluNode.input     = input;
ReluNode.output    = output;
inits               = [];
end