function [node, inits] = createSISONode(~, op_type, name, input, output)
% A helper function to create a {op_type} operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
node = NodeProto;
node.op_type   = op_type;
node.name      = name;
node.input     = input;
node.output    = output;
inits          = [];
end