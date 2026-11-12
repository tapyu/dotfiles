function [DivNode, inits] = createDivNode(opset, name, input, output)
% A helper function to create a Div operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
DivNode = NodeProto;
DivNode.op_type   = 'Div';
DivNode.name      = name;
DivNode.input     = input;
DivNode.output    = output;
inits               = [];
end