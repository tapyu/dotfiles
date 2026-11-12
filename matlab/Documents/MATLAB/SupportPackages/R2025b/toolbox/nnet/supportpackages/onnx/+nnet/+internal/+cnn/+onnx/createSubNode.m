function [SubNode, inits] = createSubNode(opset, name, input, output)
% A helper function to create a Div operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
SubNode = NodeProto;
SubNode.op_type   = 'Sub';
SubNode.name      = name;
SubNode.input     = input;
SubNode.output    = output;
inits               = [];
end