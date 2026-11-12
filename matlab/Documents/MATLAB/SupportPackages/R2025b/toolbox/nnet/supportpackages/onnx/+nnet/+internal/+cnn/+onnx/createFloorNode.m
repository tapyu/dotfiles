function [FloorNode, inits] = createFloorNode(opset, name, input, output)
% A helper function to create a Floor operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
FloorNode = NodeProto;
FloorNode.op_type   = 'Floor';
FloorNode.name      = name;
FloorNode.input     = input;
FloorNode.output    = output;
inits               = [];
end