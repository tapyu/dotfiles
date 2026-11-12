function [AddNode, inits] = createAddNode(opset, name, input, output)
% A helper function to create a Add operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
AddNode = NodeProto;
AddNode.op_type   = 'Add';
AddNode.name      = name;
AddNode.input     = input;
AddNode.output    = output;
inits               = [];
end