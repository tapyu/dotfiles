function [AddNode, inits] = createAddNode(opset, name, input, output)
% A helper function to create a Add operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
AddNode            = NodeProto;
AddNode.op_type    = 'Add';
AddNode.name       = name;
AddNode.input      = input;
AddNode.output     = output;
inits = [];
if opset < 7
    AddNode.attribute = makeAttributeProto('broadcast', 'INT', 1);
end
end
