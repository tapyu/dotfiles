function [ShapeNode, inits] = createShapeNode(opset, name, input, output)
% A helper function to create a Shape operator of the specified opset
% version. Does not support the 'start' and 'end' inputs.

%   Copyright 2022 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
ShapeNode = NodeProto;
ShapeNode.op_type   = 'Shape';
ShapeNode.name      = name;
ShapeNode.input     = input;
ShapeNode.output    = output;
inits               = [];
end