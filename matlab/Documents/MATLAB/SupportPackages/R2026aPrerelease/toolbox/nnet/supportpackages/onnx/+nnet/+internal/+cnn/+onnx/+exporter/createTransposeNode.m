function [TransposeNode, inits] = createTransposeNode(~, name, input, output, permutation)
% A helper function to create a Transpose operator of the specified opset
% version.

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
TransposeNode               = NodeProto;
TransposeNode.op_type       = 'Transpose';
TransposeNode.name          = name;
TransposeNode.input         = input;
TransposeNode.output        = output;
TransposeNode.attribute     = makeAttributeProto('perm',  'INTS',  permutation);
inits                       = [];
end