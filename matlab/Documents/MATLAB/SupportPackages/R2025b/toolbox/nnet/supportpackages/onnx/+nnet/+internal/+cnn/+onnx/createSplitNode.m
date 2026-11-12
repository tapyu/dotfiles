function [splitNode, inits] = createSplitNode(opset, name, input, output, axis, split)
% A helper function to create a Split operator of the specified opset
% version. 

%   Copyright 2021 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
splitNode            = NodeProto;
splitNode.op_type    = 'Split';
splitNode.name       = name;
splitNode.input      = input;
splitNode.output     = output;
splitNode.attribute  = makeAttributeProto('axis', 'INT', axis);
inits                  = [];
if opset < 13
    splitNode.attribute = [splitNode.attribute makeAttributeProto('split', 'INTS', split)];
else
    % In Opset 13 and later, 'split' is a node input. Make an initializer
    % for it.
    splitInit              = TensorProto;
    splitInit.name         = [name '_split'];
    splitInit.data_type    = TensorProto_DataType.INT64;
    splitInit.raw_data     = rawData(int64(split));
    splitInit.dims         = dimVector(numel(split), 1);
    splitNode.input        = [splitNode.input {splitInit.name}];
    inits                  = [inits splitInit];     
end
end
