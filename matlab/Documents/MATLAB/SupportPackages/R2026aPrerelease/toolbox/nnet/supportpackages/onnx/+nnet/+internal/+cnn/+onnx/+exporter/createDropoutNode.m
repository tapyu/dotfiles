function [dropoutNode, inits] = createDropoutNode(opset, name, input, output, ratio)
% A helper function to create a Dropout operator of the specified opset
% version. 

%   Copyright 2021-2025 The MathWorks, Inc.
import nnet.internal.cnn.onnx.*
dropoutNode            = NodeProto;
dropoutNode.op_type    = 'Dropout';
dropoutNode.name       = name;
dropoutNode.input      = input;
dropoutNode.output     = output;
inits                  = [];

% Set 'is_test' for early opsets
if opset < 7
    dropoutNode.attribute = makeAttributeProto('is_test', 'INT', 1);
end

% Set 'ratio'
if opset < 12
    dropoutNode.attribute = [dropoutNode.attribute makeAttributeProto('ratio', 'FLOAT', ratio)];
else
    % In Opset 12 and later, 'ratio' is a node input. Make an initializer
    % for it.
    ratioInit              = TensorProto;
    ratioInit.name         = [name '_ratio'];
    ratioInit.data_type    = TensorProto_DataType.FLOAT;
    ratioInit.raw_data     = rawData(single(ratio));
    ratioInit.dims         = []; % scalar
    dropoutNode.input      = [dropoutNode.input {ratioInit.name}];
    inits                  = [inits ratioInit];    
end
end
