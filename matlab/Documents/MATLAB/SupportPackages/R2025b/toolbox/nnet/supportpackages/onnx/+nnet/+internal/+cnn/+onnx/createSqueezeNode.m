function [squeezeNode, inits] = createSqueezeNode(opset, name, input, output, axes)
% A helper function to create a Squeeze operator of the specified opset
% version. 

%   Copyright 2021 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
squeezeNode            = NodeProto;
squeezeNode.op_type    = 'Squeeze';
squeezeNode.name       = name;
squeezeNode.input      = input;
squeezeNode.output     = output;
inits = [];
if opset < 13
    squeezeNode.attribute = makeAttributeProto('axes', 'INTS', axes);
else
    % In Opset 13 and later, 'axes' is a node input. Make an initializer
    % for it    
    axesInit              = TensorProto;
    axesInit.name         = [name '_axes'];
    axesInit.data_type    = TensorProto_DataType.INT64;
    axesInit.raw_data     = rawData(int64(axes));
    axesInit.dims         = dimVector(numel(axes), 1);
    squeezeNode.input     = [squeezeNode.input {axesInit.name}];
    inits                 = [inits axesInit];     
end
end
