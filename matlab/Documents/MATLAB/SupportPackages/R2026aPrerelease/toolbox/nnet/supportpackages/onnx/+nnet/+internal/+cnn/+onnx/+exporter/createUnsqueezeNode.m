function [unsqueezeNode, inits] = createUnsqueezeNode(opset, name, input, output, axes)
% A helper function to create an Unsqueeze operator of the specified opset
% version. 

%   Copyright 2021-2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
unsqueezeNode            = NodeProto;
unsqueezeNode.op_type    = 'Unsqueeze';
unsqueezeNode.name       = name;
unsqueezeNode.input      = input;
unsqueezeNode.output     = output;
inits = [];
if opset < 13
    unsqueezeNode.attribute = makeAttributeProto('axes', 'INTS', axes);
else
    % In Opset 13 and later, 'axes' is a node input. Make an initializer
    % for it    
    axesInit              = TensorProto;
    axesInit.name         = [name '_axes'];
    axesInit.data_type    = TensorProto_DataType.INT64;
    axesInit.raw_data     = rawData(int64(axes));
    axesInit.dims         = dimVector(numel(axes), 1);
    unsqueezeNode.input   = [unsqueezeNode.input {axesInit.name}];
    inits                 = [inits axesInit];     
end
end
