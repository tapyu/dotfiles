function [clipNode, inits] = createClipNode(opset, name, input, output, clipMin, clipMax)
% A helper function to create a Clip operator of the specified opset
% version. 

%   Copyright 2021-2025 The MathWorks, Inc.
import nnet.internal.cnn.onnx.*
clipNode            = NodeProto;
clipNode.op_type    = 'Clip';
clipNode.name       = name;
clipNode.input      = input;
clipNode.output     = output;
inits = [];
if opset < 11
    if ~isempty(clipMin)
        clipNode.attribute = [clipNode.attribute, makeAttributeProto('min', 'FLOAT', clipMin)];
    end
    if ~isempty(clipMax)
        clipNode.attribute = [clipNode.attribute, makeAttributeProto('max', 'FLOAT', clipMax)];
    end
else
    % In Opset 11 and later, 'min' and 'max' are optional node inputs. Create
    % initializers for them.
    if ~isempty(clipMin)
        minInit              = TensorProto;
        minInit.name         = [name '_min'];
        minInit.data_type    = TensorProto_DataType.FLOAT;
        minInit.raw_data     = rawData(single(clipMin));
        minInit.dims         = []; % scalar
        clipNode.input       = [clipNode.input {minInit.name}];
        inits                = [inits minInit];
    end
    if ~isempty(clipMax)
        maxInit              = TensorProto;
        maxInit.name         = [name '_max'];
        maxInit.data_type    = TensorProto_DataType.FLOAT;
        maxInit.raw_data     = rawData(single(clipMax));
        maxInit.dims         = []; % scalar
        clipNode.input       = [clipNode.input {maxInit.name}];        
        inits                = [inits maxInit];
    end
end
end