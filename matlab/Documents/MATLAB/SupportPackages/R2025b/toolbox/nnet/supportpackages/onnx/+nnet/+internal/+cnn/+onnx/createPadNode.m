function [padNode, inits] = createPadNode(opset, name, input, output, pads, mode, value)
% A helper function to create a Pad operator of the specified opset
% version. 

%   Copyright 2021-2024 The MathWorks, Inc.
import nnet.internal.cnn.onnx.*
padNode            = NodeProto;
padNode.op_type    = 'Pad';
padNode.name       = name;
padNode.input      = input;
padNode.output     = output;    
padNode.attribute  = makeAttributeProto('mode','STRING',mode);
inits              = [];
if opset < 11 
    padNode.attribute = [padNode.attribute, makeAttributeProto('pads','INTS',pads)];
    if strcmp(mode, 'constant') % Set 'value'
        padNode.attribute = [padNode.attribute, makeAttributeProto('value','FLOAT',value)];
    end
else
    % In Opset 11 and later, 'pads' and 'constant_value' are node inputs.
    % Create initalizers for them.
    % pads could be either "char", referring a tensor, or a list of
    % scalars.
    if ischar(pads)
        padNode.input = [padNode.input {pads}];
    else
        padsInit              = TensorProto;
        padsInit.name         = [name '_pads'];
        padsInit.data_type    = TensorProto_DataType.INT64;
        padsInit.raw_data     = rawData(int64(pads));
        padsInit.dims         = dimVector(numel(pads),1);
        padNode.input         = [padNode.input {padsInit.name}];
        inits                 = [inits padsInit];
    end
    if strcmp(mode, 'constant') % Pass constant_value
        valueInit              = TensorProto;
        valueInit.name         = [name '_value'];
        valueInit.data_type    = TensorProto_DataType.FLOAT;
        valueInit.raw_data     = rawData(single(value));
        valueInit.dims         = [];
        padNode.input          = [padNode.input {valueInit.name}];
        inits                  = [inits valueInit];
    end
end
