function attributeProto = makeAttributeProto(name, Type, value)
% name is a char array.
% type is a char array.
% value is a type corresponding to 'type' 

%   Copyright 2020-2021 The MathWorks, Inc.    

import nnet.internal.cnn.onnx.*
assert(ischar(name));
assert(ismember(upper(Type), {'FLOAT', 'INT', 'STRING', 'TENSOR', 'SPARSE_TENSOR', 'GRAPH', ...
    'FLOATS', 'INTS', 'STRINGS', 'TENSORS', 'SPARSE_TENSORS', 'GRAPHS'}));

attributeProto      = AttributeProto;
attributeProto.name = name;
attributeProto.type = AttributeProto_AttributeType.(upper(Type));
switch attributeProto.type
    case AttributeProto_AttributeType.FLOAT
        assert(isnumeric(value) && isscalar(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.f = single(value);                                   % ONNX IR spec says "float" is a 32-bit value.
    case AttributeProto_AttributeType.INT
        assert(isnumeric(value) && isscalar(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.i = int64(value);
    case AttributeProto_AttributeType.STRING
        assert(ischar(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.s = value;
    case AttributeProto_AttributeType.TENSOR
        assert(isa(value, 'nnet.internal.cnn.onnx.TensorProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.t = value;
    case AttributeProto_AttributeType.SPARSE_TENSOR
        assert(isa(value, 'nnet.internal.cnn.onnx.SparseTensorProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.sparse_tensor = value;
    case AttributeProto_AttributeType.GRAPH
        assert(isa(value, 'nnet.internal.cnn.onnx.GraphProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.g = value;
    case AttributeProto_AttributeType.FLOATS
        assert(isnumeric(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));   % ONNX IR spec says "float" is a 32-bit value.
        attributeProto.floats = single(value);
    case AttributeProto_AttributeType.INTS
        assert(isnumeric(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.ints = int64(value);
    case AttributeProto_AttributeType.STRINGS
        assert(iscellstr(value),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.strings = value;
    case AttributeProto_AttributeType.TENSORS
        assert(isa(value, 'nnet.internal.cnn.onnx.TensorProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.tensors = value;
    case AttributeProto_AttributeType.SPARSE_TENSORS
        assert(isa(value, 'nnet.internal.cnn.onnx.SparseTensorProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.sparse_tensors = value;
    case AttributeProto_AttributeType.GRAPHS
        assert(isa(value, 'nnet.internal.cnn.onnx.GraphProto'),...
            message('nnet_cnn_onnx:onnx:InvalidAttributeValue', name, Type));
        attributeProto.graphs = value;
    otherwise
        error(message('nnet_onnx_cnn:UnsupportedAttributeType', name, Type));
end
end
