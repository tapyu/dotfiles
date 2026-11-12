function vip = makeValueInfoProtoOfVariableSize(name, tensorProto_DataType, ndims)
% Make a ValueInfoProto that describes a tensor with ndims dimensions of
% unknown size

%   Copyright 2020 The MathWorks, Inc.    

import nnet.internal.cnn.onnx.*
vip = ValueInfoProto;
vip.name = name;
vip.type = TypeProto;
vip.type.tensor_type = TypeProto_Tensor;
vip.type.tensor_type.elem_type = tensorProto_DataType;
vip.type.tensor_type.shape = TensorShapeProto;
vip.type.tensor_type.shape.dim = repmat(TensorShapeProto_Dimension, 1, ndims);
for d = 1:ndims
    vip.type.tensor_type.shape.dim(d).dim_value = int64([]);
    vip.type.tensor_type.shape.dim(d).dim_param = 'Unknown';
end
end
