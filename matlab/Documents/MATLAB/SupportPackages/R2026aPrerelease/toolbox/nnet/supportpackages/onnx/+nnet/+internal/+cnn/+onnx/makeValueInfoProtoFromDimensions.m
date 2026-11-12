function vip = makeValueInfoProtoFromDimensions(name, tensorProto_DataType, dims)
% Make a ValueInfoProto that describes a tensor with dimension sizes 'dims'

%   Copyright 2018-2023 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
vip = ValueInfoProto;
vip.name = name;
vip.type = TypeProto;
vip.type.tensor_type = TypeProto_Tensor;
vip.type.tensor_type.elem_type = tensorProto_DataType;
vip.type.tensor_type.shape = TensorShapeProto;
vip.type.tensor_type.shape.dim = repmat(TensorShapeProto_Dimension, 1, numel(dims));
for d = 1:numel(dims)
    if isnumeric(dims{d})
        if isnan(dims{d})
            % Interpret NaN in a DLT layer size as variable size.
            vip.type.tensor_type.shape.dim(d).dim_param = 'Unknown';
        else
            % Fixed size
            vip.type.tensor_type.shape.dim(d).dim_value = int64(dims{d});
        end
    else
        % Variable size
        vip.type.tensor_type.shape.dim(d).dim_param = char(dims{d});
    end
end
end
