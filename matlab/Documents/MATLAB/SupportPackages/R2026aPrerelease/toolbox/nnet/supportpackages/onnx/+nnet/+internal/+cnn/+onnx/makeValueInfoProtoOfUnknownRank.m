function vip = makeValueInfoProtoOfUnknownRank(name, tensorProto_DataType)
% Make a ValueInfoProto that describes a tensor with unknown rank

%   Copyright 2021 The MathWorks, Inc.    

import nnet.internal.cnn.onnx.*
vip = ValueInfoProto;
vip.name = name;
vip.type = TypeProto;
vip.type.tensor_type = TypeProto_Tensor;
vip.type.tensor_type.elem_type = tensorProto_DataType;
vip.type.tensor_type.shape = [];                        % The shape field of TypeProto.Tensor is optional.
end
