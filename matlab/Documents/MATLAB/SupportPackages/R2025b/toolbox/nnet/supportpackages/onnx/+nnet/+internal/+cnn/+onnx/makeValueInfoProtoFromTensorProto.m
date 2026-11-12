function vip = makeValueInfoProtoFromTensorProto(tp)
% Make a ValueInfoProto that describes the TensorProto tp.

%   Copyright 2018-2020 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
vip = makeValueInfoProtoFromDimensions(tp.name, tp.data_type, num2cell(tp.dims));
end
