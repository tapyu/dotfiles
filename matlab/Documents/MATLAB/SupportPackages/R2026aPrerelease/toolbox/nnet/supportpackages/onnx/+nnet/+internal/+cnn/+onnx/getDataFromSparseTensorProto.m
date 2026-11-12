function [LinearData, ONNXShape] = getDataFromSparseTensorProto(stproto, reverseDimOrder)
% Get the data from a sparse tensor proto and return the linearized full
% data and its declared ONNX shape.
%
%   Documentation from onnx.in.proto:
%     // A serialized sparse-tensor value
%     message SparseTensorProto {
%       // The sequence of non-default values are encoded as a tensor of shape [NNZ].
%       // The default-value is zero for numeric tensors, and empty-string for string tensors.
%       optional TensorProto values = 1;
%
%       // The indices of the non-default values, which may be stored in one of two formats.
%       // (a) Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value
%       // corresponding to the j-th index of the i-th value (in the values tensor).
%       // (b) Indices can be a tensor of shape [NNZ], in which case the i-th value
%       // must be the linearized-index of the i-th value (in the values tensor).
%       // The linearized-index can be converted into an index tuple (k_1,...,k_rank)
%       // using the shape provided below.
%       // The indices must appear in ascending order without duplication.
%       // In the first format, the ordering is lexicographic-ordering:
%       // e.g., index-value [1,4] must appear before [2,1]
%       optional TensorProto indices = 2;
%
%       // The shape of the underlying dense-tensor: [dim_1, dim_2, ... dim_rank]
%       repeated int64 dims = 3;
%     }

%   Copyright 2021 The MathWorks, Inc.    

% Optional 'reverseDimOrder' flag (default true) controls whether to return
% the indices in reverse ONNX dimension ordering or forward ONNX dimension
% order.
if nargin < 2
    reverseDimOrder = true;
end
    
% 'values' is a 1D TensorProto of the nonzero values. ONNX shape is [N].
values = stproto.values;
mlValues = nnet.internal.cnn.onnx.getDataFromTensorProto(values);

% 'indices' specifies where each of the 'values' goes. A TensorProto.
%     - If indices has size [N], then element i specifies the ONNX linear
%     index in the result tensor where values(i) goes.
%     - If indices has size [N,rank], then row i specifies the ONNX
%     subscripts in the result tensor where values(i) goes.
indices = stproto.indices;
mlIndices = nnet.internal.cnn.onnx.getDataFromTensorProto(indices) + 1;

% 'ONNXShape' is the ONNX shape of the result tensor. A numeric vector.
ONNXShape = stproto.dims(:)';

% Allocate the result 'LinearData' and put the values into it.
numelData = prod(ONNXShape);
LinearData = zeros(numelData,1);
if numel(indices.dims) == 1
    % mlIndices are matlab linear indices, and the linear ordering of the
    % data is the same in ONNX and MATLAB.
    LinearData(mlIndices) = mlValues;
else
    % mlIndices are subscripts.
    % Construct arguments to pass to sub2ind
    rank = size(mlIndices,2);
    for col = 1:rank
        if reverseDimOrder
            IVars{rank-col+1} = mlIndices(:,col);   % Put dimensions in reverse onnx ordering.
        else
            IVars{col} = mlIndices(:,col);          % Put dimensions in forward onnx ordering.
        end
    end
    % Get linear indices of the sparse data points
    if reverseDimOrder
        mlShape = fliplr(ONNXShape);
    else
        mlShape = ONNXShape;
    end
    mlLinearIndices = sub2ind(mlShape, IVars{:});
    LinearData(mlLinearIndices) = mlValues;
end
end
