function [Y, numDimsY] = onnxGatherElements(X_, ONNXIndices_, ONNXAxis_, numDimsX_, numDimsIndices_)
% Implements the ONNX GatherElements operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

X               = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
ONNXIndices     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXIndices_);
ONNXAxis        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsX        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
numDimsIndices  = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsIndices_);

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end

% Convert axis to DLT axis. ONNXAxis is origin 0 and we index dimensions in
% reverse ONNX ordering
mlAxis = numDimsX - ONNXAxis;
% Convert ONNXIndices to DLT.
coder.unroll();
for i = 1:numel(ONNXIndices)
    if ONNXIndices(i)<0
        ONNXIndices(i) = ONNXIndices(i) + size(X,mlAxis); % Make negative ONNXIndices nonnegative.
    end
end
mlIndices = ONNXIndices + 1;
% Find the linear indices of X from which we Gather the data
mlLinearIndices = iGatherElementsLinearIndices(X, mlAxis, mlIndices);
% Get the elements, then reshape to the shape of Indices
Y = reshape(X(mlLinearIndices), size(ONNXIndices));
numDimsY = numDimsIndices;

    function L_ = iGatherElementsLinearIndices(X,dim,Indices)
        % Reduce to 0-based.
        L = Indices;
        L = L - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            L = L*prod(size(X, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of X.
        coder.unroll();
        for d = 1:ndims(X)
            if d ~= dim
                % Add in the component from this dimension.
                idx = ((1:size(Indices,d))-1)*sz;                   % Note we take the size of Indices here, not X.
                szvec = [ones(1, d-1),  numel(idx), 1];
                L = L + reshape(idx, szvec);
            end
            % Increase the cumulative size.
            sz = sz * size(X,d);
        end
        % Shift back to being 1-based.
        L_ = L(:) + 1;
    end
end