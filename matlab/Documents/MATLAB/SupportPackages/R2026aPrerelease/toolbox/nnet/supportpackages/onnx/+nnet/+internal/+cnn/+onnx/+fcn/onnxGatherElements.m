function [Y, numDimsY] = onnxGatherElements(X, ONNXIndices, ONNXAxis, numDimsX, numDimsIndices)

% Copyright 2022 The MathWorks, Inc.

% Implements the ONNX GatherElements operator
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
% Convert axis to DLT axis. ONNXAxis is origin 0 and we index dimensions in
% reverse ONNX ordering
mlAxis = numDimsX - ONNXAxis;
% Convert ONNXIndices to DLT.
ONNXIndices(ONNXIndices<0) = ONNXIndices(ONNXIndices<0) + size(X,mlAxis);  % Make negative ONNXIndices nonnegative.
mlIndices = ONNXIndices + 1;
% Find the linear indices of X from which we Gather the data
mlLinearIndices = gatherElementsLinearIndices(X, mlAxis, mlIndices);
% Get the elements, then reshape to the shape of Indices
Y = reshape(X(mlLinearIndices), size(ONNXIndices));
numDimsY = numDimsIndices;

    function L = gatherElementsLinearIndices(X,dim,Indices)
        % Reduce to 0-based.
        L = Indices;
        L = L - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            L = L*prod(size(X, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of X.
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
        L = L(:) + 1;
    end

end
