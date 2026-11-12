function [T, I, numDimsT, numDimsI] = onnxTopK(X, K, ONNXAxis, numDimsX)
% Implements the ONNX TopK operator

% Copyright 2020 The MathWorks, Inc.

if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTDim = numDimsX - ONNXAxis;
if isa(K, 'dlarray')
    K = extractdata(K);
end
[I, IlinInd] = maxklinind(extractdata(X), K, DLTDim);
% I contains indices for each subscript along the axis dim. IlinInd
% contains the linear indices into X.
T = X(IlinInd);
% T, and I (origin-0), are what the ONNX operator returns:
I = dlarray(I - 1);
numDimsT = numDimsX;
numDimsI = numDimsX;

    function [I, IlinInd] = maxklinind(A,k,dim)
        % Gather indices.
        [~,I] = maxk(A,k,dim);
        % Reduce to 0-based.
        IlinInd = I;
        IlinInd = IlinInd - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            IlinInd = IlinInd*prod(size(A, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of A.
        for d = 1:ndims(A)
            if d ~= dim
                % Add in the component from this dimension.
                idx = ((1:size(A,d))-1)*sz;
                szvec = [ones(1, d-1),  numel(idx), 1];
                IlinInd = IlinInd + reshape(idx, szvec);
            end
            % Increase the cumulative size.
            sz = sz * size(A,d);
        end
        % Shift back to being 1-based.
        IlinInd = IlinInd + 1;
    end
end
