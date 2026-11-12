function [Y, numDimsX] = onnxCumSum(X, ONNXAxis, exclusive, reverse, numDimsX)
% Implements the ONNX CumSum operator. It needs to work on dlarrays and be
% differentiable. dlarray has no cumsum method. To achieve
% differentiability, this implementation performs the cumsum in a loop of
% dlarray "+" operations.

% Copyright 2021-2024 The MathWorks, Inc.

Y = zeros(size(X), 'like', X);
% ONNXaxis is origin 0.
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
D = size(X, DLTAxis);
if reverse
    % Set the Dth element:
    sD = iMakeIndices(D, DLTAxis, numDimsX);
    Y(sD{:}) = X(sD{:});                                          % Y(:,:,D,:,:) = X(:,:,D,:,:)
    % Compute the cumsum downward from D-1 to 1
    for i = D-1:-1:1
        s = iMakeIndices(i, DLTAxis, numDimsX);
        sPrev = iMakeIndices(i+1, DLTAxis, numDimsX);
        Y(s{:}) = Y(sPrev{:}) + X(s{:});                          % Y(:,:,i,:,:) = Y(:,:,i+1,:,:) + X(:,:,i,:,:)
    end
    % Shift down if exclusive
    if exclusive
        sTo = iMakeIndices(1:D-1, DLTAxis, numDimsX);
        sFrom = iMakeIndices(2:D, DLTAxis, numDimsX);
        Y(sTo{:}) = Y(sFrom{:});                                  % Y(:,:,1:D-1,:,:) = Y(:,:,2:D,:,:)
        Y(sD{:}) = 0;                                             % Y(:,:,D,:,:) = 0
    end
else  % not reverse
    % Set the 1st element
    s1 = iMakeIndices(1, DLTAxis, numDimsX);
    Y(s1{:}) = X(s1{:});                                          % Y(:,:,1,:,:) = X(:,:,1,:,:)
    % Compute the cumsum upward from 2 to D
    for i = 2:D
        s = iMakeIndices(i, DLTAxis, numDimsX);
        sPrev = iMakeIndices(i-1, DLTAxis, numDimsX);
        Y(s{:}) = Y(sPrev{:}) + X(s{:});                          % Y(:,:,i,:,:) = Y(:,:,i-1,:,:) + X(:,:,i,:,:)
    end
    % Shift up if exclusive
    if exclusive
        sFrom = iMakeIndices(1:D-1, DLTAxis, numDimsX);
        sTo = iMakeIndices(2:D, DLTAxis, numDimsX);
        Y(sTo{:}) = Y(sFrom{:});                                  % Y(:,:,2:D,:,:) = Y(:,:,1:D-1,:,:)
        Y(s1{:}) = 0;                                             % Y(:,:,1,:,:) = 0
    end
end

    function s = iMakeIndices(i, DLTAxis, numDims)
        % Make a list index (:,:,i,:,:), where the i is
        % placed on axis DLTAxis. i can be a single integer of a range
        % like 1:N or 1:2:N
        s = repmat({':'}, 1, numDims);
        s{DLTAxis} = i;
    end
end
