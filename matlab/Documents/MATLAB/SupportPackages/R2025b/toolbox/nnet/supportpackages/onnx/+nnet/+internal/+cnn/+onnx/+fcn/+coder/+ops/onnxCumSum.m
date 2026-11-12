function [Y, numDimsX] = onnxCumSum(X_, ONNXAxis_, exclusive_, reverse_, numDimsX_)
% Implements the ONNX CumSum operator. It needs to work on dlarrays and be
% differentiable. dlarray has no cumsum method. To achieve
% differentiability, this implementation performs the cumsum in a loop of
% dlarray "+" operations.
%#codegen

% Copyright 2024 The MathWorks, Inc.

X           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
ONNXAxis    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
exclusive   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(exclusive_);
reverse     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(reverse_);
numDimsX    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

Y = zeros(size(X));

% ONNXaxis is origin 0.
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
D = size(X, DLTAxis);
sizeX = size(X);

if reverse
    % reverse
    Y = iComputeReverseCumSum(X, Y, D, DLTAxis, numDimsX, sizeX, exclusive);
else  
    % not reverse
    Y = iComputeCumSum(X, Y, D, DLTAxis, numDimsX, sizeX, exclusive);
end

    % sub-functions
    function Y = iComputeReverseCumSum(X, Y, D, DLTAxis, numDimsX, sizeX, exclusive)
        % Set the Dth element:
        sD = iMakeIndices(D, DLTAxis, numDimsX, sizeX);
        Y(sD{:}) = X(sD{:});                                          % Y(:,:,D,:,:) = X(:,:,D,:,:)
        % Compute the cumsum downward from D-1 to 1
        coder.unroll();
        for i = D-1:-1:1
            s = iMakeIndices(i, DLTAxis, numDimsX, sizeX);
            sPrev = iMakeIndices(i+1, DLTAxis, numDimsX, sizeX);
            Y(s{:}) = Y(sPrev{:}) + X(s{:});                          % Y(:,:,i,:,:) = Y(:,:,i+1,:,:) + X(:,:,i,:,:)
        end
        % Shift down if exclusive
        if exclusive
            sTo = iMakeIndices(1:D-1, DLTAxis, numDimsX, sizeX);
            sFrom = iMakeIndices(2:D, DLTAxis, numDimsX, sizeX);
            Y(sTo{:}) = Y(sFrom{:});                                  % Y(:,:,1:D-1,:,:) = Y(:,:,2:D,:,:)
            Y(sD{:}) = 0;                                             % Y(:,:,D,:,:) = 0
        end
    end

    function Y = iComputeCumSum(X, Y, D, DLTAxis, numDimsX, sizeX, exclusive)
         % Set the 1st element
        s1 = iMakeIndices(1, DLTAxis, numDimsX, sizeX);
        Y(s1{:}) = X(s1{:});                                          % Y(:,:,1,:,:) = X(:,:,1,:,:)
        % Compute the cumsum upward from 2 to D
        coder.unroll();
        for i = 2:D
            s = iMakeIndices(i, DLTAxis, numDimsX, sizeX);
            sPrev = iMakeIndices(i-1, DLTAxis, numDimsX, sizeX);
            Y(s{:}) = Y(sPrev{:}) + X(s{:});                          % Y(:,:,i,:,:) = Y(:,:,i-1,:,:) + X(:,:,i,:,:)
        end
        % Shift up if exclusive
        if exclusive
            sFrom = iMakeIndices(1:D-1, DLTAxis, numDimsX, sizeX);
            sTo = iMakeIndices(2:D, DLTAxis, numDimsX, sizeX);
            Y(sTo{:}) = Y(sFrom{:});                                  % Y(:,:,2:D,:,:) = Y(:,:,1:D-1,:,:)
            Y(s1{:}) = 0;                                             % Y(:,:,1,:,:) = 0
        end
    end     

    function s = iMakeIndices(idx, DLTAxis, numDims, sizeX_)
        % Make a list index (:,:,i,:,:), where the i is
        % placed on axis DLTAxis. i can be a single integer of a range
        % like 1:N or 1:2:N
        s = cell(1, numDims);
		if numDims > numel(sizeX_)
			szX = [sizeX_ ones(1, numDims - numel(sizeX_))];
		else
			szX = sizeX_;
		end
        coder.unroll();
        for j = 1:numDims
            if j ~= DLTAxis
                s{j} = 1 : szX(j);
            else
                s{j} = idx;
            end
        end        
    end
end
