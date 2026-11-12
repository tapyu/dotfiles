function [dlY, numDimsY] = onnxScatterND(dlX, dlI, dlU, reduction, numDimsX, numDimsI, numDimsU)
% Function implementing the ONNX ScatterND operator

% Copyright 2021-2024 The MathWorks, Inc.

dlX = stripdims(dlX);
dlI = stripdims(dlI);
dlU = stripdims(dlU);
dlY = dlX;
numDimsY = numDimsX;
r = numDimsX;
q = numDimsI;
assert(r>=1 && q>=1)
k = size(dlI, 1);   % In ONNX, shape of I is [<tupleHolderShape>, k], so k is first in reverse-ONNX.
assert(k>0 && k<=r)
expectedURank = q + r - k - 1;
assert(numDimsU==expectedURank)
% Convert the indices stored in dlI from origin-0 to origin-1
dlI = dlI + 1;
if r==1
    % k==1. This is simple 1D assignment no matter the size of dlI and dlU
    dlY(dlI(:)) = helpReduction(dlY(dlI(:)),dlU(:),reduction);
elseif k==r
    % dlI size is [r, <tupleHolderShape>]. dlU size is
    % [<tupleHolderShape>]. Flip the first dimension of dlI to convert the
    % onnx indices to reverse-onnx indices. Flatten out tupleHolderShape in
    % dlI and dlU.
    dlI = flip(dlI,1);
    dlI = reshape(dlI, k, []);    % [k N]
    dlU = dlU(:)';                % [1 N]
    % Compute linear indices and do assignment
    for i = k:-1:1
        ICell{i} = dlI(i,:)';
    end
    linIdx = sub2ind(size(dlX), ICell{:});
    dlY(linIdx) = helpReduction(dlY(linIdx),dlU,reduction);
else
    % dlI size is [k, <tupleHolderShape>]. dlU size is [r-k,
    % <tupleHolderShape>]. Flip the first dimension of dlI to convert the
    % onnx indices to reverse-onnx indices Loop over flattened
    % tupleHolderShape and do assignments. In ONNX, each assignment sets
    % the trailing r-k dimensions of Y. In MATLAB, we set the leading r-k
    % dimensions.
    dlI = flip(dlI,1);
    dlI = reshape(dlI, k, []);              % shape is [k N]
    uCell = num2cell(size(dlX, 1:r-k));     % value is {d1 ... d_(r-k)}
    dlU = reshape(dlU, uCell{:}, []);       % shape is [d1 ... d_(r-k) N]
    % Compute linear indices and do assignment
    for i = k:-1:1
        ICell{i} = dlI(i,:)';
    end
    trailingSizeX = size(dlX, r-k+1:r);                                             % The last k dims of dlX.
    if numel(trailingSizeX)<2
        trailingSizeX = [trailingSizeX, ones(1,2-numel(trailingSizeX))];
    end
    subspaceLinIdx = sub2ind(trailingSizeX, ICell{:});                              % Linear indices as if they indexed only the last k dims.
    numelLeadingData = prod(size(dlX, 1:r-k));                                      % Total number of elements in the first r-k dimensions of dlX.
    linIdxMat = (subspaceLinIdx(:)-1)'*numelLeadingData + (1:numelLeadingData)';    % Adding a row to a column to get a matrix!
    linIdx = linIdxMat(:);
    dlY(linIdx) = helpReduction(dlY(linIdx),dlU,reduction);
end
    function reduced = helpReduction(dlY, dlU, reduction)
        switch reduction
            case "none"
                reduced = dlU;
            case "add"
                reduced = dlY + reshape(dlU,[],1);
            case "mul"
                reduced = dlY .* reshape(dlU,[],1);
            case "max"
                reduced = max(dlY, reshape(dlU,[],1));
            case "min"
                reduced = min(dlY, reshape(dlU,[],1));
        end
    end
end
