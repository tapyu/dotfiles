function varargout = onnxScan(initialStateVars, initialScanVars, initialStateNumDims, initialScanNumDims, Vars, NumDims, Training, state, bodyGraphFcn, numScanInputs, numScanOutputs, scanInputAxes, scanInputDirections, scanOutputAxes, scanOutputDirections)

% Copyright 2020-2024 The MathWorks, Inc.

% Convert any negative axes to >= 0
scanInputAxes = makeScanAxesNonnegative(scanInputAxes, initialScanNumDims);
scanOutputAxes = makeScanAxesNonnegative(scanOutputAxes, initialScanNumDims);

% Convert the axes from 0-indexed to 1-indexed
scanInputAxes = scanInputAxes + 1;
scanOutputAxes = scanOutputAxes + 1;

% Convert the scan input axes to the reverse axis.
% Note: Cannot yet do this for the output axes since it depends on
% the number of dimensions of the output.
scanInputAxes = [initialScanNumDims{:}] - scanInputAxes + 1;

% Output variables for loop carried dependencies and scan outputs
LCDs = initialStateVars;
numDimsLCDs = initialStateNumDims;
seqLen = size(initialScanVars{1}, scanInputAxes(1));
scanOutputs = cell(seqLen, numScanOutputs);
scanOutputsNumDims = cell(1, numScanOutputs);
% Loop over sequence
for t=1:seqLen
    inputSlices = cell(1, numel(initialScanVars));
    scanOutputsOneStep = cell(1, numScanOutputs);
    % Index each scan variable along its sequence axis
    for i=1:numel(initialScanVars)
        % Determine whether to scan forward or backward. Default 0 =
        % forward
        idx = t;
        if scanInputDirections(i) == 1 % 1 = backward
            idx = seqLen - t + 1;
        end
        % Get the MATLAB 1-indexed axis
        mlAxis = scanInputAxes(i);
        % index into data
        fullIndices = repmat({':'}, 1, initialScanNumDims{i});
        fullIndices{mlAxis} = idx;
        slice = initialScanVars{i}(fullIndices{:});
        % Reshape to an array with 1 less dimension
        sliceSize = [size(slice) ones(1, initialScanNumDims{i} - ndims(slice))];
        sliceSize(mlAxis) = [];
        if numel(sliceSize) > 1 % Don't try to reshape if sliceSize has 1 element.
            inputSlices{i} = reshape(slice, sliceSize);
        else
            inputSlices{i} = slice;
        end
    end

    % Call the subgraph
    scanNumDims = cellfun(@(x) x - 1, initialScanNumDims); % The slices being passed to the subgraph have 1 fewer dimension than the original inputs.
    [LCDs{:}, scanOutputsOneStep{:}, numDimsLCDs{:}, scanOutputsNumDims{:}, state] = bodyGraphFcn(LCDs{:}, inputSlices{:}, numDimsLCDs{:}, scanNumDims, Vars, NumDims, Training, state);

    % Collect the outputs
    for i=1:numel(scanOutputsOneStep)
        scanOutputs{t, i} = scanOutputsOneStep{i};
    end

end
if isempty(scanOutputs)
    % There were no iterations. Set ScanOutputs to the correct number of
    % empties
    finalScanOutputs = repmat({dlarray([])},1,numScanOutputs);
    scanOutputsNumDims = repmat({0}, 1, numScanOutputs);
else
    scanOutputsNumDims = cellfun(@(x) x + 1, scanOutputsNumDims, 'UniformOutput', false);
    scanOutputAxes = [scanOutputsNumDims{:}]' - scanOutputAxes + 1;
    finalScanOutputs = concatenateScanOutputs(scanOutputs, scanOutputsNumDims, scanOutputAxes, scanOutputDirections);
end
% Set function outputs
varargout = [LCDs, finalScanOutputs, numDimsLCDs, scanOutputsNumDims, {state}];

% Local subfunctions
    function finalScanOutputs= concatenateScanOutputs(scanOutputs, scanOutputsNumDims, scanOutputAxes, scanOutputDirections)
        finalScanOutputs = cell(1, size(scanOutputs,2));
        for i=1:size(scanOutputs,2)
            so = scanOutputs(:,i);
            soSize = size(so{1});
            soSize = [soSize(1:scanOutputAxes(i)-1), 1, soSize(scanOutputAxes(i):end), ones(1, scanOutputsNumDims{i} - numel(soSize) - 1)];
            soSize(scanOutputAxes(i)) = numel(so);
            if scanOutputDirections(i) == 0 % direction = 0 = append
                finalScanOutputs{i} = reshape(cat(scanOutputAxes(i), so{:}), soSize);
            else % direction = 1 = prepend
                finalScanOutputs{i} = reshape(cat(scanOutputAxes(i), so{end:-1:1}), soSize);
            end
        end
    end

    function posAxes = makeScanAxesNonnegative(axes, numDims)
        posAxes = axes;
        for i=1:numel(axes)
            if axes(i) < 0
                posAxes(i) = axes(i) + numDims{i};
            end
        end
    end
end
