function varargout = onnxLoop(M, cond, v_initial, numDimsM, numDimsCond, numDimsV_initial, Vars, NumDims, Training, state, bodyGraphFcn, numScanOutputs)

% Copyright 2020-2025 The MathWorks, Inc.

% Implements the ONNX Loop operator
if isempty(M)
    M = Inf;
end
if isempty(cond)
    cond = true;        % cond = Whether to keep going.
end
LCDs = v_initial;                            	% Loop carried dependencies
numDimsLCDs = numDimsV_initial;
ScanOutputsOneIter = cell(1,numScanOutputs);    % The scan outputs for one iteration.
numDimsScanOutputsOneIter = cell(1,numScanOutputs);
% Run the loop
i = 0;
ScanOutputs = {};
while i<M && cond
    % Call the subgraph. According to ONNX Doc: "The graph run each
    % iteration. It has 2+N inputs: (iteration_num, condition, loop carried
    % dependencies...). It has 1+N+K outputs: (condition, loop carried
    % dependencies..., scan_outputs...). Each scan_output is created by
    % concatenating the value of the specified output value at the end of
    % each iteration of the loop. It is an error if the dimensions or data
    % type of these scan_outputs change across loop iterations."
    
    % So our subgraph function will have outputs:
    %
    % [condition, LCDs..., scan_outputs..., numDimsCondition, numDimsLCDs..., numDimsScan_outputs..., state]
    %
    % and inputs:
    %
    % (iteration_num, condition, LCDs..., numDimsIteration_num, numDimsCondition,
    % numDimsLCDs..., Vars, NumDims, Vars, NumDims, Training, state, State)
    %
    [cond, LCDs{:}, ScanOutputsOneIter{:}, numDimsCond, numDimsLCDs{:}, numDimsScanOutputsOneIter{:}, state] = bodyGraphFcn(...
        i, cond, LCDs{:}, numDimsM, numDimsCond, numDimsLCDs{:}, Vars, NumDims, Training, state);
    % Update ScanOutputs
    for k=1:numScanOutputs
        ScanOutputs{k}{i+1} = ScanOutputsOneIter{k};
    end
    i = i + 1;
end
if isempty(ScanOutputs)
    % There were no iterations. Set ScanOutputs to the correct number of
    % empties
    ScanOutputs = repmat({dlarray(single([]))},1,numScanOutputs);
    numDimsScanOutputs = repmat({single(0)},1,numScanOutputs);
else
    ScanOutputs = concatenateScanOutputs(ScanOutputs, numDimsScanOutputsOneIter);
    % Each scan output has numDims 1 higher than the numDims of the value returned in a single iteration:
    numDimsScanOutputs = cellfun(@(R)R+1, numDimsScanOutputsOneIter, 'UniformOutput', false);
end
% Set function outputs
v_final_and_scan_outputs        = [LCDs, ScanOutputs];
numDimsV_final_and_scan_outputs = [numDimsLCDs, numDimsScanOutputs];
varargout                       = [v_final_and_scan_outputs, numDimsV_final_and_scan_outputs, state];

    function ConcatenatedScanOutputs = concatenateScanOutputs(ScanOutputs, numDimsScanOutputs)
        % ScanOutputs{j} is a cell array of tensors which should be concatenated to
        % produce ConcatenatedScanOutputs{j}. According to the ONNX Doc equivalent
        % C-code, "user_defined_vals[j] = user_defined_val", which we interpret to
        % mean: "Concatenate ScanOutputs{j} along a new, leading dimension."
        %   Because DLT tensors have reversed dimension ordering, this means we
        %   should concatenate ScanOutputs{j} along a new, trailing dimension.
        ConcatenatedScanOutputs = ScanOutputs;
        for j=1:numel(ScanOutputs)
            % Concatenate ScanOutputs{j}
            toConcat = ScanOutputs{j};          % toConcat is a cell array.
            eltNumDims = numDimsScanOutputs{j};
            if eltNumDims > 0
                % Add a trailing dimension to each element to be concatenated
                newEltSize = [size(toConcat{1},1:eltNumDims) 1];
                toConcat = cellfun(@(x)reshape(x, newEltSize), toConcat, 'UniformOutput', false);
            end
            % concatenate along last dimension
            lastDim = eltNumDims+1;
            ConcatenatedScanOutputs{j} = cat(lastDim, toConcat{:});    % dlarray/cat
        end
    end
end
