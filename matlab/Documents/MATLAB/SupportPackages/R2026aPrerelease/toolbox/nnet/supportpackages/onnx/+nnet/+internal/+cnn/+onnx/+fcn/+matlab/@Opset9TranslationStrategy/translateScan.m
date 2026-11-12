function nodeTranslation = translateScan(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

AttributeTable = cell2table({
    "body"                   "GRAPH"     false   []
    "num_scan_inputs"        "INT"       false   []
    "scan_input_axes"        "INTS"      true    []
    "scan_input_directions"  "INTS"      true    []
    "scan_output_axes"       "INTS"      true    []
    "scan_output_directions" "INTS"      true    []
    });

% Parse the attributes
[body, num_scan_inputs, scan_input_axes, scan_input_directions,...
    scan_output_axes, scan_output_directions] = ...
    nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

% Get the names of inputs and outputs
numInputs = numel(node.input);
numOutputs = numel(node.output);
numStateInputs = numInputs - num_scan_inputs;
numScanOutputs = numOutputs - numStateInputs;
initialStateInputs = node.input(1:numStateInputs);
initialScanInputs = node.input(numStateInputs+1:end);
finalStateOutputs = node.output(1:numStateInputs);
finalScanOutputs = node.output(numStateInputs+1:end);

% Set the optional attributes to their defaults.
% Note: The defaults depend on num_scan_inputs, itself an
% attribute, so cannot be set automatically by
% parseNodeAttributes.
if isempty(scan_input_axes)
    scan_input_axes = zeros(1, num_scan_inputs);
end

if isempty(scan_input_directions)
    scan_input_directions = zeros(1, num_scan_inputs);
end

if isempty(scan_output_axes)
    scan_output_axes = zeros(1, numScanOutputs);
end

if isempty(scan_output_directions)
    scan_output_directions = zeros(1, numScanOutputs);
end

scanInputAxesVarName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'InputAxes']);
scanInputDirectionsVarName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'InputDirections']);
scanOutputAxesVarName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'OutputAxes']);
scanOutputDirectionsVarName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'OutputDirections']);

setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scanInputAxesVarName, nnet.internal.cnn.onnx.fcn.RankedArray(scan_input_axes, 1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scanInputDirectionsVarName, nnet.internal.cnn.onnx.fcn.RankedArray(scan_input_directions, 1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scanOutputAxesVarName, nnet.internal.cnn.onnx.fcn.RankedArray(scan_output_axes, 1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scanOutputDirectionsVarName, nnet.internal.cnn.onnx.fcn.RankedArray(scan_output_directions, 1));

% Lists of the inputs, outputs, and their ranks
[initialStateVarList, initialStateRankList] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), initialStateInputs, 'UniformOutput', false);
[initialScanVarList, initialScanRankList] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), initialScanInputs, 'UniformOutput', false);
[finalStateVarList, finalStateRankList] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), finalStateOutputs, 'UniformOutput', false);
[finalScanVarList, finalScanRankList] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), finalScanOutputs, 'UniformOutput', false);

% Translate the subgraph
BodyGraphFcnName     = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'BodyGraph']);
BodyGraphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(body, nodeTranslation.OpsetVersion, IntegerTensorNames, BodyGraphFcnName);

command = [...
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, state] = onnxScan({%s}, {%s}, {%s}, {%s}, Vars, NumDims, Training, state, @%s, %d, %d, Vars.%s, Vars.%s, Vars.%s, Vars.%s);\n', ...
    finalStateVarList, finalScanVarList, finalStateRankList, finalScanRankList, ...
    initialStateVarList, initialScanVarList, initialStateRankList, initialScanRankList,...
    BodyGraphFcnName, num_scan_inputs, numScanOutputs, ...
    scanInputAxesVarName, scanInputDirectionsVarName, scanOutputAxesVarName, scanOutputDirectionsVarName),...
    sprintf('Vars = appendStructs(Vars, state);\n')];
nodeTranslation.InitialState          	= BodyGraphTranslation.InitialState;
nodeTranslation.MCode                	= string(command);
nodeTranslation.IncludedFunctionNames	= unique(["onnxScan", BodyGraphTranslation.IncludedFunctionNames]);
nodeTranslation.Nonlearnables           = nnet.internal.cnn.onnx.fcn.appendStructs(nodeTranslation.Nonlearnables, BodyGraphTranslation.Nonlearnables);
nodeTranslation.SubgraphLearnables    	= BodyGraphTranslation.Learnables;
nodeTranslation.SubgraphFunctionCode  	= BodyGraphTranslation.GraphFunctionCode;
nodeTranslation.SubgraphTranslations  	= BodyGraphTranslation;
% Set the new IntegerOutputTensorNames to be the integer outputs of the subgraph
isIntGraphOutput                        = arrayfun(@nnet.internal.cnn.onnx.isONNXTypeInteger, nodeTranslation.SubgraphTranslations.ExternalOutputTypes);
nodeTranslation.IntegerOutputTensorNames = string(nodeTranslation.SubgraphTranslations.ExternalOutputNames(isIntGraphOutput));

% Collect all translation issues
bodyTranslationIssues                   = arrayfun(@(nodeTranslation) nodeTranslation.Issues, BodyGraphTranslation.NodeTranslations, 'UniformOutput', false);
nodeTranslation.Issues                  = [nodeTranslation.Issues, bodyTranslationIssues{:}];
end

