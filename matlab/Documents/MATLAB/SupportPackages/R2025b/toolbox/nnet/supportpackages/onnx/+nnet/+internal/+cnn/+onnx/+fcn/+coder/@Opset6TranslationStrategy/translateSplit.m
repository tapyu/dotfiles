function nodeTranslation = translateSplit(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    0
    "split"     "INTS"      true    []
    });
% Parse the attributes
[axis, split] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% optional split attribute. If missing, "the tensor is split to equal sized
% parts" (one per output variable).
if isempty(split)
    numSplits = numel(node.output);
else
    numSplits = 0;
end
SplitName  	= nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Split']);
numOuts = numel(node.output);
outputList = cell(1,numOuts);
outputRankList = cell(1,numOuts);
for i = 1:numOuts
    [output, outputRank] = getVarAndNumDimName(nodeTranslation.GraphTranslation,node.output{i});
    outputList{i} = output;
    outputRankList{i} = outputRank;
end
outputListStr = strjoin(outputList, ', ');
outputRankListStr = strjoin(outputRankList, ', ');
X         	= node.input{1};
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', SplitName, nnet.internal.cnn.onnx.fcn.RankedArray(split,2));

[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarSplit, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, SplitName);
fcn = "onnxSplit";
code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %s%s(%s, %d, %s, %d, %s);\n', ...
    outputListStr, outputRankListStr, getPackageName(nodeTranslation.GraphTranslation), ...
    fcn, VarX, axis, VarSplit, numSplits, NumDimX),...
    ];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = fcn;
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(node.output);
end
end