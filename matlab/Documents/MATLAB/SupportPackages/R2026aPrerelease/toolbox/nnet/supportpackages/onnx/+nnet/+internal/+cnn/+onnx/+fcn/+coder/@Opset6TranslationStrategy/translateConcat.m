function nodeTranslation = translateConcat(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       false    []
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = cat(axis,X1,X2,...)
Y       = node.output{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
numIns = numel(node.input);
inputList = cell(1,numIns);
inputRankList = cell(1,numIns);
for i = 1:numIns
    [input, inputRank] = getVarAndNumDimName(nodeTranslation.GraphTranslation,node.input{i});
    inputList{i} = input;
    inputRankList{i} = inputRank;
end
inputListStr = strjoin(inputList, ', ');
inputRankListStr = strjoin(inputRankList, ', ');

fcn = "onnxConcat";
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %s%s(%d, {%s}, [%s]);\n', VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), fcn, axis, inputListStr, inputRankListStr),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = fcn;
if ismember(node.input{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end