function nodeTranslation = translateDiv(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% This is a binary infix operator, but we're not calling
% translateUniBroadcastInfixBinaryOp because we need to generate
% different code when doing integer division.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
AttributeTable = cell2table({
    "axis"      "INT"   true    []
    "broadcast" "INT"   true    0
    });
% Parse the attributes
[axis, broadcast] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
D       = node.output{1};
A       = node.input{1};
B       = node.input{2};
[varA, numDimsA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[varB, numDimsB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[varD, numDimsD] = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);

Command = sprintf('%% %s:\n', node.op_type);
if broadcast
    Command = [Command,...
        sprintf('%s = %sprepareBroadcastInfixBinaryArgs(%s, %d);\n', varB, getPackageName(nodeTranslation.GraphTranslation), varB, axis)];
    nodeTranslation.IncludedFunctionNames   = "prepareBroadcastInfixBinaryArgs";
end
if ismember(A, IntegerTensorNames)
    divCommand = sprintf('%s = fix(%s ./ %s);\n', varD, varA, varB);      % integer division
    nodeTranslation.IntegerOutputTensorNames = string(D);
else
    divCommand = sprintf('%s = %s ./ %s;\n', varD, varA, varB);           % float division
end
Command = [Command,...
    divCommand,...
    sprintf('%s = max(%s, %s);\n', numDimsD, numDimsA, numDimsB),...
    ];
nodeTranslation.MCode = string(Command);


end

