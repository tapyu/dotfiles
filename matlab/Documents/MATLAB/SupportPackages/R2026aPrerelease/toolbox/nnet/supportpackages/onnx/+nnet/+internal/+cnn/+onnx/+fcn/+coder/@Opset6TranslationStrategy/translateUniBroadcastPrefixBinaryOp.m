function nodeTranslation = translateUniBroadcastPrefixBinaryOp(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames)
%   Copyright 2025-2026 The MathWorks, Inc.

% Undirectionally broadcast (left to right) prefix binary
% operator with attributes 'broadcast' and 'axis'
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
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarD, NumDimD]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
BTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('B');

Command = sprintf('%% %s:\n', node.op_type);
if broadcast && ~isempty(axis)
    Command = [Command,...
        sprintf('%s = %sprepareBroadcastInfixBinaryArgs(%s, %s, %d);\n', BTempName, ...
        getPackageName(nodeTranslation.GraphTranslation), VarB, NumDimA, axis)];
    nodeTranslation.IncludedFunctionNames   = "prepareBroadcastInfixBinaryArgs";
else
    Command = [Command,...
        sprintf('%s = %s;\n', BTempName, VarB)];
end
Command = [Command,...
    sprintf('%s = %s(%s, %s);\n', VarD, MATLABFcn, VarA, BTempName),...
    sprintf('%s = max(%s, %s);\n', NumDimD, NumDimA, NumDimB),...
    ];
nodeTranslation.MCode = string(Command);
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(D);
end
end