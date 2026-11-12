function nodeTranslation = translateClip(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "max"  	"FLOAT"     true    Inf
    "min" 	"FLOAT"  	true    -Inf
    });
% Parse the attributes
[Max, Min] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
MinName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'min']);
MaxName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'max']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', MinName, nnet.internal.cnn.onnx.fcn.RankedArray(Min,0));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', MaxName, nnet.internal.cnn.onnx.fcn.RankedArray(Max,0));
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varMin, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, MinName);
[varMax, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, MaxName);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = min(%s, max(%s, %s));\n', varY, varMax, varMin, varX),...
    sprintf('%s = %s;\n', numDimsY, numDimsX),...
    ];

nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

