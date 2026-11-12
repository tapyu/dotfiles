function nodeTranslation = translateInstanceNormalization(~, nodeTranslation, node, ~)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "epsilon"      "FLOAT"      true    1e-5
    });
% Parse the attributes
epsilon = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if epsilon <= 0
    epsilon = double(1e-5);
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
        message('nnet_cnn_onnx:onnx:BadEpsilon'))];
end
ScaleName	= node.input{2};
BName       = node.input{3};
Y          	= node.output{1};
X       	= node.input{1};
setInitializerType(nodeTranslation.GraphTranslation,'Learnables', X);
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables', BName);
[varB, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, BName);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables', ScaleName);
[varScale, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ScaleName);
code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('if %s > 2\n', numDimsX),...
    sprintf('dataFormat = [repmat(''S'',[1 %s-2]),''CB''];\n', numDimsX),...    
    sprintf('%s = instancenorm(%s, %s, %s, ''Epsilon'', %f, ''DataFormat'', dataFormat);\n',...
    varY, varX, varB, varScale, epsilon),...
    sprintf('else\n'),...
    sprintf('%s = %s;\n', varY, varX),...
    sprintf('end\n'),...
    sprintf('%s = %s;\n', numDimsY, numDimsX);
    ];
nodeTranslation.MCode = string(code);
end

