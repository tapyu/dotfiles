function nodeTranslation = translateGemm(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "alpha"        "FLOAT"      true    1
    "beta"         "FLOAT"      true    1
    "transA"       "INT"        true    0
    "transB"       "INT"        true    0
    "broadcast"    "INT"        true    0
    });
% Parse the attributes
[alpha, beta, transA, transB, ~] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Translate into: Y = alpha * A' * B' + beta * C, where the transposes are
% optional and indicated by transA and transB.
Y = node.output{1};
A = node.input{1};
B = node.input{2};
C = node.input{3};
AlphaName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'alpha']);
BetaName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'beta']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AlphaName, nnet.internal.cnn.onnx.fcn.RankedArray(alpha,0));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', BetaName, nnet.internal.cnn.onnx.fcn.RankedArray(beta,0));

setInitializerType(nodeTranslation.GraphTranslation,'Learnables',A);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',B);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',C);
[varA, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[varB, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[varC, numDimsC] = getVarAndNumDimName(nodeTranslation.GraphTranslation, C);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varAlpha, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AlphaName);
[varBeta, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, BetaName);
AName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('A');
BName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('B');
CName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('C');
alphaArgName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('alpha');
betaArgName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('beta');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, %s, %s] = %sprepareGemmArgs(%s, %s, %s, %s, %s, %s, %s, %s);\n', ...
    AName, BName, CName, alphaArgName, betaArgName, numDimsY,...
    getPackageName(nodeTranslation.GraphTranslation), varA, varB, varC, varAlpha, varBeta, string(transA), string(transB), numDimsC),...
    sprintf('%s = %s*%s*%s + %s*%s;\n', varY, alphaArgName, BName, AName, betaArgName, CName), ...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareGemmArgs";
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end