function nodeTranslation = translateReduceSum(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axes"          "INTS"      true    []
    "keepdims"      "INT"       true    1
    });
% Parse the attributes
[axes, keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y	= node.output{1};
X	= node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
AxesName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Axes']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AxesName, nnet.internal.cnn.onnx.fcn.RankedArray(axes,2));
[VarAxes, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
dimsName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dims');
reduceArg = nnet.internal.cnn.onnx.fcn.uniqueCoderName('xReduced'); 
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %sprepareReduceArgs(%s, %s);\n', dimsName, getPackageName(nodeTranslation.GraphTranslation), VarAxes, NumDimX),...    
    ];
if ismember(X, IntegerTensorNames)
    command = [command, sprintf('%s = fix(sum(%s, %s));\n', reduceArg, VarX, dimsName)];
else
    command = [command, sprintf('%s = sum(%s, %s);\n', reduceArg, VarX, dimsName)];
end
if keepdims
	command = [command, sprintf('%s = %s;\n', VarY, reduceArg)];													  
    command = [command, sprintf('%s = %s;\n', NumDimY, NumDimX)];
    nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
else
    command = [command, sprintf('[%s, %s] = %sonnxSqueeze(%s, %s, %s);\n', ...
            VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), reduceArg, VarAxes, NumDimX)];
    nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];
end

nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

