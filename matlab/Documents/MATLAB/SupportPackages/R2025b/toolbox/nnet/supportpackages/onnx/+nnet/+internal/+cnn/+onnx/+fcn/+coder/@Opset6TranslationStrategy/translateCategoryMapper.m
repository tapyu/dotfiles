function nodeTranslation = translateCategoryMapper(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#aionnxmlcategorymapper
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "cats_int64s"       "INTS"      false	[]
    "cats_strings"      "STRINGS"	false	[]
    "default_int64"     "INT"       true    -1
    "default_string"	"STRING"	true    '_Unused'
    });
% Parse the attributes
[cats_int64s, cats_strings, default_int64, default_string] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = CategoryMapper(X)
Y = node.output{1};
X = node.input{1};

Strings = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Strings']);
Ints = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Ints']);

% Put each map into struct (so it's 1x1), then into a ranked array of rank 0
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Strings, nnet.internal.cnn.onnx.fcn.RankedArray(cellstr(cats_strings),1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Ints, nnet.internal.cnn.onnx.fcn.RankedArray(cats_int64s,1));

[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varStrings, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Strings);
[varInts, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Ints);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxCategoryMapper(%s, %s, %s, %d, "%s", %s);\n',...
    varY, numDimsY, getPackageName(nodeTranslation.GraphTranslation), ...
    varX, varStrings, varInts, default_int64, default_string, numDimsX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxCategoryMapper";
nodeTranslation.IntegerOutputTensorNames = string(Y);
end

