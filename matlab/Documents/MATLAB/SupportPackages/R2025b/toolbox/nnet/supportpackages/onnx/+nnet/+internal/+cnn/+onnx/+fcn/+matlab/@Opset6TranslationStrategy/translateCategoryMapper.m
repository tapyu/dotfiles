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
% _CatToInt is a fake Nonlearnable holding a Map from strings to integers.
% _IntToCat is a fake Nonlearnable holding a Map from integers to strings.
Cat2Int = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Cat2Int']);
Int2Cat = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Int2Cat']);
% Put each map into struct (so it's 1x1), then into a ranked array of rank 0
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Cat2Int, nnet.internal.cnn.onnx.fcn.RankedArray(struct('map',containers.Map(cats_strings, cats_int64s)),0));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Int2Cat, nnet.internal.cnn.onnx.fcn.RankedArray(struct('map',containers.Map(cats_int64s, cats_strings)),0));

[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varCat2Int, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Cat2Int);
[varInt2Cat, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Int2Cat);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxCategoryMapper(%s, %s, %s, %d, "%s", %s);\n',...
    varY, numDimsY, getPackageName(nodeTranslation.GraphTranslation), ...
    varX, varCat2Int, varInt2Cat, default_int64, default_string, numDimsX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxCategoryMapper";
nodeTranslation.IntegerOutputTensorNames = string(Y);
end

